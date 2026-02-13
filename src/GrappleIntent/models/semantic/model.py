"""Semantic path model — intent classification via Vision-Language Transformer.

Architecture (from vla-architecture.md §1, §3, §4):
    - Rate: 10Hz (decoupled from cursor)
    - Latency: ≤100ms hard limit, ≤50ms target
    - Input: dual-scale foveated image + gaze + hand velocity + UI context
    - Output: intent classification + attractive gradient field over screen space
    - Fusion: cross-attention (each modality attends to others)
    - Runtime: GPU-exclusive via DirectML (default) or TensorRT

Intent field (§4):
    - 2D Gaussian probability distribution over screen (64×64 grid)
    - Parameters (μ, Σ) are model outputs, not hardcoded
    - Gradient of intent field feeds into potential field blending
    - High entropy = uncertain, low = confident
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..token_types import MultimodalFrame, TokenType, get_null_token


@dataclass
class IntentField:
    """Output of the semantic path — a 2D probability field over screen space."""

    # (B, H, W) — log-probability field over the grid
    log_prob: torch.Tensor
    # (B, 2) — predicted mean (μ_x, μ_y) in normalized [0,1] space
    mu: torch.Tensor
    # (B, 2, 2) — predicted covariance matrix
    sigma: torch.Tensor
    # (B,) — intent field entropy (high = uncertain)
    entropy: torch.Tensor
    # (B, 2) — gradient at the current cursor position (feeds into blending)
    gradient: torch.Tensor
    # (B, num_intents) — intent class logits (future: "click", "scroll", "select")
    intent_logits: torch.Tensor | None = None


class ModalityEncoder(nn.Module):
    """Encode a single modality into a sequence of tokens for cross-attention."""

    def __init__(self, input_dim: int, embed_dim: int, num_tokens: int = 1):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.num_tokens = num_tokens
        if num_tokens > 1:
            self.expand = nn.Linear(embed_dim, embed_dim * num_tokens)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) or (B, C, H, W) for images

        Returns:
            (B, num_tokens, embed_dim) — token sequence
        """
        if x.ndim == 4:
            # Flatten spatial dims for images
            B = x.shape[0]
            x = x.reshape(B, -1)

        h = self.proj(x)  # (B, embed_dim)

        if self.num_tokens > 1:
            h = self.expand(h)  # (B, embed_dim × num_tokens)
            h = h.reshape(h.shape[0], self.num_tokens, -1)
        else:
            h = h.unsqueeze(1)  # (B, 1, embed_dim)

        return self.norm(h)


class CrossAttentionBlock(nn.Module):
    """Cross-attention block where all modalities attend to each other."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention over concatenated multimodal tokens.

        Args:
            x: (B, total_tokens, embed_dim) — all modalities concatenated

        Returns:
            (B, total_tokens, embed_dim) — attended features
        """
        # Self-attention (all tokens attend to all others)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class IntentFieldHead(nn.Module):
    """Predict a 2D Gaussian intent field over screen space.

    Outputs:
        - μ (mean position in [0,1] normalized space)
        - Σ (covariance matrix, parameterized via Cholesky factor for PD guarantee)
        - Unnormalized log-density over the spatial grid
    """

    def __init__(self, embed_dim: int, grid_h: int = 64, grid_w: int = 64):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Predict Gaussian parameters
        self.mu_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),  # μ ∈ [0, 1]
        )

        # Cholesky factor: 3 params for 2×2 lower triangular (L11, L21, L22)
        self.cholesky_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

        # Pre-compute grid coordinates
        gy = torch.linspace(0, 1, grid_h)
        gx = torch.linspace(0, 1, grid_w)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        # (H, W, 2)
        self.register_buffer("grid", torch.stack([grid_x, grid_y], dim=-1))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, embed_dim) — pooled multimodal features

        Returns:
            log_prob: (B, H, W) — log-probability field
            mu: (B, 2) — predicted mean
            sigma: (B, 2, 2) — covariance matrix
        """
        B = features.shape[0]

        # Predict mean
        mu = self.mu_head(features)  # (B, 2)

        # Predict Cholesky factor → covariance
        chol_params = self.cholesky_head(features)  # (B, 3)
        L = torch.zeros(B, 2, 2, device=features.device, dtype=features.dtype)
        L[:, 0, 0] = F.softplus(chol_params[:, 0]) + 1e-4  # Positive diagonal
        L[:, 1, 0] = chol_params[:, 1]
        L[:, 1, 1] = F.softplus(chol_params[:, 2]) + 1e-4
        sigma = L @ L.transpose(-1, -2)  # (B, 2, 2) — guaranteed PD

        # Compute log-probability over grid
        # grid: (H, W, 2), mu: (B, 2) → diff: (B, H, W, 2)
        diff = self.grid.unsqueeze(0) - mu[:, None, None, :]  # type: ignore[index]

        # Mahalanobis distance: diff @ Σ^{-1} @ diff^T
        sigma_inv = torch.linalg.inv(sigma)  # (B, 2, 2)
        # (B, H, W, 2) @ (B, 1, 2, 2) → (B, H, W, 2) → sum → (B, H, W)
        mahal = torch.einsum("bhwi,bij,bhwj->bhw", diff, sigma_inv[:, None, None, :, :].expand(-1, self.grid_h, self.grid_w, -1, -1).reshape(B, self.grid_h, self.grid_w, 2, 2), diff)

        # Log-probability (unnormalized Gaussian)
        log_det = torch.logdet(sigma)  # (B,)
        log_prob = -0.5 * (mahal + log_det[:, None, None] + 2 * math.log(2 * math.pi))

        return log_prob, mu, sigma


class SemanticModel(nn.Module):
    """Vision-Language Transformer for intent classification.

    Multimodal inputs are encoded into token sequences and fused via
    cross-attention. The fused representation drives an intent field head
    that outputs a 2D Gaussian probability distribution over screen space.
    """

    def __init__(
        self,
        backbone_name: str = "vit_small_patch16_224",
        embed_dim: int = 256,
        cross_attention_heads: int = 8,
        cross_attention_layers: int = 4,
        grid_h: int = 64,
        grid_w: int = 64,
        num_intents: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Image encoders (reuse backbone for both scales)
        self.global_encoder = self._make_image_encoder(backbone_name, embed_dim, (112, 112))
        self.foveal_encoder = self._make_image_encoder(backbone_name, embed_dim, (224, 224))

        # Vector modality encoders
        self.gaze_encoder = ModalityEncoder(3, embed_dim, num_tokens=1)
        self.velocity_encoder = ModalityEncoder(3, embed_dim, num_tokens=1)

        # Learned [NO_CONTEXT] token for missing UI context
        self.no_context_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Cross-attention fusion layers
        self.fusion_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, cross_attention_heads, dropout)
            for _ in range(cross_attention_layers)
        ])

        # Pool fused tokens to a single vector
        self.pool = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
        )

        # Output heads
        self.intent_field_head = IntentFieldHead(embed_dim, grid_h, grid_w)
        self.intent_classifier = nn.Linear(embed_dim, num_intents)

    @staticmethod
    def _make_image_encoder(
        backbone_name: str, embed_dim: int, img_size: tuple[int, int]
    ) -> nn.Module:
        """Create a ViT image encoder with projection to embed_dim."""
        backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            img_size=img_size,
        )
        backbone_dim = backbone.num_features
        return nn.Sequential(
            backbone,
            nn.Linear(backbone_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, frame: MultimodalFrame) -> IntentField:
        """
        Args:
            frame: MultimodalFrame with null-filled modalities

        Returns:
            IntentField with probability distribution and gradient
        """
        filled = frame.fill_nulls(device=next(self.parameters()).device)
        B = 1  # Infer batch size from first available tensor

        tokens = []

        # Encode image modalities
        if filled.image_global is not None:
            img_g = filled.image_global
            if img_g.ndim == 3:
                img_g = img_g.unsqueeze(0)
            B = img_g.shape[0]
            g_feat = self.global_encoder(img_g)  # (B, embed_dim)
            tokens.append(g_feat.unsqueeze(1))  # (B, 1, embed_dim)

        if filled.image_foveal is not None:
            img_f = filled.image_foveal
            if img_f.ndim == 3:
                img_f = img_f.unsqueeze(0)
            B = img_f.shape[0]
            f_feat = self.foveal_encoder(img_f)  # (B, embed_dim)
            tokens.append(f_feat.unsqueeze(1))

        # Encode vector modalities
        if filled.gaze_vector is not None:
            gaze = filled.gaze_vector
            if gaze.ndim == 1:
                gaze = gaze.unsqueeze(0)
            tokens.append(self.gaze_encoder(gaze))

        if filled.hand_velocity is not None:
            vel = filled.hand_velocity
            if vel.ndim == 1:
                vel = vel.unsqueeze(0)
            tokens.append(self.velocity_encoder(vel))

        # UI context (always the learned null token for now — Phase 1 future)
        tokens.append(self.no_context_token.expand(B, -1, -1))

        # Concatenate all modality tokens
        fused = torch.cat(tokens, dim=1)  # (B, total_tokens, embed_dim)

        # Cross-attention fusion
        for layer in self.fusion_layers:
            fused = layer(fused)

        # Pool to single vector (mean pooling + projection)
        pooled = self.pool(fused.mean(dim=1))  # (B, embed_dim)

        # Intent field
        log_prob, mu, sigma = self.intent_field_head(pooled)

        # Compute entropy: H = 0.5 * log(det(2πeΣ))
        entropy = 0.5 * torch.logdet(2 * math.pi * math.e * sigma)

        # Compute gradient at current cursor position (center of foveal crop ≈ mu)
        # Gradient of Gaussian log-prob at mu is zero by definition,
        # so we compute it at a slightly offset point (the actual cursor)
        # For now, use the gradient of the intent field evaluated at grid center
        gradient = self._compute_gradient(log_prob, mu)

        # Intent classification
        intent_logits = self.intent_classifier(pooled)

        return IntentField(
            log_prob=log_prob,
            mu=mu,
            sigma=sigma,
            entropy=entropy,
            gradient=gradient,
            intent_logits=intent_logits,
        )

    @staticmethod
    def _compute_gradient(log_prob: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Compute the gradient of the intent field at the predicted mean.

        This feeds into the potential field blending equation:
            cursor_delta = reflexive_delta + α · semantic_gradient

        For a Gaussian, the gradient at any point p is: -Σ^{-1}(p - μ)
        At μ itself the gradient is zero, so we sample nearby to get directional pull.
        In practice, we use the spatial gradient of the log_prob grid.

        Args:
            log_prob: (B, H, W)
            mu: (B, 2) — in [0, 1] normalized space

        Returns:
            (B, 2) — gradient vector (dx, dy) in normalized space
        """
        B, H, W = log_prob.shape

        # Spatial gradients via finite differences
        grad_x = log_prob[:, :, 1:] - log_prob[:, :, :-1]  # (B, H, W-1)
        grad_y = log_prob[:, 1:, :] - log_prob[:, :-1, :]  # (B, H-1, W)

        # Sample gradient at mu position
        # Convert mu from [0,1] to grid indices
        gx = (mu[:, 0] * (W - 2)).clamp(0, W - 2).long()
        gy = (mu[:, 1] * (H - 2)).clamp(0, H - 2).long()

        gradient = torch.zeros(B, 2, device=log_prob.device)
        for b in range(B):
            gradient[b, 0] = grad_x[b, gy[b], gx[b]]
            gradient[b, 1] = grad_y[b, gy[b], gx[b]]

        return gradient
