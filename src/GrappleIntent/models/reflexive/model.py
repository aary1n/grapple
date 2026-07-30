"""Reflexive path model — fast cursor control at 120Hz.

Architecture (from vla-architecture.md §1):
    - Quantized MobileNetV3 (or equivalent lightweight backbone)
    - Input: raw hand landmarks (21×3) + velocity vector (3) = 66 dims
    - Output: cursor delta (dx, dy) + gesture confidence (5 classes)
    - Runtime: ONNX CPU (AVX-512), INT4-AWQ quantized
    - Latency: ≤10ms hard limit, ≤5ms target

Design decisions:
    - We use the MobileNetV3 feature extractor as a learned backbone rather than
      hand-engineering features from landmarks. The 1D landmark sequence is projected
      into the backbone's expected input space.
    - Dual-head output: regression head for cursor delta, classification head for gestures.
    - The gesture head feeds into prototypical network calibration (§5) — its penultimate
      layer is the embedding space for prototype lookup.
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReflexiveOutput:
    """Output of a single reflexive inference step."""

    cursor_delta: torch.Tensor  # (B, 2) — dx, dy in normalized space
    gesture_logits: torch.Tensor  # (B, num_classes)
    gesture_confidence: torch.Tensor  # (B,) — max softmax probability
    embedding: torch.Tensor  # (B, embed_dim) — for prototypical calibration


class LandmarkProjection(nn.Module):
    """Project 1D landmark+velocity vector into a spatial feature map.

    MobileNetV3 expects (B, 3, H, W) image input. We reshape the 1D landmark
    vector into a pseudo-spatial representation that the conv backbone can process.

    This is a learned projection — not a hand-engineered reshape — so the model
    can discover whatever spatial arrangement of landmarks is most useful.
    """

    def __init__(self, input_dim: int = 66, hidden_dim: int = 128, spatial_size: int = 7):
        super().__init__()
        self.spatial_size = spatial_size
        # Project landmarks to a 3-channel spatial feature map
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 3 * spatial_size * spatial_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) — landmark coordinates + velocity

        Returns:
            (B, 3, spatial_size, spatial_size) — pseudo-image for backbone
        """
        B = x.shape[0]
        return self.proj(x).reshape(B, 3, self.spatial_size, self.spatial_size)


class ReflexiveModel(nn.Module):
    """Dual-head reflexive model for cursor control and gesture classification.

    Architecture:
        landmarks → LandmarkProjection → MobileNetV3 backbone → shared features
                                                                    ├── cursor head (dx, dy)
                                                                    └── gesture head (5 classes)

    The gesture head's penultimate layer serves as the embedding space for
    prototypical network calibration (see §5 of architecture doc).
    """

    def __init__(
        self,
        backbone_name: str = "mobilenetv3_small_100",
        input_dim: int = 66,
        cursor_output_dim: int = 2,
        gesture_classes: int = 5,
        dropout: float = 0.1,
        embed_dim: int = 128,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.gesture_classes = gesture_classes
        self.embed_dim = embed_dim

        # Landmark → pseudo-spatial projection
        self.projection = LandmarkProjection(input_dim=input_dim)

        # MobileNetV3 backbone (features only, no classifier)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=False,
            num_classes=0,  # Remove classifier, get pooled features
        )
        # Probe the actual pooled output dim: num_features reports the final
        # conv dim (576 for mobilenetv3_small), but with num_classes=0 timm
        # emits the head-hidden features (1024) for mobilenetv3-style heads.
        with torch.no_grad():
            spatial = self.projection.spatial_size
            was_training = self.backbone.training
            self.backbone.eval()  # BatchNorm can't run train-mode on batch of 1
            backbone_dim = self.backbone(torch.zeros(1, 3, spatial, spatial)).shape[1]
            self.backbone.train(was_training)

        # Shared feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Cursor delta regression head
        self.cursor_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, cursor_output_dim),
            nn.Tanh(),  # Bound cursor delta to [-1, 1]
        )

        # Gesture classification head with embedding bottleneck
        # The embedding layer is the prototypical network's feature space
        self.gesture_embed = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.SiLU(inplace=True),
        )
        self.gesture_classifier = nn.Linear(embed_dim, gesture_classes)

    def forward(
        self, landmarks: torch.Tensor, return_embedding: bool = True
    ) -> ReflexiveOutput:
        """
        Args:
            landmarks: (B, input_dim) — 21×3 flattened landmarks + 3D velocity
            return_embedding: if True, return the gesture embedding for calibration

        Returns:
            ReflexiveOutput with cursor_delta, gesture_logits, confidence, embedding
        """
        # Project landmarks to pseudo-spatial representation
        x = self.projection(landmarks)  # (B, 3, 7, 7)

        # Backbone feature extraction
        features = self.backbone(x)  # (B, backbone_dim)

        # Shared projection
        shared = self.feature_proj(features)  # (B, 256)

        # Cursor head
        cursor_delta = self.cursor_head(shared)  # (B, 2)

        # Gesture head (through embedding bottleneck)
        embedding = self.gesture_embed(shared)  # (B, embed_dim)
        gesture_logits = self.gesture_classifier(embedding)  # (B, gesture_classes)

        # Confidence = max softmax probability
        gesture_confidence = F.softmax(gesture_logits, dim=-1).max(dim=-1).values

        return ReflexiveOutput(
            cursor_delta=cursor_delta,
            gesture_logits=gesture_logits,
            gesture_confidence=gesture_confidence,
            embedding=embedding if return_embedding else embedding.detach(),
        )

    def get_embedding(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Extract embedding only — used during prototypical calibration.

        This is a convenience method that avoids computing the cursor head
        when we only need gesture embeddings for prototype computation.
        """
        x = self.projection(landmarks)
        features = self.backbone(x)
        shared = self.feature_proj(features)
        return self.gesture_embed(shared)
