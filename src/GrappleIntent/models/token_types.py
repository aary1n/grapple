"""Multimodal token definitions for the VLA fusion pipeline.

From vla-architecture.md §3, every token type must define:
    1. Tensor shape and dtype
    2. Null/missing representation
    3. Registration in the fusion registry (config-driven)

Token types:
    - ImagePatch_Global:  (3, 112, 112) float32 — low-res full screen
    - ImagePatch_Foveal:  (3, 224, 224) float32 — high-res crop at gaze
    - GazeVector:         (3,) float32 — eye direction unit vector
    - HandVelocity:       (3,) float32 — 3D velocity in normalized space
    - UIContext:           variable-length token sequence (future Phase 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch


class TokenType(Enum):
    IMAGE_GLOBAL = "ImagePatch_Global"
    IMAGE_FOVEAL = "ImagePatch_Foveal"
    GAZE_VECTOR = "GazeVector"
    HAND_VELOCITY = "HandVelocity"
    UI_CONTEXT = "UIContext"


@dataclass(frozen=True)
class TokenSpec:
    """Specification for a multimodal token type."""

    token_type: TokenType
    shape: tuple[int, ...]
    dtype: torch.dtype
    null_value: Any  # Value used when modality is missing
    description: str


# ── Token Registry ────────────────────────────────────────────────────────────

TOKEN_REGISTRY: dict[TokenType, TokenSpec] = {
    TokenType.IMAGE_GLOBAL: TokenSpec(
        token_type=TokenType.IMAGE_GLOBAL,
        shape=(3, 112, 112),
        dtype=torch.float32,
        null_value="zeros",  # torch.zeros(3, 112, 112)
        description="Low-res downscale of full screen/region for spatial layout",
    ),
    TokenType.IMAGE_FOVEAL: TokenSpec(
        token_type=TokenType.IMAGE_FOVEAL,
        shape=(3, 224, 224),
        dtype=torch.float32,
        null_value="zeros",  # torch.zeros(3, 224, 224)
        description="High-res crop centered on cursor/gaze for target disambiguation",
    ),
    TokenType.GAZE_VECTOR: TokenSpec(
        token_type=TokenType.GAZE_VECTOR,
        shape=(3,),
        dtype=torch.float32,
        null_value=np.array([0.0, 0.0, -1.0], dtype=np.float32),  # forward-facing
        description="Eye tracking direction (unit vector)",
    ),
    TokenType.HAND_VELOCITY: TokenSpec(
        token_type=TokenType.HAND_VELOCITY,
        shape=(3,),
        dtype=torch.float32,
        null_value="zeros",  # torch.zeros(3)
        description="3D velocity vector from temporal differencing",
    ),
    TokenType.UI_CONTEXT: TokenSpec(
        token_type=TokenType.UI_CONTEXT,
        shape=(-1,),  # Variable length
        dtype=torch.float32,
        null_value="learned",  # Learned [NO_CONTEXT] embedding
        description="Screen region descriptor (future: Windows UI Automation API)",
    ),
}


def get_null_token(token_type: TokenType, device: torch.device | None = None) -> torch.Tensor:
    """Get the null/missing representation for a token type.

    Used for graceful degradation when a modality is unavailable.
    """
    spec = TOKEN_REGISTRY[token_type]

    # ndarray first — comparing an array against a string is ambiguous
    if isinstance(spec.null_value, np.ndarray):
        return torch.from_numpy(spec.null_value).to(dtype=spec.dtype, device=device)
    elif spec.null_value == "zeros":
        return torch.zeros(spec.shape, dtype=spec.dtype, device=device)
    elif spec.null_value == "learned":
        # Placeholder — the actual learned embedding is part of the model
        # Return zeros as a fallback; the model's [NO_CONTEXT] token overrides this
        return torch.zeros(64, dtype=spec.dtype, device=device)
    else:
        return torch.zeros(spec.shape, dtype=spec.dtype, device=device)


@dataclass
class MultimodalFrame:
    """A single frame of multimodal input for the semantic path.

    Any field can be None — the fusion pipeline substitutes null tokens
    for missing modalities (graceful degradation).
    """

    image_global: torch.Tensor | None = None  # (3, 112, 112)
    image_foveal: torch.Tensor | None = None  # (3, 224, 224)
    gaze_vector: torch.Tensor | None = None  # (3,)
    hand_velocity: torch.Tensor | None = None  # (3,)
    ui_context: torch.Tensor | None = None  # variable
    timestamp: int = 0  # QPC ticks

    def to_device(self, device: torch.device) -> MultimodalFrame:
        """Move all tensors to the specified device."""
        return MultimodalFrame(
            image_global=self.image_global.to(device) if self.image_global is not None else None,
            image_foveal=self.image_foveal.to(device) if self.image_foveal is not None else None,
            gaze_vector=self.gaze_vector.to(device) if self.gaze_vector is not None else None,
            hand_velocity=self.hand_velocity.to(device) if self.hand_velocity is not None else None,
            ui_context=self.ui_context.to(device) if self.ui_context is not None else None,
            timestamp=self.timestamp,
        )

    def fill_nulls(self, device: torch.device | None = None) -> MultimodalFrame:
        """Replace None fields with their defined null tokens."""
        return MultimodalFrame(
            image_global=(
                self.image_global
                if self.image_global is not None
                else get_null_token(TokenType.IMAGE_GLOBAL, device)
            ),
            image_foveal=(
                self.image_foveal
                if self.image_foveal is not None
                else get_null_token(TokenType.IMAGE_FOVEAL, device)
            ),
            gaze_vector=(
                self.gaze_vector
                if self.gaze_vector is not None
                else get_null_token(TokenType.GAZE_VECTOR, device)
            ),
            hand_velocity=(
                self.hand_velocity
                if self.hand_velocity is not None
                else get_null_token(TokenType.HAND_VELOCITY, device)
            ),
            ui_context=self.ui_context,  # Leave as None — model handles [NO_CONTEXT]
            timestamp=self.timestamp,
        )
