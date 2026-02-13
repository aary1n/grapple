"""Dual-scale foveated image preprocessing for the semantic path.

From vla-architecture.md §3:
    - Global context (112×112): captures spatial layout
    - Foveal crop (224×224): high-res window around point of attention
    - Solves spatial precision (disambiguating adjacent buttons), NOT text reading
    - Text reading is UIContext's job (Phase 1 — Windows UI Automation API)

Latency budget (§9): ≤2ms for foveated crop generation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np


def preprocess_foveated(
    frame_rgb: np.ndarray | torch.Tensor,
    gaze_x: float,
    gaze_y: float,
    global_size: tuple[int, int] = (112, 112),
    foveal_size: tuple[int, int] = (224, 224),
    foveal_crop_ratio: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate dual-scale foveated images from a single frame.

    Args:
        frame_rgb: (H, W, 3) uint8 or (3, H, W) float32 frame
        gaze_x: Normalized gaze/cursor X position [0, 1]
        gaze_y: Normalized gaze/cursor Y position [0, 1]
        global_size: Output size for global downscale
        foveal_size: Output size for foveal crop
        foveal_crop_ratio: Fraction of frame width/height to crop for foveal view

    Returns:
        (global_image, foveal_image) — both as (3, H, W) float32 tensors
        normalized to [0, 1] range
    """
    # Convert to torch tensor if needed
    if isinstance(frame_rgb, np.ndarray):
        if frame_rgb.dtype == np.uint8:
            t = torch.from_numpy(frame_rgb).float() / 255.0
        else:
            t = torch.from_numpy(frame_rgb).float()

        # HWC → CHW
        if t.ndim == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
    else:
        t = frame_rgb.float()
        if t.ndim == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        if t.max() > 1.0:
            t = t / 255.0

    C, H, W = t.shape

    # ── Global: simple bilinear downscale ─────────────────────────────────
    global_img = F.interpolate(
        t.unsqueeze(0),
        size=global_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # (3, 112, 112)

    # ── Foveal: crop around gaze point, then resize ──────────────────────
    crop_h = int(H * foveal_crop_ratio)
    crop_w = int(W * foveal_crop_ratio)

    # Center crop on gaze position
    cx = int(gaze_x * W)
    cy = int(gaze_y * H)

    # Clamp to frame bounds
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(W, x1 + crop_w)
    y2 = min(H, y1 + crop_h)

    # Adjust if we hit a boundary
    if x2 - x1 < crop_w:
        x1 = max(0, x2 - crop_w)
    if y2 - y1 < crop_h:
        y1 = max(0, y2 - crop_h)

    foveal_crop = t[:, y1:y2, x1:x2]  # (3, crop_h, crop_w)

    foveal_img = F.interpolate(
        foveal_crop.unsqueeze(0),
        size=foveal_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # (3, 224, 224)

    return global_img, foveal_img
