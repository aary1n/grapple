"""Synthetic intent-field training data for the semantic path.

Each sample simulates a screen with one salient target: a Gaussian blob
painted into the dual-scale images at position μ, a gaze vector pointing
toward it, and an intent label derived from the target's screen quadrant.
The supervised target is a point sampled from N(μ, σ_blob²·I) — training with
Gaussian NLL lets the model learn both the mean and a calibrated covariance
(σ̂ → σ_blob), per vla-architecture.md §4 (μ, Σ are model outputs).

Deterministic per (seed, index) so runs are reproducible without holding the
whole image set in memory. This is a bootstrap dataset: it validates that the
training loop converges, not that the model is good.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

GLOBAL_SIZE = 112  # matches TokenType.IMAGE_GLOBAL
FOVEAL_SIZE = 224  # matches TokenType.IMAGE_FOVEAL
NUM_QUADRANT_INTENTS = 4


@dataclass(frozen=True)
class SemanticSyntheticConfig:
    """Generation parameters — log to W&B with every run."""

    num_samples: int = 512
    blob_sigma_range: tuple[float, float] = (0.03, 0.08)
    background_noise_std: float = 0.05
    foveal_jitter: float = 0.05  # crop-center offset from μ, normalized
    seed: int = 42


def _paint_blob(
    size: int, cx: float, cy: float, sigma: float, channel: int, noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """(3, size, size) float32 image: noise background + Gaussian blob."""
    img = rng.normal(0.0, noise_std, size=(3, size, size)).astype(np.float32)
    coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
    gx, gy = np.meshgrid(coords, coords)  # gx = x (cols), gy = y (rows)
    blob = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma**2))
    img[channel] += blob.astype(np.float32)
    return img


class SemanticIntentDataset(Dataset):
    """Yields (image_global, image_foveal, gaze, velocity, target_point, intent_label)."""

    def __init__(self, config: SemanticSyntheticConfig) -> None:
        self._config = config

    def __len__(self) -> int:
        return self._config.num_samples

    def __getitem__(self, idx: int):
        cfg = self._config
        rng = np.random.default_rng(cfg.seed * 1_000_003 + idx)

        mu = rng.uniform(0.15, 0.85, size=2)
        blob_sigma = rng.uniform(*cfg.blob_sigma_range)
        # Intent = screen quadrant of the target (left/right × top/bottom)
        label = int(mu[0] >= 0.5) + 2 * int(mu[1] >= 0.5)

        image_global = _paint_blob(
            GLOBAL_SIZE, mu[0], mu[1], blob_sigma,
            channel=label % 3, noise_std=cfg.background_noise_std, rng=rng,
        )

        # Foveal crop: centered near μ (imperfect gaze), so the blob sits near
        # the crop center at higher effective resolution
        offset = rng.uniform(-cfg.foveal_jitter, cfg.foveal_jitter, size=2)
        image_foveal = _paint_blob(
            FOVEAL_SIZE, 0.5 - offset[0], 0.5 - offset[1], blob_sigma * 2,
            channel=label % 3, noise_std=cfg.background_noise_std, rng=rng,
        )

        gaze = np.array([mu[0] - 0.5, mu[1] - 0.5, -1.0], dtype=np.float32)
        gaze /= np.linalg.norm(gaze)
        velocity = rng.normal(0, 0.1, size=3).astype(np.float32)

        target_point = rng.normal(mu, blob_sigma).astype(np.float32)
        target_point = np.clip(target_point, 0.0, 1.0)

        return (
            torch.from_numpy(image_global),
            torch.from_numpy(image_foveal),
            torch.from_numpy(gaze),
            torch.from_numpy(velocity),
            torch.from_numpy(target_point),
            torch.tensor(label, dtype=torch.int64),
        )
