"""Dataset and DataLoader factories for reflexive path training.

Fulfills the training loop's DataLoader contract:
    (landmarks (B, 66) float32, cursor_target (B, 2) float32, gesture_label (B,) long)

Splits are deterministic given the seed, per ml-research.md §6.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .synthetic import SyntheticConfig, generate_reflexive_dataset

logger = logging.getLogger(__name__)


class ReflexiveLandmarkDataset(Dataset):
    """In-memory dataset of (landmarks, cursor_target, gesture_label) samples."""

    def __init__(
        self,
        features: np.ndarray,
        cursor_targets: np.ndarray,
        gesture_labels: np.ndarray,
    ) -> None:
        self.features = torch.from_numpy(features)
        self.cursor_targets = torch.from_numpy(cursor_targets)
        self.gesture_labels = torch.from_numpy(gesture_labels)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[idx],
            self.cursor_targets[idx],
            self.gesture_labels[idx],
        )


def make_synthetic_dataloaders(
    synth_config: SyntheticConfig,
    batch_size: int,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build deterministic train/val DataLoaders from synthetic data.

    Args:
        synth_config: Synthetic generation parameters (includes its own seed).
        batch_size: Training batch size.
        val_fraction: Fraction of samples held out for validation.
        seed: Seed for the split permutation and shuffle generator.

    Returns:
        (train_loader, val_loader)
    """
    features, targets, labels = generate_reflexive_dataset(synth_config)

    n = features.shape[0]
    split_rng = np.random.default_rng(seed)
    perm = split_rng.permutation(n)
    val_count = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:val_count], perm[val_count:]

    train_ds = ReflexiveLandmarkDataset(
        features[train_idx], targets[train_idx], labels[train_idx]
    )
    val_ds = ReflexiveLandmarkDataset(
        features[val_idx], targets[val_idx], labels[val_idx]
    )

    logger.info(
        "Synthetic dataset: %d train / %d val samples (seed=%d)",
        len(train_ds), len(val_ds), synth_config.seed,
    )

    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_gen,
        num_workers=0,  # Windows: keep in-process; dataset is in-memory anyway
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader
