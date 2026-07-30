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


def _make_dataloaders(
    features: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    val_fraction: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Deterministic split + DataLoaders from in-memory arrays."""
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
    logger.info(
        "Synthetic dataset: %d samples (seed=%d)",
        features.shape[0], synth_config.seed,
    )
    return _make_dataloaders(
        features, targets, labels, batch_size, val_fraction, seed
    )


def load_recorded_arrays(
    recorded_paths: list[str], verify_hash: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Load and concatenate recorded .npz datasets (schema/hash validated).

    Returns (features, targets, labels, per_file_metadata).
    """
    from .recorder import load_recording

    feats, targs, labs, metas = [], [], [], []
    for path in recorded_paths:
        f, t, l, meta = load_recording(path, verify_hash=verify_hash)
        logger.info("Recorded dataset %s: %d samples", path, f.shape[0])
        feats.append(f)
        targs.append(t)
        labs.append(l)
        metas.append({"path": str(path), **meta})

    return (
        np.concatenate(feats),
        np.concatenate(targs),
        np.concatenate(labs),
        metas,
    )


def make_mixed_dataloaders(
    recorded_paths: list[str],
    synth_config: SyntheticConfig | None,
    batch_size: int,
    val_fraction: float = 0.1,
    seed: int = 42,
    verify_hash: bool = True,
) -> tuple[DataLoader, DataLoader, dict]:
    """Build DataLoaders mixing recorded (real) and synthetic data.

    Recorded data is the primary source; synthetic (if a config is given)
    augments it. The split permutation covers the combined pool so both
    sources appear in train and val.

    Returns:
        (train_loader, val_loader, data_summary) — data_summary is meant for
        W&B config logging (sample counts + recorded-file hashes).
    """
    features, targets, labels, metas = load_recorded_arrays(
        recorded_paths, verify_hash=verify_hash
    )
    num_recorded = features.shape[0]

    num_synthetic = 0
    if synth_config is not None:
        sf, st, sl = generate_reflexive_dataset(synth_config)
        num_synthetic = sf.shape[0]
        features = np.concatenate([features, sf])
        targets = np.concatenate([targets, st])
        labels = np.concatenate([labels, sl])

    logger.info(
        "Mixed dataset: %d recorded + %d synthetic = %d samples",
        num_recorded, num_synthetic, features.shape[0],
    )

    summary = {
        "num_recorded": num_recorded,
        "num_synthetic": num_synthetic,
        "recorded_files": [
            {"path": m.get("path"), "sha256": m.get("content_sha256")}
            for m in metas
        ],
    }
    train_loader, val_loader = _make_dataloaders(
        features, targets, labels, batch_size, val_fraction, seed
    )
    return train_loader, val_loader, summary
