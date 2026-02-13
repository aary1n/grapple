"""Training loop for the reflexive path model.

Trains the MobileNetV3-based dual-head model on (landmarks, velocity) → (cursor_delta, gesture)
pairs. Supports W&B experiment tracking per architecture rules.

This is the simplest training loop — supervised regression + classification.
The model learns from ground-truth cursor targets and gesture labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..configs import GrappleIntentConfig
from ..models.reflexive.model import ReflexiveModel

logger = logging.getLogger(__name__)


@dataclass
class TrainMetrics:
    epoch: int
    cursor_loss: float
    gesture_loss: float
    total_loss: float
    gesture_accuracy: float


def train_reflexive(
    model: ReflexiveModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: GrappleIntentConfig,
    output_dir: str | Path = "checkpoints/reflexive",
    wandb_run: object | None = None,
) -> list[TrainMetrics]:
    """Train the reflexive path model.

    Args:
        model: ReflexiveModel to train
        train_loader: DataLoader yielding (landmarks, cursor_target, gesture_label)
            - landmarks: (B, 66) float32
            - cursor_target: (B, 2) float32 — ground truth dx, dy
            - gesture_label: (B,) long — gesture class index
        val_loader: Optional validation DataLoader
        config: Full GrappleIntent config
        output_dir: Where to save checkpoints
        wandb_run: Optional W&B run for logging

    Returns:
        List of per-epoch training metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tc = config.training.reflexive
    device = torch.device(config.system.device_reflexive)  # Always CPU
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc.learning_rate,
        weight_decay=tc.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc.epochs
    )

    # Loss functions
    cursor_criterion = nn.MSELoss()
    gesture_criterion = nn.CrossEntropyLoss()

    history: list[TrainMetrics] = []
    best_val_loss = float("inf")

    for epoch in range(1, tc.epochs + 1):
        model.train()
        epoch_cursor_loss = 0.0
        epoch_gesture_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for landmarks, cursor_target, gesture_label in train_loader:
            landmarks = landmarks.to(device)
            cursor_target = cursor_target.to(device)
            gesture_label = gesture_label.to(device)

            optimizer.zero_grad()

            output = model(landmarks, return_embedding=False)

            # Dual loss: regression + classification
            c_loss = cursor_criterion(output.cursor_delta, cursor_target)
            g_loss = gesture_criterion(output.gesture_logits, gesture_label)
            loss = c_loss + g_loss

            loss.backward()
            optimizer.step()

            epoch_cursor_loss += c_loss.item() * landmarks.shape[0]
            epoch_gesture_loss += g_loss.item() * landmarks.shape[0]
            epoch_correct += (output.gesture_logits.argmax(dim=-1) == gesture_label).sum().item()
            epoch_total += landmarks.shape[0]

        scheduler.step()

        metrics = TrainMetrics(
            epoch=epoch,
            cursor_loss=epoch_cursor_loss / epoch_total,
            gesture_loss=epoch_gesture_loss / epoch_total,
            total_loss=(epoch_cursor_loss + epoch_gesture_loss) / epoch_total,
            gesture_accuracy=epoch_correct / epoch_total,
        )
        history.append(metrics)

        logger.info(
            "Epoch %d/%d — cursor_loss=%.4f, gesture_loss=%.4f, acc=%.2f%%",
            epoch, tc.epochs, metrics.cursor_loss, metrics.gesture_loss,
            metrics.gesture_accuracy * 100,
        )

        # W&B logging
        if wandb_run is not None:
            try:
                import wandb
                wandb.log({
                    "train/cursor_loss": metrics.cursor_loss,
                    "train/gesture_loss": metrics.gesture_loss,
                    "train/total_loss": metrics.total_loss,
                    "train/gesture_accuracy": metrics.gesture_accuracy,
                    "train/lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                })
            except Exception:
                pass

        # Checkpoint best model
        if val_loader is not None:
            val_loss = _validate(model, val_loader, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_dir / "best.pt")
                logger.info("Saved best model (val_loss=%.4f)", val_loss)
        else:
            # Save every 10 epochs if no validation
            if epoch % 10 == 0:
                torch.save(model.state_dict(), output_dir / f"epoch_{epoch}.pt")

    # Always save final
    torch.save(model.state_dict(), output_dir / "final.pt")
    return history


def _validate(
    model: ReflexiveModel, loader: DataLoader, device: torch.device
) -> float:
    """Run validation and return total loss."""
    model.eval()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for landmarks, cursor_target, gesture_label in loader:
            landmarks = landmarks.to(device)
            cursor_target = cursor_target.to(device)
            gesture_label = gesture_label.to(device)

            output = model(landmarks, return_embedding=False)
            c_loss = F.mse_loss(output.cursor_delta, cursor_target)
            g_loss = F.cross_entropy(output.gesture_logits, gesture_label)

            total_loss += (c_loss + g_loss).item() * landmarks.shape[0]
            n += landmarks.shape[0]

    model.train()
    return total_loss / n if n > 0 else float("inf")
