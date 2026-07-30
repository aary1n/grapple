"""Training loop for the reflexive path model.

Trains the MobileNetV3-based dual-head model on (landmarks, velocity) → (cursor_delta, gesture)
pairs. Supports W&B experiment tracking per architecture rules.

This is the simplest training loop — supervised regression + classification.
The model learns from ground-truth cursor targets and gesture labels.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..configs import GrappleIntentConfig, load_config
from ..models.reflexive.model import ReflexiveModel

logger = logging.getLogger(__name__)

# Semantic checkpoint name per ml-research.md §2: {arch}_{task}_{version}_{quant}
CHECKPOINT_NAME = "mobilenetv3_cursor_v0.1_fp32.pt"


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


# ─── CLI entry point ──────────────────────────────────────────────────────────


def _seed_everything(seed: int) -> None:
    """Pin all RNGs, per ml-research.md §1."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _git_commit_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _wandb_mode() -> str:
    """Pick W&B mode without ever blocking on an interactive login prompt."""
    if os.environ.get("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    netrc = Path.home() / ("_netrc" if os.name == "nt" else ".netrc")
    try:
        if netrc.exists() and "api.wandb.ai" in netrc.read_text(errors="ignore"):
            return "online"
    except OSError:
        pass
    return "offline"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the reflexive path model on synthetic data"
    )
    parser.add_argument("--config", default=None, help="Path to GrappleIntent YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-sequences", type=int, default=250,
                        help="Synthetic sequences to generate")
    parser.add_argument("--recorded-data", action="append", default=[],
                        metavar="NPZ",
                        help="Recorded .npz dataset(s) from GrappleIntent.data.recorder "
                             "(repeatable). Mixed with synthetic unless --no-synthetic.")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Train on recorded data only (requires --recorded-data)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="checkpoints/reflexive")
    parser.add_argument("--run-name", default=None,
                        help="W&B run name (default derived from data source)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip downloading pretrained backbone weights")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    from ..data.dataset import make_mixed_dataloaders, make_synthetic_dataloaders
    from ..data.synthetic import SyntheticConfig

    _seed_everything(args.seed)

    if args.no_synthetic and not args.recorded_data:
        parser.error("--no-synthetic requires at least one --recorded-data")

    config = load_config(args.config)
    tc = config.training.reflexive
    if args.epochs is not None:
        tc = dataclasses.replace(tc, epochs=args.epochs)
    if args.batch_size is not None:
        tc = dataclasses.replace(tc, batch_size=args.batch_size)
    config = dataclasses.replace(
        config, training=dataclasses.replace(config.training, reflexive=tc)
    )

    synth_config = None if args.no_synthetic else SyntheticConfig(
        num_sequences=args.num_sequences, seed=args.seed
    )
    if args.recorded_data:
        train_loader, val_loader, data_summary = make_mixed_dataloaders(
            args.recorded_data, synth_config,
            batch_size=tc.batch_size, seed=args.seed,
        )
        data_source = "recorded" if args.no_synthetic else "mixed"
    else:
        train_loader, val_loader = make_synthetic_dataloaders(
            synth_config, batch_size=tc.batch_size, seed=args.seed
        )
        data_summary = {}
        data_source = "synthetic"
    run_name = args.run_name or f"cursor-mobilenetv3-{data_source}"

    mc = config.reflexive.model
    model = ReflexiveModel(
        backbone_name=mc.backbone,
        input_dim=mc.input_dim,
        cursor_output_dim=mc.cursor_output_dim,
        gesture_classes=mc.gesture_classes,
        dropout=mc.dropout,
    )
    if args.no_pretrained:
        # Rebuild backbone without pretrained weights (offline environments)
        import timm
        model.backbone = timm.create_model(mc.backbone, pretrained=False, num_classes=0)

    git_hash = _git_commit_hash()
    full_run_config = {
        "seed": args.seed,
        "git_commit": git_hash,
        "torch_version": str(torch.__version__),  # TorchVersion breaks yaml.safe_dump
        "model": dataclasses.asdict(mc),
        "training": dataclasses.asdict(tc),
        "data": {
            "source": data_source,
            **(dataclasses.asdict(synth_config) if synth_config else {}),
            **data_summary,
        },
    }

    wandb_run = None
    if not args.no_wandb and config.system.wandb_enabled:
        try:
            import wandb

            wandb_run = wandb.init(
                project=config.system.wandb_project,
                name=run_name,
                config=full_run_config,
                tags=["reflexive", "v0.1", data_source],
                mode=_wandb_mode(),
            )
            logger.info("W&B run started (mode=%s)", _wandb_mode())
        except Exception:
            logger.exception("W&B init failed — continuing without tracking")
            wandb_run = None

    history = train_reflexive(
        model, train_loader, val_loader, config,
        output_dir=args.output_dir, wandb_run=wandb_run,
    )

    # Save the semantically-named checkpoint + its config YAML side-by-side,
    # per ml-research.md §2 (checkpoint hygiene)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / CHECKPOINT_NAME
    torch.save(model.state_dict(), checkpoint_path)

    import yaml

    with open(output_dir / f"{checkpoint_path.stem}.yaml", "w") as f:
        yaml.safe_dump(full_run_config, f, sort_keys=False)

    final = history[-1]
    logger.info(
        "Training complete: %d epochs, cursor_loss=%.4f, gesture_acc=%.2f%% -> %s",
        final.epoch, final.cursor_loss, final.gesture_accuracy * 100, checkpoint_path,
    )

    if wandb_run is not None:
        try:
            import wandb

            artifact = wandb.Artifact("mobilenetv3_cursor_v0.1_fp32", type="model")
            artifact.add_file(str(checkpoint_path))
            wandb_run.log_artifact(artifact)
            wandb_run.finish()
        except Exception:
            logger.exception("W&B artifact logging failed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
