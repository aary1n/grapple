"""Training loop for the semantic path model (research-scoped scaffold).

Trains the VL-Transformer on the synthetic intent-field dataset
(data/synthetic_semantic.py): dual-scale images + gaze + velocity → 2D
Gaussian intent field (μ, Σ) + intent classification.

Losses:
    - Gaussian NLL of the target point under the predicted (μ, Σ) — trains
      both the mean and a calibrated covariance (§4: μ, Σ are model outputs)
    - Cross-entropy on the quadrant intent label

Goal of this scaffold: the loop runs and converges on synthetic data. It is
NOT expected to produce a good model — real screen data and the UIContext
token are future milestones.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..configs import GrappleIntentConfig, load_config
from ..data.synthetic_semantic import (
    NUM_QUADRANT_INTENTS,
    SemanticIntentDataset,
    SemanticSyntheticConfig,
)
from ..models.semantic.model import SemanticModel
from ..models.token_types import MultimodalFrame
from .train_reflexive import _git_commit_hash, _seed_everything, _wandb_mode

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "vl_transformer_intent_v0.1_fp32.pt"


@dataclass
class SemanticTrainMetrics:
    epoch: int
    nll_loss: float
    intent_loss: float
    total_loss: float
    mu_mse: float
    intent_accuracy: float
    mean_entropy: float


def gaussian_nll(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean negative log-likelihood of target points under N(μ, Σ).

    Σ is Cholesky-parameterized by the model, so it is guaranteed PD.
    """
    diff = (target - mu).unsqueeze(-1)  # (B, 2, 1)
    sigma_inv = torch.linalg.inv(sigma)
    mahal = (diff.transpose(-1, -2) @ sigma_inv @ diff).squeeze(-1).squeeze(-1)
    return 0.5 * (mahal + torch.logdet(sigma) + 2 * math.log(2 * math.pi)).mean()


def _run_epoch(
    model: SemanticModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> SemanticTrainMetrics:
    training = optimizer is not None
    model.train(training)

    sums = {"nll": 0.0, "ce": 0.0, "mse": 0.0, "entropy": 0.0}
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for img_g, img_f, gaze, vel, target, label in loader:
            frame = MultimodalFrame(
                image_global=img_g.to(device),
                image_foveal=img_f.to(device),
                gaze_vector=gaze.to(device),
                hand_velocity=vel.to(device),
            )
            target = target.to(device)
            label = label.to(device)

            out = model(frame)
            nll = gaussian_nll(out.mu, out.sigma, target)
            # Only the first NUM_QUADRANT_INTENTS logits carry labels here;
            # the head keeps its full width for future intent vocabularies.
            ce = F.cross_entropy(out.intent_logits[:, :NUM_QUADRANT_INTENTS], label)
            loss = nll + ce

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            b = label.shape[0]
            sums["nll"] += nll.item() * b
            sums["ce"] += ce.item() * b
            sums["mse"] += F.mse_loss(out.mu, target).item() * b
            sums["entropy"] += out.entropy.mean().item() * b
            pred = out.intent_logits[:, :NUM_QUADRANT_INTENTS].argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += b

    return SemanticTrainMetrics(
        epoch=0,
        nll_loss=sums["nll"] / total,
        intent_loss=sums["ce"] / total,
        total_loss=(sums["nll"] + sums["ce"]) / total,
        mu_mse=sums["mse"] / total,
        intent_accuracy=correct / total,
        mean_entropy=sums["entropy"] / total,
    )


def train_semantic(
    model: SemanticModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: GrappleIntentConfig,
    output_dir: str | Path = "checkpoints/semantic",
    wandb_run: object | None = None,
) -> list[SemanticTrainMetrics]:
    """Train the semantic model. Returns per-epoch training metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tc = config.training.semantic
    device_name = config.system.device_semantic
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc.epochs)

    history: list[SemanticTrainMetrics] = []
    best_val = float("inf")

    for epoch in range(1, tc.epochs + 1):
        metrics = _run_epoch(model, train_loader, device, optimizer)
        metrics.epoch = epoch
        scheduler.step()
        history.append(metrics)

        logger.info(
            "Epoch %d/%d — nll=%.4f ce=%.4f mu_mse=%.5f acc=%.1f%% H=%.2f",
            epoch, tc.epochs, metrics.nll_loss, metrics.intent_loss,
            metrics.mu_mse, metrics.intent_accuracy * 100, metrics.mean_entropy,
        )

        log_payload = {
            "train/nll_loss": metrics.nll_loss,
            "train/intent_loss": metrics.intent_loss,
            "train/total_loss": metrics.total_loss,
            "train/mu_mse": metrics.mu_mse,
            "train/intent_accuracy": metrics.intent_accuracy,
            "train/intent_field_entropy": metrics.mean_entropy,
            "train/lr": scheduler.get_last_lr()[0],
            "epoch": epoch,
        }

        if val_loader is not None:
            val = _run_epoch(model, val_loader, device, optimizer=None)
            log_payload.update({
                "val/nll_loss": val.nll_loss,
                "val/mu_mse": val.mu_mse,
                "val/intent_accuracy": val.intent_accuracy,
            })
            if val.total_loss < best_val:
                best_val = val.total_loss
                torch.save(model.state_dict(), output_dir / "best.pt")
                logger.info("Saved best model (val_loss=%.4f)", val.total_loss)

        if wandb_run is not None:
            try:
                import wandb

                wandb.log(log_payload)
            except Exception:
                pass

    torch.save(model.state_dict(), output_dir / "final.pt")
    return history


# ─── CLI entry point ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the semantic path model on synthetic intent fields"
    )
    parser.add_argument("--config", default=None, help="GrappleIntent YAML config path")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-samples", type=int, default=512,
                        help="Synthetic samples to generate")
    parser.add_argument("--backbone", default=None,
                        help="Override image backbone (e.g. vit_tiny_patch16_224 "
                             "for CPU smoke runs)")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="checkpoints/semantic")
    parser.add_argument("--run-name", default="intent-vlt-synthetic")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip downloading pretrained backbone weights")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    _seed_everything(args.seed)

    config = load_config(args.config)
    tc = config.training.semantic
    if args.epochs is not None:
        tc = dataclasses.replace(tc, epochs=args.epochs)
    if args.batch_size is not None:
        tc = dataclasses.replace(tc, batch_size=args.batch_size)
    config = dataclasses.replace(
        config, training=dataclasses.replace(config.training, semantic=tc)
    )

    mc = config.semantic.model
    backbone = args.backbone or mc.backbone

    synth_config = SemanticSyntheticConfig(num_samples=args.num_samples, seed=args.seed)
    dataset = SemanticIntentDataset(synth_config)
    split_rng = np.random.default_rng(args.seed)
    perm = split_rng.permutation(len(dataset))
    val_count = max(1, len(dataset) // 10)
    train_ds = Subset(dataset, perm[val_count:].tolist())
    val_ds = Subset(dataset, perm[:val_count].tolist())

    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=tc.batch_size, shuffle=True,
        generator=shuffle_gen, num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=tc.batch_size, num_workers=0)
    logger.info("Semantic dataset: %d train / %d val", len(train_ds), len(val_ds))

    model = SemanticModel(
        backbone_name=backbone,
        embed_dim=args.embed_dim,
        cross_attention_heads=mc.cross_attention_heads,
        cross_attention_layers=mc.cross_attention_layers,
        grid_h=mc.intent_field_resolution[0],
        grid_w=mc.intent_field_resolution[1],
        pretrained=not args.no_pretrained,
    )

    full_run_config = {
        "seed": args.seed,
        "git_commit": _git_commit_hash(),
        "torch_version": str(torch.__version__),
        "model": {
            **dataclasses.asdict(mc),
            "backbone": backbone,
            "embed_dim": args.embed_dim,
            "pretrained": not args.no_pretrained,
        },
        "training": dataclasses.asdict(tc),
        "data": {"source": "synthetic-intent-field", **dataclasses.asdict(synth_config)},
    }

    wandb_run = None
    if not args.no_wandb and config.system.wandb_enabled:
        try:
            import wandb

            wandb_run = wandb.init(
                project=config.system.wandb_project,
                name=args.run_name,
                config=full_run_config,
                tags=["semantic", "v0.1", "synthetic"],
                mode=_wandb_mode(),
            )
            logger.info("W&B run started (mode=%s)", _wandb_mode())
        except Exception:
            logger.exception("W&B init failed — continuing without tracking")
            wandb_run = None

    history = train_semantic(
        model, train_loader, val_loader, config,
        output_dir=args.output_dir, wandb_run=wandb_run,
    )

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / CHECKPOINT_NAME
    torch.save(model.state_dict(), checkpoint_path)

    import yaml

    with open(output_dir / f"{checkpoint_path.stem}.yaml", "w") as f:
        yaml.safe_dump(full_run_config, f, sort_keys=False)

    final = history[-1]
    logger.info(
        "Training complete: %d epochs, nll=%.4f, mu_mse=%.5f, acc=%.1f%% -> %s",
        final.epoch, final.nll_loss, final.mu_mse,
        final.intent_accuracy * 100, checkpoint_path,
    )

    if wandb_run is not None:
        try:
            import wandb

            artifact = wandb.Artifact("vl_transformer_intent_v0.1_fp32", type="model")
            artifact.add_file(str(checkpoint_path))
            wandb_run.log_artifact(artifact)
            wandb_run.finish()
        except Exception:
            logger.exception("W&B artifact logging failed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
