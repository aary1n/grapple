"""Configuration management for GrappleIntent.

Loads YAML configs into typed dataclasses. Supports hierarchical override
(default.yaml → environment-specific → CLI overrides).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ─── Reflexive Path ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReflexiveModelConfig:
    backbone: str = "mobilenetv3_small_100"
    input_dim: int = 66  # 21 landmarks × 3 + velocity 3
    cursor_output_dim: int = 2
    gesture_classes: int = 5
    dropout: float = 0.1


@dataclass(frozen=True)
class ReflexiveInferenceConfig:
    rate_hz: int = 120
    latency_budget_ms: float = 10.0
    latency_target_ms: float = 5.0
    watchdog_threshold_ms: float = 8.0
    watchdog_recovery_frames: int = 10
    onnx_path: str = "checkpoints/reflexive/mobilenetv3_cursor_v0.1_int8.onnx"


@dataclass(frozen=True)
class ReflexiveQuantConfig:
    # ONNX static INT8 (QDQ) per ADR-002 — INT4-AWQ is infeasible on the
    # CPU-pinned reflexive path (CUDA-only kernels, LLM-only tooling).
    method: str = "onnx-static-int8"
    bits: int = 8
    per_channel: bool = True
    calibration_samples: int = 500


@dataclass(frozen=True)
class ReflexiveCalibrationConfig:
    method: str = "prototypical"
    num_anchor_gestures: int = 10
    prototype_storage: str = "calibration/prototypes_{user_id}_v{version}.npz"
    augmentation_seed: int = 42


@dataclass(frozen=True)
class ReflexiveConfig:
    model: ReflexiveModelConfig = field(default_factory=ReflexiveModelConfig)
    inference: ReflexiveInferenceConfig = field(default_factory=ReflexiveInferenceConfig)
    quantization: ReflexiveQuantConfig = field(default_factory=ReflexiveQuantConfig)
    calibration: ReflexiveCalibrationConfig = field(default_factory=ReflexiveCalibrationConfig)


# ─── Semantic Path ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SemanticModelConfig:
    backbone: str = "vit_small_patch16_224"
    cross_attention_heads: int = 8
    cross_attention_layers: int = 4
    intent_field_resolution: tuple[int, int] = (64, 64)
    image_global_size: tuple[int, int] = (112, 112)
    image_foveal_size: tuple[int, int] = (224, 224)
    gaze_dim: int = 3
    hand_velocity_dim: int = 3
    ui_context_enabled: bool = False


@dataclass(frozen=True)
class SemanticInferenceConfig:
    rate_hz: int = 10
    latency_budget_ms: float = 100.0
    latency_target_ms: float = 50.0
    runtime: str = "directml"
    onnx_path: str = "checkpoints/semantic/semantic.onnx"


@dataclass(frozen=True)
class SemanticCalibrationConfig:
    method: str = "lora"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    adapter_storage: str = "calibration/lora_adapter_{user_id}_v{version}.safetensors"
    adapter_switch_budget_ms: float = 500.0
    augmentation_seed: int = 42


@dataclass(frozen=True)
class SemanticConfig:
    model: SemanticModelConfig = field(default_factory=SemanticModelConfig)
    inference: SemanticInferenceConfig = field(default_factory=SemanticInferenceConfig)
    calibration: SemanticCalibrationConfig = field(default_factory=SemanticCalibrationConfig)


# ─── Blending & System ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BlendingConfig:
    gain_k: float = 5.0
    offset_c0: float = 0.5
    decay_tau: float = 0.3


@dataclass(frozen=True)
class SpeculationConfig:
    enabled: bool = False
    horizon_frames: int = 5
    max_velocity: float = 10.0
    max_acceleration: float = 50.0


@dataclass(frozen=True)
class DegradationConfig:
    semantic_recovery_ramp_seconds: float = 1.0
    l3_fallback_detector: str = "GrappleDetector.py"


@dataclass(frozen=True)
class SystemConfig:
    log_level: str = "INFO"
    wandb_project: str = "grapple-intent"
    wandb_enabled: bool = True
    device_semantic: str = "auto"
    device_reflexive: str = "cpu"  # ALWAYS cpu


@dataclass(frozen=True)
class TrainingPathConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 0


@dataclass(frozen=True)
class TrainingConfig:
    reflexive: TrainingPathConfig = field(default_factory=TrainingPathConfig)
    semantic: TrainingPathConfig = field(
        default_factory=lambda: TrainingPathConfig(
            epochs=30,
            batch_size=32,
            learning_rate=5e-5,
            weight_decay=1e-2,
            scheduler="cosine_warmup",
            warmup_steps=500,
        )
    )


# ─── Root Config ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GrappleIntentConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    reflexive: ReflexiveConfig = field(default_factory=ReflexiveConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    blending: BlendingConfig = field(default_factory=BlendingConfig)
    speculation: SpeculationConfig = field(default_factory=SpeculationConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _merge_into_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively merge a dictionary into a frozen dataclass."""
    if not isinstance(data, dict):
        return data

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key not in field_types:
            continue

        field_type = field_types[key]

        # Resolve string type annotations
        if isinstance(field_type, str):
            # Handle forward references in the local namespace
            field_type = eval(field_type)  # noqa: S307

        # If the field is itself a dataclass, recurse
        if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[key] = _merge_into_dataclass(field_type, value)
        elif isinstance(value, list) and key in (
            "intent_field_resolution",
            "image_global_size",
            "image_foveal_size",
        ):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value

    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> GrappleIntentConfig:
    """Load config from YAML file, falling back to defaults.

    Args:
        path: Path to YAML config. If None, uses configs/default.yaml
              relative to the GrappleIntent package root.

    Returns:
        Frozen GrappleIntentConfig with all values populated.
    """
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "default.yaml"
    else:
        path = Path(path)

    if not path.exists():
        return GrappleIntentConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return _merge_into_dataclass(GrappleIntentConfig, raw)
