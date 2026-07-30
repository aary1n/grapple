# GrappleIntent — Hierarchical VLA System

Replaces rule-based gesture detection (GrappleDetector.py) with learned models running as a sidecar alongside the existing Grapple pipeline.

## Architecture

```
Camera → SharedMemoryArena → [GrappleIntent] → FlatBufferSensorArena → MouseControllerNode
                                    │
                         ┌──────────┴──────────┐
                    Reflexive Path         Semantic Path
                    (MobileNetV3)         (VL-Transformer)
                   120Hz, ≤10ms           10Hz, ≤100ms
                   CPU-pinned (ONNX)      GPU (DirectML/TRT)
                   cursor control         intent parsing
```

**Key principle:** The cursor must always move. The VLA is an enhancement layer with graceful degradation (L0→L3).

## Quick Start

```bash
# Install (from repo root, into the project venv)
.venv/Scripts/python -m pip install -e "src/GrappleIntent[dev]"

# Run tests
.venv/Scripts/python -m pytest --pyargs GrappleIntent.tests

# Train reflexive model on synthetic data (W&B offline unless logged in)
python -m GrappleIntent.training.train_reflexive

# Export to ONNX (verifies eager/ONNX parity)
python -m GrappleIntent.inference.export_onnx

# Benchmark latency against the 10ms budget
python -m GrappleIntent.evaluation.latency_bench

# Run the sidecar (requires the C# pipeline for shared-memory arenas)
python -m GrappleIntent
```

To have the C# `PythonProcessManager` launch GrappleIntent instead of
`GrappleDetector.py`, point `python.detectorPath` in `grapple_config.json`
at `src/GrappleIntent/run_grapple_intent.py`.

## Structure

```
configs/          # YAML configs + typed dataclass loader
models/
  reflexive/      # MobileNetV3 dual-head (cursor + gesture)
  semantic/       # VL-Transformer + intent field head
  adapters/       # Prototypical networks + LoRA
  token_types.py  # Multimodal token definitions (§3)
training/         # Training loops (supervised + calibration)
inference/        # ONNX engines, blending, degradation, export
data/             # MediaPipe landmark extraction, synthetic data, datasets
evaluation/       # Latency benchmarks, metrics
integration/      # Shared memory bridge to C# pipeline
tests/            # Unit + integration tests
runtime.py        # Sidecar main loop (frame → landmarks → model → arena)
run_grapple_intent.py  # Launcher for the C# PythonProcessManager
```

## Implementation Status

- [x] Project scaffolding + configs
- [x] Reflexive model (MobileNetV3 dual-head)
- [x] Prototypical network calibration
- [x] ONNX export pipeline
- [x] Reflexive inference engine + watchdog
- [x] Potential field blending
- [x] Degradation state machine (L0-L3)
- [x] Multimodal token types
- [x] Semantic model (VL-Transformer + intent field)
- [x] Foveated image preprocessing
- [x] Latency benchmarking
- [x] FlatBuffer arena bridge (integration) — generated protocol code, header-driven geometry
- [x] Training loop (reflexive)
- [x] Training data pipeline (synthetic generation, seeded + schema-validated)
- [x] MediaPipe landmark extraction (frame → 66-dim feature vector)
- [x] Sidecar runtime loop + CLI entry points (train / export / bench / run)
- [x] W&B experiment tracking wiring (offline fallback, config + git hash logged)
- [x] End-to-end integration test with C# pipeline (headless arena round-trip + cross-language golden buffer)
- [x] Real-data recording pipeline (guided protocol CLI, .npz + hashed YAML, real+synthetic mixing)
- [ ] Semantic training loop
- [ ] INT4-AWQ quantization pipeline
- [ ] LoRA calibration for semantic path
- [ ] Speculative trajectory decoding (Phase 3)
- [ ] IRL error-correction module (Phase 3)

## Reference

See `.claude/rules/vla-architecture.md` for the full specification.
