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
# Install
pip install -e ".[dev]" --break-system-packages

# Run tests
pytest tests/ -v

# Train reflexive model (once data is available)
python -m GrappleIntent.training.train_reflexive

# Export to ONNX
python -m GrappleIntent.inference.export_onnx

# Benchmark latency
python -m GrappleIntent.evaluation.latency_bench
```

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
data/             # Foveated preprocessing, data loading
evaluation/       # Latency benchmarks, metrics
integration/      # Shared memory bridge to C# pipeline
tests/            # Unit + integration tests
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
- [x] FlatBuffer arena bridge (integration)
- [x] Training loop (reflexive)
- [ ] Training data pipeline (synthetic generation)
- [ ] Semantic training loop
- [ ] INT4-AWQ quantization pipeline
- [ ] LoRA calibration for semantic path
- [ ] Speculative trajectory decoding (Phase 3)
- [ ] IRL error-correction module (Phase 3)
- [ ] End-to-end integration test with C# pipeline

## Reference

See `.claude/rules/vla-architecture.md` for the full specification.
