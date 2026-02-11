# VLA Architecture Rules — GrappleIntent

## System Overview

GrappleIntent is a **hierarchical Vision-Language-Action (VLA)** system that replaces rule-based gesture detection with learned models. It runs as a sidecar alongside the existing Grapple pipeline.

```
Camera → SharedMemoryArena → [GrappleIntent] → FlatBufferSensorArena → MouseControllerNode
                                    │
                         ┌──────────┴──────────┐
                    Reflexive Path         Semantic Path
                    (MobileNetV3)         (VL-Transformer)
                     120Hz, ≤10ms          10Hz, ≤100ms
                    cursor control        intent parsing
```

---

## 1. Dual Inference Path Architecture

### Reflexive Path (Fast)
- **Model:** Quantized MobileNetV3 (or equivalent lightweight backbone)
- **Rate:** 120Hz (matching cursor update loop)
- **Latency:** ≤10ms hard limit (≤5ms target)
- **Input:** Raw hand landmarks + velocity vector
- **Output:** Cursor delta (dx, dy) + gesture confidence
- **Quantization:** INT4-AWQ (salient weight preservation)
- **Runtime:** Must run in-process or with <1ms IPC overhead

### Semantic Path (Slow)
- **Model:** Vision-Language Transformer
- **Rate:** 10Hz (decoupled from cursor)
- **Latency:** ≤100ms hard limit (≤50ms target)
- **Input:** Image patch + gaze vector + hand velocity + UI context
- **Output:** Intent classification + probabilistic target heatmap
- **Purpose:** "Select that button", "scroll this panel", contextual actions
- **Runtime:** Can tolerate slightly higher IPC latency

### Fusion Rule
The reflexive path runs continuously. The semantic path overrides reflexive output when:
1. Intent confidence > threshold (configurable)
2. Semantic prediction is fresh (within TTL window)
3. No active reflexive gesture (pinch/drag takes priority)

---

## 2. Integration Contract with C# Pipeline

### Output Format
GrappleIntent writes to `FlatBufferSensorArena` using the existing FlatBuffer schema. The C# side does NOT know or care that the data came from a neural network vs rule-based logic.

**This is the integration boundary.** Everything upstream (model inference, token fusion, intent fields) is Python/ONNX. Everything downstream (cursor extrapolation, input injection, DPI scaling) is C#.

### Shared Memory Protocol
- GrappleIntent writes `SensorFrame` FlatBuffers (same as current `GrappleDetector.py`)
- Arena magic: `0x4C505247` ("GRPL")
- Sequence number: monotonically increasing
- Timestamp: QPC ticks from original frame capture

### Zero-GC Boundary
- Python/ONNX Runtime can allocate freely (it's a separate process)
- The **only** constraint: the FlatBuffer written to shared memory must be valid
- C# reads with pre-allocated buffer (existing `FlatBufferSensorArena` pattern)

---

## 3. Multimodal Token Fusion

### Token Types
```
<ImagePatch>    — Cropped hand region from camera frame (224×224 RGB)
<GazeVector>    — Eye tracking direction (when available, else estimated from head pose)
<HandVelocity>  — 3D velocity vector from temporal differencing
<UIContext>      — Screen region descriptor (future: from accessibility API)
```

### Fusion Strategy
- Early fusion for reflexive path (concatenate features before final layers)
- Cross-attention fusion for semantic path (each modality attends to others)
- Missing modalities get learned null tokens (graceful degradation)

### Convention
When defining new token types:
1. Document the tensor shape and dtype
2. Define the null/missing representation
3. Add to the fusion registry (config-driven, not hardcoded)

---

## 4. Probabilistic Intent Fields

### Concept
Replace deterministic "cursor at (x, y)" with a 2D Gaussian probability distribution over the screen, modulated by UI saliency.

```
P(target | hand, gaze, ui) = Gaussian(μ_hand, Σ) × Saliency(ui_elements)
```

### Implementation Rules
- Intent field resolution: configurable (default 64×64 grid over virtual desktop)
- Gaussian parameters (μ, Σ) are model outputs, not hardcoded
- Saliency map is optional input — system works without it (uniform prior)
- Final cursor position = argmax of intent field (or expected value for smooth motion)
- Log intent field entropy to W&B — high entropy = uncertain, low = confident

### "Midas Touch" Solution
The intent field solves accidental activation by requiring both:
1. High spatial confidence (low Gaussian variance → user is targeting something specific)
2. Gesture confirmation (pinch gesture from reflexive path)

Only when both conditions are met does a "click" fire.

---

## 5. LoRA Adapter Management

### Calibration Protocol
1. User performs 5-10 anchor gestures (predefined poses)
2. System fine-tunes LoRA adapter on frozen backbone (< 1 minute)
3. Adapter saved as `lora_adapter_{user_id}_v{N}.safetensors`
4. Hot-swappable at runtime (≤500ms adapter switch)

### Rules
- Base model is NEVER modified — LoRA only
- Adapter rank: start with r=8, increase only if accuracy insufficient
- Store adapters per-user, version them
- Validate adapter doesn't regress base model performance on standard test set

---

## 6. Speculative Trajectory Decoding

### Concept
Adapted from LLM speculative decoding: batch-predict K future cursor positions, verify against physics constraints, commit the valid prefix.

### Rules
- Speculation horizon: configurable (default K=5 frames = ~42ms at 120Hz)
- Physics constraints: max velocity, max acceleration, screen bounds
- If speculation is invalidated by new sensor data, discard and re-predict
- Never speculate across gesture boundaries (e.g., don't speculate through a pinch)
- Log speculation hit rate to W&B and telemetry

---

## 7. IRL Error-Correction Module

### Concept
Learn personalized reward functions from observed user self-corrections:
- User pinches → misses target → releases → moves → pinches again
- This "retry pattern" is a natural reward signal

### Rules
- Collect correction trajectories passively (no explicit labeling required)
- Reward function is per-user (part of LoRA adapter context)
- Update reward asynchronously (not in the inference hot path)
- Privacy: correction data stays on-device, never uploaded

---

## 8. Runtime Options

GrappleIntent must support multiple inference backends. The choice is deployment-dependent.

| Runtime | Pros | Cons | When to Use |
|---------|------|------|-------------|
| ONNX Runtime (CPU) | Portable, no GPU needed | Slower for large models | Default / laptops |
| ONNX Runtime (DirectML) | GPU-accelerated, Windows-native | DirectML maturity varies | Windows desktops with GPU |
| TensorRT | Fastest NVIDIA inference | NVIDIA-only, complex setup | High-perf workstations |
| PyTorch (eager) | Easy debugging | Slow, high memory | Development/training only |
| Custom C++ | Maximum control | High engineering cost | Future (1.2+) |

### Convention
- Training always uses PyTorch
- Inference export targets ONNX first (widest compatibility)
- Benchmark all target runtimes before choosing for a deployment tier
- Runtime selection is config-driven (`grapple_config.json`), not hardcoded

---

## 9. Latency Accounting

Every component in the VLA pipeline must declare its latency budget:

```
Reflexive Path (120Hz, ≤10ms total):
├── Frame read from arena:     <0.5ms
├── Preprocessing:             ≤1ms
├── MobileNetV3 inference:     ≤5ms
├── Postprocessing:            ≤1ms
├── FlatBuffer write:          <0.5ms
└── Headroom:                  ~2ms

Semantic Path (10Hz, ≤100ms total):
├── Frame + context read:      ≤2ms
├── Token fusion:              ≤5ms
├── VL-Transformer inference:  ≤70ms
├── Intent field computation:  ≤10ms
├── FlatBuffer write:          <1ms
└── Headroom:                  ~12ms
```

**Rule:** If any component exceeds its budget, it must be profiled and optimized before merging. Log per-component latency breakdowns in benchmarks.
