# VLA Architecture Rules — GrappleIntent

## System Overview

GrappleIntent is a **hierarchical Vision-Language-Action (VLA)** system that replaces rule-based gesture detection with learned models. It runs as a sidecar alongside the existing Grapple pipeline.

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

---

## 1. Dual Inference Path Architecture

### Reflexive Path (Fast)
- **Model:** Quantized MobileNetV3 (or equivalent lightweight backbone)
- **Rate:** 120Hz (matching cursor update loop)
- **Latency:** ≤10ms hard limit (≤5ms target)
- **Input:** Raw hand landmarks + velocity vector
- **Output:** Cursor delta (dx, dy) + gesture confidence
- **Quantization:** ONNX static INT8, per-channel (see ADR-002 — INT4-AWQ is CUDA/LLM-only and infeasible on the CPU-pinned reflexive path)
- **Runtime:** ONNX Runtime CPU with AVX-512. **Always CPU — never GPU.** Deterministic latency, no VRAM contention, no GC pauses from GPU driver.

### Semantic Path (Slow)
- **Model:** Vision-Language Transformer
- **Rate:** 10Hz (decoupled from cursor)
- **Latency:** ≤100ms hard limit (≤50ms target)
- **Input:** Dual-scale foveated image + gaze vector + hand velocity + UI context (see Section 3)
- **Output:** Intent classification + attractive gradient field over screen space
- **Purpose:** "Select that button", "scroll this panel", contextual actions
- **Runtime:** GPU-exclusive via ONNX Runtime DirectML (default) or TensorRT (NVIDIA). GPU is reserved for this path.

### Potential Field Blending (Fusion)

The reflexive and semantic paths are fused via **continuous potential field blending**, never discrete switching. The semantic path outputs an attractive gradient field, not a coordinate override.

**Blending equation:**
```
cursor_delta = reflexive_delta + α · semantic_gradient
```

**Blending coefficient α:**
- `α = sigmoid(k · (semantic_confidence - c₀))` where `k` and `c₀` are configurable gain and offset
- α is always continuous — no hard thresholds, no binary mode switching
- Range: α ∈ [0, 1], smoothly modulated by confidence

**Temporal decay (staleness):**
- When no fresh semantic prediction arrives: `α *= exp(-Δt / τ)` where τ is the configurable decay time constant
- This replaces any hard TTL window — if the semantic path goes silent, α decays exponentially toward 0
- As `Δt → ∞`, `α → 0` gracefully (pure reflexive control)

**Invariants:**
- The reflexive path ALWAYS contributes to cursor_delta — it is never suppressed
- The semantic gradient modulates, it never overrides
- Active reflexive gestures (pinch/drag) naturally dominate because they produce large reflexive_delta values

### Reflexive Watchdog

If reflexive inference latency exceeds **8ms** (80% of 10ms budget):
1. Log a **CRITICAL** warning with measured latency
2. Immediately fall back to **raw landmark passthrough** — forward sensor data directly to FlatBuffer output without model inference
3. The cursor must NEVER stall waiting for a model
4. Resume model inference when latency returns below threshold for 10 consecutive frames

---

## 2. Integration Contract with C# Pipeline

### Output Format
GrappleIntent writes to `FlatBufferSensorArena` using the existing FlatBuffer schema. The C# side does NOT know or care that the data came from a neural network vs rule-based logic.

**This is the integration boundary.** Everything upstream (model inference, potential field blending, intent fields) is Python/ONNX. Everything downstream (cursor extrapolation, input injection, DPI scaling) is C#.

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
<ImagePatch_Global>  — Low-res downscale of full screen/region
                       Shape: (3, 112, 112) float32
                       Purpose: spatial layout understanding
                       Null: zeros tensor

<ImagePatch_Foveal>  — High-res crop centered on current cursor/gaze position
                       Shape: (3, 224, 224) float32
                       Purpose: spatial disambiguation of nearby targets
                       Null: zeros tensor

<GazeVector>         — Eye tracking direction (when available, else estimated from head pose)
                       Shape: (3,) float32 — unit vector
                       Null: (0, 0, -1) — forward-facing default

<HandVelocity>       — 3D velocity vector from temporal differencing
                       Shape: (3,) float32 — units/sec in normalized space
                       Null: zeros

<UIContext>           — Screen region descriptor (future: from OS accessibility API)
                       Shape: variable-length token sequence
                       Null: learned [NO_CONTEXT] token
```

### Dual-Scale Foveated Input (Semantic Path)

The semantic path uses two image scales to address the resolution mismatch between ViT input size and 4K screen content:

- **Global context** (`<ImagePatch_Global>`): Captures spatial layout. The model understands where things are on screen.
- **Foveal crop** (`<ImagePatch_Foveal>`): High-res window around the user's point of attention. The model disambiguates which of several nearby targets the user is pointing at.

**Scoping note:** The foveated crop solves **spatial precision** (disambiguating adjacent buttons), NOT text reading. Semantic UI understanding (reading labels like "Save", "Cancel") is the domain of the `<UIContext>` token, fed by the **Windows UI Automation accessibility API**. This is a future milestone — do not attempt OCR as a VLA subproblem.

**Roadmap:**
1. Phase 0: Foveated attention for spatial precision (current)
2. Phase 1: Windows UI Automation integration for `<UIContext>` token (enables "click Save")
3. Phase 2: Cross-application context via accessibility tree traversal

### Fusion Strategy
- Early fusion for reflexive path (concatenate features before final layers)
- Cross-attention fusion for semantic path (each modality attends to others, including both image scales)
- Missing modalities get their defined null tokens (graceful degradation)

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
- The **gradient** of the intent field is what feeds into potential field blending (Section 1)
- Log intent field entropy to W&B — high entropy = uncertain, low = confident

### "Midas Touch" Solution
The intent field solves accidental activation by requiring both:
1. High spatial confidence (low Gaussian variance → user is targeting something specific)
2. Gesture confirmation (pinch gesture from reflexive path)

Only when both conditions are met does a "click" fire.

---

## 5. Calibration & Personalization

User calibration uses a **split strategy** by inference path. See [ADR-001](.claude/rules/adr-001-calibration-strategy.md) for the full decision rationale.

### Reflexive Path → Prototypical Networks (Metric Learning)

For discrete gesture classification on the lightweight reflexive model:
- **Method:** Compute and store frozen anchor embeddings from a pre-trained embedding network. Classify at runtime via cosine distance to prototypes.
- **No weight updates.** The embedding model is frozen. Calibration is a forward pass, not training.
- **Stable with few shots:** 5-10 anchor gestures are sufficient because we're computing centroids in embedding space, not optimizing parameters.
- **Latency:** Nearest-neighbor lookup adds <0.5ms to the reflexive budget — well within the 5ms target.
- **Storage:** `prototypes_{user_id}_v{N}.npz` — lightweight numpy arrays, not model weights.

### Semantic Path → LoRA Fine-Tuning

For nuanced intent personalization on the VL-Transformer:
- **Method:** LoRA adapters on frozen backbone (rank r=8 default, increase only if accuracy insufficient)
- **Budget:** ≤100ms latency allows the richer expressiveness that fine-tuning provides
- **Storage:** `lora_adapter_{user_id}_v{N}.safetensors`
- **Hot-swappable:** ≤500ms adapter switch at runtime
- **Validation:** Every adapter must be tested against the standard evaluation set to detect regression

### Calibration Protocol (Both Paths)

1. User performs 5-10 anchor gestures (predefined poses)
2. **Augmentation required:** All anchor data is augmented with temporal jittering and synthetic perturbations before any learning (metric or LoRA). This is non-negotiable — raw few-shot data is insufficient.
3. Reflexive: compute prototype embeddings (instant — seconds)
4. Semantic: fine-tune LoRA adapter (< 1 minute)
5. Validate both paths against standard test set
6. Log calibration metrics to W&B

### Rules
- Reflexive base embedding model is NEVER modified — prototype lookup only
- Semantic base VL-Transformer is NEVER modified — LoRA only
- Store calibration artifacts per-user, version them
- Augmentation pipeline must be deterministic given the same seed

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
- Reward function is per-user (part of semantic path LoRA context)
- Update reward asynchronously (not in the inference hot path)
- Privacy: correction data stays on-device, never uploaded

---

## 8. Runtime & Compute Isolation

The reflexive and semantic paths have **hard compute isolation**. They do not share accelerator resources.

### Reflexive Path (CPU-Pinned)

| Attribute | Value |
|-----------|-------|
| Runtime | ONNX Runtime CPU (AVX-512) |
| Deployment | **Always CPU. Not configurable.** |
| Rationale | Deterministic latency, no GC pauses from GPU driver, no VRAM contention |
| Quantization | ONNX static INT8, per-channel (ADR-002) |
| Watchdog | >8ms → landmark passthrough fallback |

### Semantic Path (GPU)

| Runtime | Pros | Cons | When to Use |
|---------|------|------|-------------|
| ONNX Runtime (DirectML) | Windows-native GPU, widest AMD/Intel/NVIDIA support | DirectML maturity varies | **Default** for Windows desktops |
| TensorRT | Fastest NVIDIA inference | NVIDIA-only, complex setup | High-performance workstations |
| ONNX Runtime (CPU) | No GPU required | May exceed 100ms budget for large VL models | Fallback / laptops without discrete GPU |
| PyTorch (eager) | Easy debugging | Slow, high memory | Development/training only |

### Conventions
- Training always uses PyTorch
- Reflexive inference is **always** ONNX Runtime CPU — this is not a deployment decision
- Semantic inference export targets ONNX first (widest compatibility), runtime is config-driven
- Benchmark semantic path on target GPU before choosing runtime for a deployment tier
- Runtime selection for semantic path is set in `grapple_config.json`

---

## 9. Latency Accounting

Every component in the VLA pipeline must declare its latency budget:

```
Reflexive Path (120Hz, ≤10ms total) — Runtime: CPU / ONNX (AVX-512)
├── Frame read from arena:     <0.5ms
├── Preprocessing:             ≤1ms
├── MobileNetV3 inference:     ≤5ms  [WATCHDOG: >8ms total → passthrough]
├── Prototype lookup:          <0.5ms
├── Postprocessing:            ≤0.5ms
├── Potential field blend:     <0.1ms
├── FlatBuffer write:          <0.5ms
└── Headroom:                  ~1.9ms

Semantic Path (10Hz, ≤100ms total) — Runtime: GPU (DirectML / TensorRT)
├── Frame + context read:      ≤2ms
├── Foveated crop (dual-scale):≤2ms
├── Token fusion:              ≤5ms
├── VL-Transformer inference:  ≤65ms
├── Intent field computation:  ≤10ms
├── Gradient field output:     <1ms
├── FlatBuffer write:          <1ms
└── Headroom:                  ~14ms
```

**Rule:** If any component exceeds its budget, it must be profiled and optimized before merging. Log per-component latency breakdowns in benchmarks.

---

## 10. Graceful Degradation

**Principle: The cursor must always move.**

The VLA is an enhancement layer, not a hard dependency. The system defines a degradation hierarchy that guarantees cursor responsiveness under all failure modes.

### Degradation Hierarchy

| Level | State | Behavior | Trigger |
|-------|-------|----------|---------|
| **L0** | Full VLA | Both paths active, potential field blending (α > 0) | Normal operation |
| **L1** | Semantic unavailable | Pure reflexive control. α = 0. No semantic gradient contribution. | Semantic model loading, crash, OOM, GPU error |
| **L2** | Reflexive over-budget | Raw landmark passthrough — sensor data forwarded directly, no model inference | Watchdog: reflexive latency > 8ms |
| **L3** | Both paths degraded | Fall back to `GrappleDetector.py` rule-based system (Schmitt trigger + state machine) | Reflexive model load failure, CPU saturation |

### Recovery Protocol
- **L1 → L0:** Background thread attempts semantic model reload. On success, α ramps from 0 over 1 second (not instant — avoid jerk).
- **L2 → L0/L1:** Resume model inference when latency returns below 8ms for 10 consecutive frames.
- **L3 → L2/L1:** Requires process restart or operator intervention.

### Logging
- All degradation state transitions are logged to:
  - `GrappleLogger` (structured JSON-lines) with level `Warning` (L1) or `Critical` (L2, L3)
  - `TelemetryCollector` for real-time dashboard visibility
  - W&B (if experiment tracking is active)
- Include: previous level, new level, trigger reason, timestamp
