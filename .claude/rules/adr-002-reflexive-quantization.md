# ADR-002: Reflexive Path Quantization — ONNX Static INT8 Instead of INT4-AWQ

## Status

**Accepted**

---

## Context

The VLA architecture spec (`vla-architecture.md` §1, §8) prescribes INT4-AWQ
quantization for the reflexive path. Feasibility was validated on the actual
deployment configuration (Windows, CPU-only, MobileNetV3 backbone) before
implementation, with these findings:

### INT4-AWQ is not implementable for the reflexive model

1. **Toolchain unavailable on the target platform.** `pip install autoawq`
   fails on this Windows/CPU box (no compatible wheel published; the source
   build errors during build-isolation because its `setup.py` imports torch).
   Verified empirically 2026-07-30 against autoawq latest with torch 2.13 CPU.
2. **CUDA-only inference kernels.** AutoAWQ's INT4 kernels (GEMM/GEMV) require
   CUDA at runtime. The reflexive path is **CPU-pinned by architecture rule**
   (`vla-architecture.md` §8: "Always CPU. Not configurable.") — the kernels
   could never run there even if the toolchain installed.
3. **Wrong model class.** AWQ (and AutoAWQ's API, `AutoAWQForCausalLM`)
   targets transformer LLMs: it scales and packs `nn.Linear` weights in
   attention/MLP blocks, guided by activation statistics. The reflexive model
   is a conv-dominated MobileNetV3 (Conv2d + SE blocks) with small MLP heads —
   the algorithm and the packing format do not apply.
4. **No INT4 conv path in ONNX Runtime CPU.** ORT 1.28 exposes
   `QuantType.QInt4`, but 4-bit support is weight-only `MatMulNBits`
   (LLM MatMul path). There are no INT4 convolution kernels for CPU.

### Quantization is margin, not necessity, for this model

The FP32 ONNX model already benchmarks at P50 0.39ms / P95 0.58ms / P99
0.75ms against the 10ms hard budget (17× headroom). Quantization here buys
power/headroom on weaker deployment hardware and validates the workflow — it
is not required to meet the current budget.

---

## Decision

### Reflexive Path → ONNX Runtime Static INT8 (QDQ)

Quantize the exported reflexive ONNX model with `onnxruntime.quantization.
quantize_static`:

- **Format:** QDQ (QuantizeLinear/DequantizeLinear), the mature path for CNNs
  on ORT CPU (MLAS/oneDNN INT8 kernels)
- **Types:** INT8 activations + INT8 weights, **per-channel** weight scales
  (the CNN analogue of AWQ's "salient weight preservation" intent — per-channel
  scaling protects outlier channels)
- **Calibration:** landmark feature vectors from the training distribution
  (synthetic and/or recorded datasets), a few hundred samples
- **Naming:** `mobilenetv3_cursor_v{ver}_int8.onnx` per ml-research.md §2;
  `grapple_config` / `ReflexiveInferenceConfig.onnx_path` points at the INT8
  artifact

### Acceptance gates (per ml-research.md §7)

An INT8 artifact ships only if, on the held-out validation set:

1. Gesture prediction agreement with FP32 ≥ 99% AND gesture accuracy drop
   vs FP32 < 1 percentage point
2. Cursor-delta MSE increase vs FP32 < 10% relative
3. Latency P95 within the 5ms design target on ORT CPU (report P50/P95/P99,
   100-iteration warmup, pre/post-processing included)

### Refinement: sensitivity-driven conv exclusion (measured, not assumed)

Naive full-graph INT8 failed the gates badly (70% gesture agreement, +56%
cursor MSE): the convs consuming the landmark-projection output — an
unbounded-range pseudo-image, not a normalized photo — are quantization
sensitive, and so is one mid-network stage. Bisection over conv groups
localized the damage; quantizing only the insensitive groups restored parity.

The shipped pipeline (`inference/quantize_onnx.py`) therefore:
1. Partitions Conv nodes into topological groups (default 4)
2. Probes each group quantized alone against a selection split (disjoint from
   calibration and from the final eval split)
3. Excludes groups whose solo run shows >2% relative cursor-MSE increase or
   <99.9% gesture agreement
4. Quantizes the union of insensitive groups; final gates run on the held-out
   eval split

Quantization scope is Conv-only with QUInt8 activations / QInt8 per-channel
weights: including Gemm (head/projection Linears) pushed MSE past the gate,
and QInt8 activations performed no better. MinMax calibration — ORT 1.28's
Percentile/Entropy histogram collector crashes on this graph (inhomogeneous
tensor shapes in the collector).

### Measured results (v0.1, synthetic val, 2026-07-30)

| Metric | FP32 | INT8 (27/53 convs) |
|--------|------|--------------------|
| Gesture agreement vs FP32 | — | 1.0000 |
| Gesture accuracy | 1.0000 | 1.0000 |
| Cursor MSE | 0.00860 | 0.00872 (+1.4%) |
| Latency P50 / P95 | 1.48 / 3.17 ms | 0.83 / 1.61 ms |

All three acceptance gates pass; INT8 is ~2× faster at P50 despite partial
coverage.

### Semantic path is unaffected

INT4-AWQ remains the plan **where it actually applies**: the semantic
VL-Transformer (attention/MLP Linear layers, GPU runtime). This ADR covers
the reflexive path only.

---

## Consequences

### Positive
- Quantization runs entirely on the deployment toolchain (ORT CPU) — no CUDA
  dependency, no extra package
- QDQ INT8 is broadly portable (same artifact runs on ORT CPU/DirectML/mobile)
- Per-channel INT8 on a small CNN typically costs well under 1% accuracy

### Negative
- INT8 model is ~4× larger than a hypothetical INT4 packing (irrelevant at
  MobileNetV3-small scale: a few MB)
- INT8 conv on CPU may not be faster than FP32 at batch 1 for a model this
  small (kernel overhead); the artifact must earn its place via the benchmark,
  and FP32 remains the fallback if it regresses

### Revisit criteria
- Reflexive backbone changes to a transformer-style architecture → re-evaluate
  weight-only INT4 (MatMulNBits)
- Deployment moves to hardware with native INT4 conv support (NPU) → revisit
- `autoawq` (or a successor) ships CPU inference kernels → revisit

---

## References

- Lin, J., et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM
  Compression and Acceleration." MLSys 2024.
- ONNX Runtime quantization docs: static quantization (QDQ), CalibrationDataReader.
- ADR-001 (calibration strategy) for the precedent of splitting methods by path.
