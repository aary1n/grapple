# GrappleIntent Phase 1 Figures

Publication-style figures (Nature double-column, 519 pt / 183 mm wide,
Wong/Okabe-Ito colorblind-safe palette) documenting the Phase 1 results.
PNG previews are 300 DPI renders of the PDFs.

## Figure 1 — `fig1_quantization.pdf`

**Sensitivity-driven INT8 quantization of the reflexive path preserves parity
with a 2× latency improvement (ADR-002).**
**(A)** Per-group quantization sensitivity. Conv nodes of the exported
MobileNetV3 graph were partitioned into four topological groups; each group
was statically quantized alone (QDQ, QUInt8 activations / QInt8 per-channel
weights) and compared against FP32 on a 1,000-sample selection split.
Bars: relative cursor mean-squared-error (MSE) increase (log scale);
percentages above bars: gesture-prediction agreement with FP32. Groups
exceeding the 2% exclusion threshold (dashed line, vermilion bars) remain
FP32 in the shipped artifact; green groups are quantized (27/53 convs).
**(B)** Empirical cumulative distribution of single-frame inference latency
(1,000 iterations after 100-iteration warmup, ONNX Runtime CPU, batch 1)
for FP32 and the shipped INT8 model, against the 5 ms design target and
10 ms hard budget (dashed lines). INT8: P50 0.8 ms, P95 1.4 ms.
**(C)** Cursor-head output parity on a held-out 2,000-sample evaluation
split (1,500 points shown; dx and dy pooled). The dashed line is identity.
Gesture agreement 100.0%; cursor MSE +1.4% relative to FP32.
All data synthetic (`data/synthetic.py`, seed 42).

## Figure 2 — `fig2_semantic.pdf`

**The semantic-path training scaffold converges on synthetic intent fields.**
Vision-Language Transformer (ViT-tiny backbone, 128-d fusion embedding,
512 samples, 15 epochs, CPU) trained on the synthetic intent-field dataset
(`data/synthetic_semantic.py`): a Gaussian target blob rendered into both
image scales, gaze pointing at it, quadrant intent labels.
**(A)** Gaussian negative log-likelihood (NLL) of target points under the
predicted intent field N(μ, Σ). **(B)** Mean squared error of the predicted
field mean μ. **(C)** Intent classification accuracy (4 quadrant classes;
dotted line: chance = 25%). **(D)** Mean intent-field entropy
0.5·logdet(2πeΣ) — decreasing entropy shows the predicted covariance
sharpening as the model gains confidence (vla-architecture.md §4).
Solid green: training split; dashed blue: validation split (10%).

## Regenerating

From the repo root (C# artifacts in `checkpoints/reflexive/` required):

```bash
# 1. Raw data (quantization probes, latency, parity) -> data/
.venv/Scripts/python docs/figures/src/fig_data_quant.py

# 2. Semantic per-epoch history from the W&B offline run -> data/
.venv/Scripts/python docs/figures/src/extract_semantic_history.py wandb/offline-run-<id>

# 3. Panel elements (600-DPI transparent PNGs) -> elements/
uv run --no-project --with matplotlib --with numpy python docs/figures/src/fig_elements.py

# 4. Compose PDFs (one-time: cd docs/figures/src && bun add @react-pdf/renderer react)
cd docs/figures/src && bun run render.tsx
```

`data/` and `elements/` are regenerable intermediates and are gitignored.
