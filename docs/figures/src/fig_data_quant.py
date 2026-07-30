"""Generate raw data for the quantization figure: sensitivity, latency, parity.

Run from the repo root with the project venv:
    .venv/Scripts/python docs/figures/src/fig_data_quant.py
Requires the trained fp32 + int8 ONNX artifacts in checkpoints/reflexive/.
Outputs to docs/figures/data/ (gitignored — regenerable).
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SCRATCH = Path(__file__).resolve().parents[1] / "data"
SCRATCH.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "src"))

import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

from GrappleIntent.data.synthetic import SyntheticConfig, generate_reflexive_dataset
from GrappleIntent.inference.quantize_onnx import (
    LandmarkCalibrationReader, _preprocess, evaluate_parity,
)

FP32 = REPO / "checkpoints/reflexive/mobilenetv3_cursor_v0.1_fp32.onnx"
INT8 = REPO / "checkpoints/reflexive/mobilenetv3_cursor_v0.1_int8.onnx"

features, targets, labels = generate_reflexive_dataset(SyntheticConfig(seed=42))
rng = np.random.default_rng(42)
perm = rng.permutation(features.shape[0])
calib = features[perm[:500]]
sel = perm[500:][:1000]
ev = perm[1500:][:2000]

# ── 1. Per-group sensitivity (mirror select_conv_exclusions probes) ──────────
tmp = Path(tempfile.mkdtemp())
pre = tmp / "pre.onnx"
_preprocess(FP32, pre)
convs = [n.name for n in onnx.load(str(pre)).graph.node if n.op_type == "Conv"]
bounds = np.linspace(0, len(convs), 5, dtype=int)
groups = [convs[bounds[i]:bounds[i + 1]] for i in range(4)]

sensitivity = []
for i, group in enumerate(groups):
    probe = tmp / f"probe_{i}.onnx"
    quantize_static(
        model_input=str(pre), model_output=str(probe),
        calibration_data_reader=LandmarkCalibrationReader(calib),
        quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv"],
        nodes_to_exclude=[c for c in convs if c not in group],
    )
    r = evaluate_parity(FP32, probe, features[sel], targets[sel], labels[sel])
    sensitivity.append({
        "group": i + 1, "num_convs": len(group),
        "agreement": r.gesture_agreement,
        "mse_rel_increase": r.cursor_mse_rel_increase,
    })
    print(f"group {i+1}: agree={r.gesture_agreement:.4f} mse+{r.cursor_mse_rel_increase:.1%}")

# ── 2. Latency samples through the engine protocol (100 warmup, 1000 timed) ──
def bench(path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 66).astype(np.float32)
    for _ in range(100):
        sess.run(None, {"landmarks": dummy})
    lat = []
    for _ in range(1000):
        t0 = time.perf_counter()
        sess.run(None, {"landmarks": dummy})
        lat.append((time.perf_counter() - t0) * 1000)
    return np.array(lat)

lat_fp32 = bench(FP32)
lat_int8 = bench(INT8)
for name, lat in (("fp32", lat_fp32), ("int8", lat_int8)):
    q = np.percentile(lat, [50, 95, 99])
    print(f"{name}: P50={q[0]:.2f} P95={q[1]:.2f} P99={q[2]:.2f}")

# ── 3. Parity outputs on final-eval split ────────────────────────────────────
def run_model(path, feats):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    cursor, logits = [], []
    for i in range(0, feats.shape[0], 256):
        out = sess.run(None, {"landmarks": feats[i:i + 256]})
        cursor.append(out[0]); logits.append(out[1])
    return np.concatenate(cursor), np.concatenate(logits)

c_fp32, g_fp32 = run_model(FP32, features[ev])
c_int8, g_int8 = run_model(INT8, features[ev])
report = evaluate_parity(FP32, INT8, features[ev], targets[ev], labels[ev])
print(f"final parity: agree={report.gesture_agreement:.4f} mse+{report.cursor_mse_rel_increase:.1%}")

np.savez(
    SCRATCH / "fig_quant_data.npz",
    lat_fp32=lat_fp32, lat_int8=lat_int8,
    cursor_fp32=c_fp32, cursor_int8=c_int8,
    agree_mask=(g_fp32.argmax(1) == g_int8.argmax(1)),
)
(SCRATCH / "fig_quant_meta.json").write_text(json.dumps({
    "sensitivity": sensitivity,
    "num_convs_total": len(convs),
    "gesture_agreement": report.gesture_agreement,
    "mse_rel_increase": report.cursor_mse_rel_increase,
    "fp32_p": list(np.percentile(lat_fp32, [50, 95, 99])),
    "int8_p": list(np.percentile(lat_int8, [50, 95, 99])),
}, indent=1))
print("saved fig_quant_data.npz + fig_quant_meta.json")
