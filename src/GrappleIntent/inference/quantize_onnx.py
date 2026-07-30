"""Static INT8 quantization of the reflexive ONNX model.

Per ADR-002 (.claude/rules/adr-002-reflexive-quantization.md): the reflexive
path uses ONNX Runtime static QDQ INT8 with per-channel weight scales instead
of INT4-AWQ, which is CUDA/LLM-only and cannot run on the CPU-pinned path.

Pipeline: FP32 ONNX → quant_pre_process → quantize_static (calibrated on
landmark features from the training distribution) → parity evaluation vs FP32
→ latency benchmark. The artifact ships only if the ADR-002 acceptance gates
hold:

    1. gesture agreement with FP32 >= 99% and accuracy drop < 1pp
    2. cursor MSE increase < 10% relative
    3. P95 latency within the 5ms design target
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class LandmarkCalibrationReader:
    """CalibrationDataReader feeding landmark feature vectors batch-by-batch."""

    def __init__(self, features: np.ndarray, batch_size: int = 32) -> None:
        self._features = features.astype(np.float32)
        self._batch_size = batch_size
        self._pos = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._pos >= self._features.shape[0]:
            return None
        batch = self._features[self._pos : self._pos + self._batch_size]
        self._pos += self._batch_size
        return {"landmarks": batch}

    def rewind(self) -> None:
        self._pos = 0


def _preprocess(fp32_onnx: Path, preprocessed: Path) -> None:
    """Shape inference + graph optimization prepass (ORT-recommended).

    skip_symbolic_shape: sympy-based inference asserts on the dynamo-exported
    graph (dynamic batch through Conv); standard ONNX shape inference and
    graph optimization still run.
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process

    quant_pre_process(str(fp32_onnx), str(preprocessed), skip_symbolic_shape=True)


def _conv_node_names(model_path: Path) -> list[str]:
    """Conv node names in topological order."""
    import onnx

    model = onnx.load(str(model_path))
    return [n.name for n in model.graph.node if n.op_type == "Conv"]


def _quantize(
    preprocessed: Path,
    output_path: Path,
    calibration_features: np.ndarray,
    nodes_to_exclude: list[str],
    per_channel: bool,
) -> None:
    """One static-quantization run with the ADR-002 recipe.

    Conv-only, QUInt8 activations: full-graph QDQ and Gemm quantization both
    failed the parity gates empirically (see ADR-002 findings). MinMax
    calibration — ORT 1.28's Percentile/Entropy histogram collector crashes
    on this graph (inhomogeneous tensor shapes).
    """
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    quantize_static(
        model_input=str(preprocessed),
        model_output=str(output_path),
        calibration_data_reader=LandmarkCalibrationReader(calibration_features),
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv"],
        nodes_to_exclude=nodes_to_exclude,
    )


def select_conv_exclusions(
    preprocessed: Path,
    calibration_features: np.ndarray,
    fp32_onnx: Path,
    selection_features: np.ndarray,
    selection_targets: np.ndarray,
    selection_labels: np.ndarray,
    per_channel: bool = True,
    num_groups: int = 4,
    max_group_mse_increase: float = 0.02,
    min_group_agreement: float = 0.999,
) -> list[str]:
    """Sensitivity-driven exclusion: probe each conv group quantized alone.

    The early convs (consuming the unbounded landmark-projection activations)
    are quantization-sensitive; which ones exactly shifts with retraining, so
    the sensitive set is measured per artifact rather than hardcoded. Groups
    whose solo parity on the selection split exceeds the thresholds stay FP32.
    """
    convs = _conv_node_names(preprocessed)
    bounds = np.linspace(0, len(convs), num_groups + 1, dtype=int)
    groups = [convs[bounds[i] : bounds[i + 1]] for i in range(num_groups)]

    excluded: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        for i, group in enumerate(groups):
            probe = Path(td) / f"probe_{i}.onnx"
            _quantize(
                preprocessed, probe, calibration_features,
                nodes_to_exclude=[c for c in convs if c not in group],
                per_channel=per_channel,
            )
            report = evaluate_parity(
                fp32_onnx, probe,
                selection_features, selection_targets, selection_labels,
            )
            sensitive = (
                report.cursor_mse_rel_increase > max_group_mse_increase
                or report.gesture_agreement < min_group_agreement
            )
            logger.info(
                "Conv group %d/%d (%d nodes): agreement=%.4f mse=%+.1f%% -> %s",
                i + 1, num_groups, len(group),
                report.gesture_agreement, report.cursor_mse_rel_increase * 100,
                "EXCLUDE (sensitive)" if sensitive else "quantize",
            )
            if sensitive:
                excluded.extend(group)

    return excluded


def quantize_reflexive_int8(
    fp32_onnx: str | Path,
    output_path: str | Path,
    calibration_features: np.ndarray,
    selection_features: np.ndarray | None = None,
    selection_targets: np.ndarray | None = None,
    selection_labels: np.ndarray | None = None,
    per_channel: bool = True,
) -> Path:
    """Statically quantize the FP32 reflexive ONNX model to INT8 (QDQ).

    When a selection split is provided, sensitivity-driven conv exclusion
    runs first (recommended — full-conv quantization fails parity).
    """
    fp32_onnx = Path(fp32_onnx)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessed = output_path.with_suffix(".preprocessed.onnx")
    _preprocess(fp32_onnx, preprocessed)

    try:
        excluded: list[str] = []
        if selection_features is not None:
            excluded = select_conv_exclusions(
                preprocessed, calibration_features, fp32_onnx,
                selection_features, selection_targets, selection_labels,
                per_channel=per_channel,
            )
        _quantize(
            preprocessed, output_path, calibration_features,
            nodes_to_exclude=excluded, per_channel=per_channel,
        )
    finally:
        preprocessed.unlink(missing_ok=True)

    logger.info("Quantized INT8 model written to %s", output_path)
    return output_path


# ─── Parity evaluation ────────────────────────────────────────────────────────


@dataclass
class ParityReport:
    """FP32-vs-INT8 comparison on a labeled validation set (ADR-002 gates)."""

    num_samples: int
    gesture_agreement: float  # fraction of identical argmax predictions
    fp32_gesture_accuracy: float
    int8_gesture_accuracy: float
    fp32_cursor_mse: float
    int8_cursor_mse: float
    cursor_mse_rel_increase: float
    passed: bool
    failures: list[str]


def _run_batched(session, features: np.ndarray, batch_size: int = 256):
    cursor, logits = [], []
    for i in range(0, features.shape[0], batch_size):
        out = session.run(None, {"landmarks": features[i : i + batch_size]})
        cursor.append(out[0])
        logits.append(out[1])
    return np.concatenate(cursor), np.concatenate(logits)


def evaluate_parity(
    fp32_onnx: str | Path,
    int8_onnx: str | Path,
    features: np.ndarray,
    cursor_targets: np.ndarray,
    gesture_labels: np.ndarray,
    min_agreement: float = 0.99,
    max_accuracy_drop: float = 0.01,
    max_mse_rel_increase: float = 0.10,
) -> ParityReport:
    """Compare INT8 against FP32 on a validation set per the ADR-002 gates."""
    import onnxruntime as ort

    fp32 = ort.InferenceSession(str(fp32_onnx), providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(int8_onnx), providers=["CPUExecutionProvider"])

    features = features.astype(np.float32)
    fp32_cursor, fp32_logits = _run_batched(fp32, features)
    int8_cursor, int8_logits = _run_batched(int8, features)

    fp32_pred = fp32_logits.argmax(axis=1)
    int8_pred = int8_logits.argmax(axis=1)

    agreement = float((fp32_pred == int8_pred).mean())
    fp32_acc = float((fp32_pred == gesture_labels).mean())
    int8_acc = float((int8_pred == gesture_labels).mean())
    fp32_mse = float(np.mean((fp32_cursor - cursor_targets) ** 2))
    int8_mse = float(np.mean((int8_cursor - cursor_targets) ** 2))
    rel_increase = (int8_mse - fp32_mse) / fp32_mse if fp32_mse > 0 else 0.0

    failures = []
    if agreement < min_agreement:
        failures.append(
            f"gesture agreement {agreement:.4f} < {min_agreement:.2f}"
        )
    if fp32_acc - int8_acc > max_accuracy_drop:
        failures.append(
            f"accuracy drop {fp32_acc - int8_acc:.4f} > {max_accuracy_drop:.2f}"
        )
    if rel_increase > max_mse_rel_increase:
        failures.append(
            f"cursor MSE increase {rel_increase:.1%} > {max_mse_rel_increase:.0%}"
        )

    return ParityReport(
        num_samples=int(features.shape[0]),
        gesture_agreement=agreement,
        fp32_gesture_accuracy=fp32_acc,
        int8_gesture_accuracy=int8_acc,
        fp32_cursor_mse=fp32_mse,
        int8_cursor_mse=int8_mse,
        cursor_mse_rel_increase=rel_increase,
        passed=not failures,
        failures=failures,
    )


# ─── CLI entry point ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from ..configs import load_config
    from ..data.synthetic import SyntheticConfig, generate_reflexive_dataset
    from ..evaluation.latency_bench import ReflexiveBenchmark

    for stream in (sys.stdout, sys.stderr):
        if stream and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(
        description="Quantize the reflexive ONNX model to static INT8 (ADR-002)"
    )
    parser.add_argument(
        "--fp32",
        default="checkpoints/reflexive/mobilenetv3_cursor_v0.1_fp32.onnx",
        help="Input FP32 ONNX model",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/reflexive/mobilenetv3_cursor_v0.1_int8.onnx",
        help="Output INT8 ONNX path",
    )
    parser.add_argument("--config", default=None, help="GrappleIntent YAML config")
    parser.add_argument("--recorded-data", action="append", default=[],
                        help="Recorded .npz dataset(s) for calibration/eval "
                             "(default: synthetic)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Latency benchmark iterations")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    fp32_path = Path(args.fp32)
    if not fp32_path.exists():
        logger.error("FP32 ONNX not found: %s — export first "
                     "(python -m GrappleIntent.inference.export_onnx)", fp32_path)
        return 1

    config = load_config(args.config)
    qc = config.reflexive.quantization

    # Calibration + eval data from the training distribution
    if args.recorded_data:
        from ..data.dataset import load_recorded_arrays

        features, targets, labels, _ = load_recorded_arrays(args.recorded_data)
        data_source = "recorded"
    else:
        features, targets, labels = generate_reflexive_dataset(
            SyntheticConfig(seed=args.seed)
        )
        data_source = "synthetic"

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(features.shape[0])
    calib_idx = perm[: qc.calibration_samples]
    # Disjoint splits: sensitivity selection must not see the final eval set
    select_idx = perm[qc.calibration_samples :][:1000]
    eval_idx = perm[qc.calibration_samples + 1000 :][:2000]

    logger.info(
        "%s data: %d calibration / %d selection / %d final-eval samples",
        data_source, len(calib_idx), len(select_idx), len(eval_idx),
    )

    int8_path = quantize_reflexive_int8(
        fp32_path, args.output, features[calib_idx],
        selection_features=features[select_idx],
        selection_targets=targets[select_idx],
        selection_labels=labels[select_idx],
        per_channel=qc.per_channel,
    )

    report = evaluate_parity(
        fp32_path, int8_path,
        features[eval_idx], targets[eval_idx], labels[eval_idx],
    )
    print(
        f"Parity ({report.num_samples} samples): "
        f"{'PASS' if report.passed else 'FAIL'}\n"
        f"  gesture agreement:  {report.gesture_agreement:.4f}\n"
        f"  gesture accuracy:   fp32 {report.fp32_gesture_accuracy:.4f} "
        f"-> int8 {report.int8_gesture_accuracy:.4f}\n"
        f"  cursor MSE:         fp32 {report.fp32_cursor_mse:.6f} "
        f"-> int8 {report.int8_cursor_mse:.6f} "
        f"({report.cursor_mse_rel_increase:+.1%})"
    )
    for failure in report.failures:
        print(f"  GATE FAILED: {failure}")

    # Latency: FP32 vs INT8 under the same protocol
    import onnxruntime as ort

    bench = ReflexiveBenchmark(budget_ms=config.reflexive.inference.latency_budget_ms)
    results = {}
    for name, path in (("fp32", fp32_path), ("int8", int8_path)):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        def infer(x, _s=session):
            return _s.run(None, {"landmarks": x.reshape(1, -1)})

        results[name] = bench.run(infer, num_iterations=args.iterations)
        print(f"\n[{name}] " + bench.report(results[name]))

    target = config.reflexive.inference.latency_target_ms
    p95_ok = results["int8"].p95_ms <= target
    print(
        f"\nINT8 P95 {results['int8'].p95_ms:.2f}ms vs {target:.1f}ms design "
        f"target: {'PASS' if p95_ok else 'FAIL'}"
    )

    if not report.passed or not p95_ok:
        logger.error("ADR-002 acceptance gates not met — do not ship this artifact")
        return 1

    logger.info("Quantization complete: %s", int8_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
