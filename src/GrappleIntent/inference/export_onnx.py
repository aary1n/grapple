"""ONNX export for the reflexive path model.

Exports the PyTorch ReflexiveModel to ONNX format with three named outputs:
    - cursor_delta: (1, 2)
    - gesture_logits: (1, num_classes)
    - embedding: (1, embed_dim)

These output names must match what ReflexiveEngine._run_onnx() expects.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..models.reflexive.model import ReflexiveModel

logger = logging.getLogger(__name__)


def export_reflexive_onnx(
    model: ReflexiveModel,
    output_path: str | Path,
    input_dim: int = 66,
    opset_version: int = 17,
) -> Path:
    """Export a trained ReflexiveModel to ONNX.

    Args:
        model: Trained PyTorch model.
        output_path: Where to save the .onnx file.
        input_dim: Input dimension (must match model.input_dim).
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Dummy input for tracing
    dummy = torch.randn(1, input_dim)

    # We need to wrap the model to return a tuple (ONNX doesn't support dataclasses)
    class _OnnxWrapper(torch.nn.Module):
        def __init__(self, inner: ReflexiveModel):
            super().__init__()
            self.inner = inner

        def forward(self, landmarks: torch.Tensor) -> tuple[torch.Tensor, ...]:
            out = self.inner(landmarks, return_embedding=True)
            return out.cursor_delta, out.gesture_logits, out.embedding

    wrapper = _OnnxWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(output_path),
        input_names=["landmarks"],
        output_names=["cursor_delta", "gesture_logits", "embedding"],
        dynamic_axes={
            "landmarks": {0: "batch"},
            "cursor_delta": {0: "batch"},
            "gesture_logits": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info("Exported reflexive ONNX model to %s", output_path)
    return output_path


def verify_onnx(onnx_path: str | Path, input_dim: int = 66) -> bool:
    """Verify an exported ONNX model loads and produces valid outputs."""
    import numpy as np

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )

        dummy = np.random.randn(1, input_dim).astype(np.float32)
        outputs = session.run(None, {"landmarks": dummy})

        assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"
        assert outputs[0].shape == (1, 2), f"cursor_delta shape: {outputs[0].shape}"
        assert outputs[2].ndim == 2, f"embedding shape: {outputs[2].shape}"

        logger.info("ONNX verification passed: %s", onnx_path)
        return True

    except Exception:
        logger.exception("ONNX verification FAILED for %s", onnx_path)
        return False


def verify_parity(
    model: ReflexiveModel,
    onnx_path: str | Path,
    input_dim: int = 66,
    tolerance: float = 1e-4,
    seed: int = 42,
) -> float:
    """Compare ONNX outputs against PyTorch eager on a fixed input.

    Returns the max abs delta across all three outputs. Raises if it exceeds
    tolerance — an export that changes the model's behavior must not ship.
    """
    import numpy as np
    import onnxruntime as ort

    torch.manual_seed(seed)
    dummy = torch.randn(1, input_dim)

    model.eval()
    with torch.no_grad():
        eager = model(dummy, return_embedding=True)
    eager_outputs = [
        eager.cursor_delta.numpy(),
        eager.gesture_logits.numpy(),
        eager.embedding.numpy(),
    ]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"landmarks": dummy.numpy()})

    max_delta = max(
        float(np.abs(e - o).max()) for e, o in zip(eager_outputs, onnx_outputs)
    )
    if max_delta > tolerance:
        raise RuntimeError(
            f"ONNX/eager parity FAILED: max delta {max_delta:.2e} > {tolerance:.0e}"
        )
    logger.info("ONNX/eager parity OK: max output delta %.2e", max_delta)
    return max_delta


# ─── CLI entry point ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from ..configs import load_config

    # Windows consoles are often cp1252; torch's exporter prints unicode
    # status glyphs — replace rather than crash.
    for stream in (sys.stdout, sys.stderr):
        if stream and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Export reflexive model to ONNX")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/reflexive/mobilenetv3_cursor_v0.1_fp32.pt",
        help="Path to trained .pt state dict",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/reflexive/mobilenetv3_cursor_v0.1_fp32.onnx",
        help="Output .onnx path",
    )
    parser.add_argument("--config", default=None, help="GrappleIntent YAML config")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        logger.error("Checkpoint not found: %s — train first "
                     "(python -m GrappleIntent.training.train_reflexive)", checkpoint)
        return 1

    mc = load_config(args.config).reflexive.model
    model = ReflexiveModel(
        backbone_name=mc.backbone,
        input_dim=mc.input_dim,
        cursor_output_dim=mc.cursor_output_dim,
        gesture_classes=mc.gesture_classes,
        dropout=mc.dropout,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))

    onnx_path = export_reflexive_onnx(
        model, args.output, input_dim=mc.input_dim, opset_version=args.opset
    )
    if not verify_onnx(onnx_path, input_dim=mc.input_dim):
        return 1
    verify_parity(model, onnx_path, input_dim=mc.input_dim)
    logger.info("Export complete: %s", onnx_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
