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
