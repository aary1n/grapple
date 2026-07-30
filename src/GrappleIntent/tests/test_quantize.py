"""Tests for the INT8 quantization pipeline (inference/quantize_onnx.py)."""

from pathlib import Path

import numpy as np
import pytest

from ..inference.quantize_onnx import LandmarkCalibrationReader, evaluate_parity

FP32_ONNX = Path("checkpoints/reflexive/mobilenetv3_cursor_v0.1_fp32.onnx")
INT8_ONNX = Path("checkpoints/reflexive/mobilenetv3_cursor_v0.1_int8.onnx")


class TestCalibrationReader:
    def test_batching_covers_all_samples(self):
        features = np.arange(10 * 66, dtype=np.float32).reshape(10, 66)
        reader = LandmarkCalibrationReader(features, batch_size=4)

        batches = []
        while (batch := reader.get_next()) is not None:
            assert set(batch) == {"landmarks"}
            batches.append(batch["landmarks"])

        assert [b.shape[0] for b in batches] == [4, 4, 2]
        np.testing.assert_array_equal(np.concatenate(batches), features)

    def test_rewind(self):
        features = np.zeros((5, 66), dtype=np.float32)
        reader = LandmarkCalibrationReader(features, batch_size=5)
        assert reader.get_next() is not None
        assert reader.get_next() is None
        reader.rewind()
        assert reader.get_next() is not None

    def test_casts_to_float32(self):
        features = np.zeros((3, 66), dtype=np.float64)
        reader = LandmarkCalibrationReader(features)
        assert reader.get_next()["landmarks"].dtype == np.float32


class TestParityGates:
    @pytest.fixture
    def eval_data(self):
        from ..data.synthetic import SyntheticConfig, generate_reflexive_dataset

        f, t, l = generate_reflexive_dataset(
            SyntheticConfig(num_sequences=10, frames_per_sequence=20, seed=5)
        )
        return f, t, l

    @pytest.mark.skipif(not FP32_ONNX.exists(), reason="fp32 artifact not present")
    def test_model_vs_itself_is_perfect_parity(self, eval_data):
        f, t, l = eval_data
        report = evaluate_parity(FP32_ONNX, FP32_ONNX, f, t, l)
        assert report.passed
        assert report.gesture_agreement == 1.0
        assert report.cursor_mse_rel_increase == 0.0
        assert report.failures == []

    @pytest.mark.skipif(
        not (FP32_ONNX.exists() and INT8_ONNX.exists()),
        reason="quantized artifact not present",
    )
    def test_shipped_int8_meets_adr002_gates(self, eval_data):
        """The committed-config artifact must satisfy the ADR-002 gates."""
        f, t, l = eval_data
        report = evaluate_parity(FP32_ONNX, INT8_ONNX, f, t, l)
        assert report.passed, report.failures
