"""Tests for the reflexive path model and inference engine."""

import numpy as np
import pytest
import torch

from ..configs import load_config
from ..models.reflexive.model import ReflexiveModel, ReflexiveOutput


class TestReflexiveModel:
    """Test the MobileNetV3-based reflexive model."""

    @pytest.fixture
    def model(self):
        return ReflexiveModel(
            backbone_name="mobilenetv3_small_100",
            input_dim=66,
            cursor_output_dim=2,
            gesture_classes=5,
            dropout=0.0,
        )

    @pytest.fixture
    def dummy_input(self):
        return torch.randn(4, 66)  # Batch of 4

    def test_forward_output_type(self, model, dummy_input):
        output = model(dummy_input)
        assert isinstance(output, ReflexiveOutput)

    def test_cursor_delta_shape(self, model, dummy_input):
        output = model(dummy_input)
        assert output.cursor_delta.shape == (4, 2)

    def test_cursor_delta_bounded(self, model, dummy_input):
        """Cursor delta should be in [-1, 1] due to Tanh."""
        output = model(dummy_input)
        assert output.cursor_delta.min() >= -1.0
        assert output.cursor_delta.max() <= 1.0

    def test_gesture_logits_shape(self, model, dummy_input):
        output = model(dummy_input)
        assert output.gesture_logits.shape == (4, 5)

    def test_gesture_confidence_range(self, model, dummy_input):
        output = model(dummy_input)
        assert output.gesture_confidence.shape == (4,)
        assert (output.gesture_confidence >= 0.0).all()
        assert (output.gesture_confidence <= 1.0).all()

    def test_embedding_shape(self, model, dummy_input):
        output = model(dummy_input)
        assert output.embedding.shape == (4, 128)  # Default embed_dim

    def test_get_embedding(self, model, dummy_input):
        """get_embedding should return same embeddings as forward."""
        emb = model.get_embedding(dummy_input)
        assert emb.shape == (4, 128)

    def test_single_sample(self, model):
        """Model should work with batch size 1 in eval mode (the 120Hz
        inference scenario — train-mode batch-1 is forbidden by BatchNorm)."""
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 66))
        assert output.cursor_delta.shape == (1, 2)

    def test_gradients_flow(self, model, dummy_input):
        """Ensure gradients flow through both heads."""
        output = model(dummy_input)
        loss = output.cursor_delta.sum() + output.gesture_logits.sum()
        loss.backward()
        # Check that projection layer has gradients
        assert model.projection.proj[0].weight.grad is not None


class TestWatchdog:
    """Test the reflexive watchdog fallback mechanism."""

    def test_triggers_on_high_latency(self):
        from ..inference.reflexive_engine import ReflexiveWatchdog

        wd = ReflexiveWatchdog(threshold_ms=8.0, recovery_frames=10)
        assert not wd.is_passthrough
        assert wd.check(9.0)  # Over threshold
        assert wd.is_passthrough

    def test_recovery_requires_consecutive_frames(self):
        from ..inference.reflexive_engine import ReflexiveWatchdog

        wd = ReflexiveWatchdog(threshold_ms=8.0, recovery_frames=3)
        wd.check(9.0)  # Trigger
        assert wd.is_passthrough

        # Recover with 3 consecutive good frames
        wd.check(5.0)  # 1
        assert wd.is_passthrough
        wd.check(5.0)  # 2
        assert wd.is_passthrough
        wd.check(5.0)  # 3 → recovered
        assert not wd.is_passthrough

    def test_recovery_resets_on_spike(self):
        from ..inference.reflexive_engine import ReflexiveWatchdog

        wd = ReflexiveWatchdog(threshold_ms=8.0, recovery_frames=5)
        wd.check(9.0)  # Trigger

        wd.check(5.0)  # 1
        wd.check(5.0)  # 2
        wd.check(9.0)  # Spike! Reset counter
        assert wd.is_passthrough

        # Need 5 consecutive again
        for _ in range(5):
            wd.check(5.0)
        assert not wd.is_passthrough


class TestBlending:
    """Test potential field blending."""

    def test_pure_reflexive_without_semantic(self):
        from ..inference.blending import PotentialFieldBlender

        blender = PotentialFieldBlender()
        result = blender.blend(0.5, -0.3)
        # No semantic data → α = 0 → pure reflexive
        assert result.alpha == 0.0
        assert result.dx == 0.5
        assert result.dy == -0.3

    def test_semantic_contribution(self):
        from ..inference.blending import PotentialFieldBlender

        blender = PotentialFieldBlender(gain_k=10.0, offset_c0=0.5, decay_tau=10.0)
        blender.update_semantic(0.1, 0.2, confidence=0.9)
        result = blender.blend(0.5, -0.3)
        # High confidence + fresh data → α > 0
        assert result.alpha > 0.0
        assert result.dx != 0.5  # Semantic added something


class TestDegradation:
    """Test degradation state machine."""

    def test_initial_state(self):
        from ..inference.degradation import DegradationLevel, DegradationManager

        mgr = DegradationManager()
        assert mgr.level == DegradationLevel.L1_SEMANTIC_UNAVAILABLE

    def test_semantic_loaded(self):
        from ..inference.degradation import DegradationLevel, DegradationManager

        mgr = DegradationManager()
        mgr.on_semantic_loaded()
        assert mgr.level == DegradationLevel.L0_FULL_VLA

    def test_degradation_hierarchy(self):
        from ..inference.degradation import DegradationLevel, DegradationManager

        mgr = DegradationManager()
        mgr.on_semantic_loaded()
        assert mgr.level == DegradationLevel.L0_FULL_VLA

        mgr.on_semantic_failed("OOM")
        assert mgr.level == DegradationLevel.L1_SEMANTIC_UNAVAILABLE

        mgr.on_reflexive_over_budget()
        assert mgr.level == DegradationLevel.L2_REFLEXIVE_OVER_BUDGET

    def test_recovery_ramp(self):
        from ..inference.degradation import DegradationLevel, DegradationManager

        mgr = DegradationManager(semantic_ramp_seconds=1.0)
        mgr.on_semantic_loaded()
        # Right after loading, alpha_multiplier should be near 0 (start of ramp)
        assert mgr.alpha_multiplier < 0.1  # Very early in ramp


class TestConfig:
    """Test configuration loading."""

    def test_default_config(self):
        config = load_config()
        assert config.reflexive.model.input_dim == 66
        assert config.reflexive.inference.latency_budget_ms == 10.0
        assert config.system.device_reflexive == "cpu"
        assert config.blending.decay_tau == 0.3

    def test_reflexive_always_cpu(self):
        config = load_config()
        assert config.system.device_reflexive == "cpu"
