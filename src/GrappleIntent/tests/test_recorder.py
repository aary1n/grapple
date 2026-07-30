"""Tests for the real-data recording pipeline (data/recorder.py)."""

import numpy as np
import pytest
import torch

from ..data.landmarks import FEATURE_DIMS, LandmarkFrame
from ..data.recorder import (
    GestureRecorder,
    GestureSegment,
    RecordingConfig,
    load_recording,
    save_recording,
    segment_to_samples,
    segments_to_arrays,
)
from ..data.synthetic import CURSOR_DELTA_SCALE


def _segment(gesture, wrists, times):
    seg = GestureSegment(gesture=gesture)
    for (x, y), t in zip(wrists, times):
        f = np.zeros(FEATURE_DIMS, dtype=np.float32)
        f[0], f[1] = x, y
        seg.features.append(f)
        seg.wrists.append(np.array([x, y], dtype=np.float64))
        seg.times.append(t)
    return seg


class TestSegmentToSamples:
    def test_next_frame_delta_targets(self):
        seg = _segment(
            gesture=2,
            wrists=[(0.5, 0.5), (0.51, 0.52), (0.53, 0.51)],
            times=[0.0, 0.033, 0.066],
        )
        feats, targets, labels = segment_to_samples(seg, max_frame_gap_s=0.15)

        assert feats.shape == (2, FEATURE_DIMS)
        assert (labels == 2).all()
        np.testing.assert_allclose(
            targets[0], [0.01 * CURSOR_DELTA_SCALE, 0.02 * CURSOR_DELTA_SCALE],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            targets[1], [0.02 * CURSOR_DELTA_SCALE, -0.01 * CURSOR_DELTA_SCALE],
            atol=1e-6,
        )

    def test_targets_clipped_to_tanh_range(self):
        seg = _segment(
            gesture=4, wrists=[(0.1, 0.1), (0.9, 0.9)], times=[0.0, 0.033]
        )
        _, targets, _ = segment_to_samples(seg, max_frame_gap_s=0.15)
        assert np.abs(targets).max() <= 1.0

    def test_detection_gap_drops_sample(self):
        # Frame 1 -> 2 gap of 0.5s (hand lost) must not fabricate a delta
        seg = _segment(
            gesture=0,
            wrists=[(0.5, 0.5), (0.51, 0.5), (0.9, 0.9), (0.91, 0.9)],
            times=[0.0, 0.033, 0.533, 0.566],
        )
        feats, targets, labels = segment_to_samples(seg, max_frame_gap_s=0.15)
        assert feats.shape[0] == 2  # frames 0 and 2 keep their targets
        np.testing.assert_allclose(
            targets[:, 0], [0.01 * CURSOR_DELTA_SCALE, 0.01 * CURSOR_DELTA_SCALE],
            atol=1e-6,
        )

    def test_too_short_segment_yields_empty(self):
        seg = _segment(gesture=1, wrists=[(0.5, 0.5)], times=[0.0])
        feats, targets, labels = segment_to_samples(seg, max_frame_gap_s=0.15)
        assert feats.shape[0] == 0


class TestSegmentsToArrays:
    def test_short_segments_dropped(self):
        good = _segment(
            gesture=1,
            wrists=[(0.5 + 0.001 * i, 0.5) for i in range(10)],
            times=[0.033 * i for i in range(10)],
        )
        short = _segment(
            gesture=2, wrists=[(0.5, 0.5), (0.51, 0.5)], times=[0.0, 0.033]
        )
        cfg = RecordingConfig(min_segment_frames=5)
        feats, targets, labels = segments_to_arrays([good, short], cfg)
        assert (labels == 1).all()  # short pinch segment dropped

    def test_no_usable_samples_raises(self):
        empty = _segment(gesture=0, wrists=[], times=[])
        with pytest.raises(RuntimeError, match="No usable samples"):
            segments_to_arrays([empty], RecordingConfig())


class _FakeSource:
    """Yields frames whose first pixel encodes an incrementing index."""

    def __init__(self):
        self.index = 0

    def __call__(self):
        self.index += 1
        frame = np.full((4, 4, 3), self.index % 251, dtype=np.uint8)
        return frame, self.index * 0.02


class _FakeExtractor:
    """Returns a wrist that drifts right; reuses its feature buffer like the
    real LandmarkExtractor (recorder must copy)."""

    def __init__(self):
        self._features = np.zeros(FEATURE_DIMS, dtype=np.float32)
        self.calls = 0

    def extract(self, frame, time_s):
        self.calls += 1
        x = 0.3 + 0.001 * self.calls
        self._features[0] = x
        self._features[1] = 0.5
        return LandmarkFrame(
            features=self._features,
            wrist_x=x, wrist_y=0.5, wrist_z=0.0,
            velocity_x=0.0, velocity_y=0.0, velocity_z=0.0,
            handedness="Right", detection_confidence=1.0,
        )

    def close(self):
        pass


class TestGestureRecorder:
    def _make_recorder(self, config):
        clock_state = {"t": 0.0}

        def clock():
            clock_state["t"] += 0.01
            return clock_state["t"]

        return GestureRecorder(
            _FakeSource(), _FakeExtractor(), config,
            clock=clock, prompt=lambda *_: None, idle_sleep_s=0.0,
        )

    def test_record_segment_labels_and_copies(self):
        rec = self._make_recorder(RecordingConfig())
        seg = rec.record_segment(gesture=3, seconds=0.5)
        assert seg.gesture == 3
        assert len(seg) > 2
        # Features must be copies, not views of the extractor's reused buffer
        assert seg.features[0][0] != seg.features[-1][0]

    def test_run_protocol_covers_all_gestures_and_passes(self):
        cfg = RecordingConfig(
            seconds_per_gesture=0.2, rest_seconds=0.05,
            gestures=(0, 2), passes=2,
        )
        rec = self._make_recorder(cfg)
        segments = rec.run_protocol()
        assert [s.gesture for s in segments] == [0, 2, 0, 2]
        assert all(len(s) > 0 for s in segments)


class TestPersistence:
    def _arrays(self, n=20):
        rng = np.random.default_rng(0)
        feats = rng.normal(size=(n, FEATURE_DIMS)).astype(np.float32)
        targets = np.clip(rng.normal(size=(n, 2)), -1, 1).astype(np.float32)
        labels = rng.integers(0, 5, size=n).astype(np.int64)
        return feats, targets, labels

    def test_roundtrip_with_hash(self, tmp_path):
        feats, targets, labels = self._arrays()
        npz, yml = save_recording(
            tmp_path / "rec", feats, targets, labels, {"user_id": "test"}
        )
        assert npz.exists() and yml.exists()

        f2, t2, l2, meta = load_recording(npz)
        np.testing.assert_array_equal(f2, feats)
        np.testing.assert_array_equal(t2, targets)
        np.testing.assert_array_equal(l2, labels)
        assert meta["user_id"] == "test"
        assert meta["num_samples"] == 20
        assert len(meta["content_sha256"]) == 64

    def test_tampered_file_rejected(self, tmp_path):
        feats, targets, labels = self._arrays()
        npz, _ = save_recording(tmp_path / "rec", feats, targets, labels, {})
        npz.write_bytes(npz.read_bytes() + b"tamper")
        with pytest.raises(ValueError, match="hash mismatch"):
            load_recording(npz)
        # But loads when verification is explicitly disabled... the appended
        # bytes still parse (npz readers ignore trailing garbage)
        load_recording(npz, verify_hash=False)

    def test_invalid_schema_rejected_on_save(self, tmp_path):
        feats, targets, labels = self._arrays()
        with pytest.raises(AssertionError):
            save_recording(
                tmp_path / "bad", feats, targets * 5.0, labels, {}
            )  # targets outside [-1, 1]


class TestMixedDataloaders:
    def test_mixing_counts_and_shapes(self, tmp_path):
        from ..data.dataset import make_mixed_dataloaders
        from ..data.synthetic import SyntheticConfig

        rng = np.random.default_rng(1)
        n_rec = 40
        feats = rng.normal(size=(n_rec, FEATURE_DIMS)).astype(np.float32)
        targets = np.clip(rng.normal(size=(n_rec, 2)), -1, 1).astype(np.float32)
        labels = rng.integers(0, 5, size=n_rec).astype(np.int64)
        npz, _ = save_recording(tmp_path / "rec", feats, targets, labels, {})

        synth = SyntheticConfig(num_sequences=5, frames_per_sequence=10, seed=7)
        train, val, summary = make_mixed_dataloaders(
            [str(npz)], synth, batch_size=16, seed=3
        )

        assert summary["num_recorded"] == n_rec
        assert summary["num_synthetic"] == 5 * 9  # frames_per_sequence - 1
        total = summary["num_recorded"] + summary["num_synthetic"]
        assert len(train.dataset) + len(val.dataset) == total
        assert summary["recorded_files"][0]["sha256"]

        lm, ct, gl = next(iter(train))
        assert lm.shape[1] == FEATURE_DIMS
        assert ct.shape[1] == 2
        assert gl.dtype == torch.int64

    def test_recorded_only(self, tmp_path):
        from ..data.dataset import make_mixed_dataloaders

        rng = np.random.default_rng(2)
        feats = rng.normal(size=(30, FEATURE_DIMS)).astype(np.float32)
        targets = np.clip(rng.normal(size=(30, 2)), -1, 1).astype(np.float32)
        labels = rng.integers(0, 5, size=30).astype(np.int64)
        npz, _ = save_recording(tmp_path / "only", feats, targets, labels, {})

        train, val, summary = make_mixed_dataloaders(
            [str(npz)], None, batch_size=8, seed=3
        )
        assert summary["num_synthetic"] == 0
        assert len(train.dataset) + len(val.dataset) == 30
