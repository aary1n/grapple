"""End-to-end integration tests against the C# arena protocol (headless).

These tests play the C# side of both shared-memory boundaries, byte-for-byte
per Grapple.Core/SharedMemoryArena.cs and FlatBufferSensorArena.cs, so the
full sidecar loop is exercised without a webcam or a running C# process:

    FakeVideoArenaProducer (this file — C# SharedMemoryArena role)
        → VideoFrameReader (GrappleIntent)
        → stub LandmarkExtractor (deterministic, derived from frame bytes)
        → GrappleIntentRuntime
        → SensorArenaWriter (GrappleIntent)
        → read_sensor_frame_like_csharp (this file — FlatBufferSensorArena role)

All arena/event names are test-specific so a live pipeline is never touched.
The FlatBuffer wire format itself is additionally locked cross-language by
Grapple.Tests/ProtocolCompatibilityTests.cs (same flatc schema).
"""

from __future__ import annotations

import mmap
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows named shared memory required"
)

from ..integration import arena_bridge  # noqa: E402
from ..integration.arena_bridge import (  # noqa: E402
    ARENA_MAGIC,
    PROTOCOL_VERSION,
    SENSOR_HEADER_FORMAT,
    VIDEO_ARENA_MAGIC,
    VIDEO_FIRST_SLOT_OFFSET,
    VIDEO_HEADER_FORMAT,
    VIDEO_METADATA_SIZE,
    SensorArenaWriter,
    VideoFrameReader,
    qpc_frequency,
)

# Test arena geometry — small frames so the full loop runs fast. Geometry is
# read from the header by VideoFrameReader, exactly as with the C# producer.
TEST_WIDTH = 8
TEST_HEIGHT = 8
TEST_FRAME_SIZE = TEST_WIDTH * TEST_HEIGHT * 3
TEST_SLOT_COUNT = 4
TEST_SLOT_SIZE = 4096
TEST_VIDEO_CAPACITY = VIDEO_FIRST_SLOT_OFFSET + TEST_SLOT_COUNT * TEST_SLOT_SIZE
TEST_SENSOR_CAPACITY = 8192

VIDEO_NAME = "Local\\GrappleIntentTestMap"
VIDEO_SIGNAL = "Local\\GrappleIntentTestSignal"
SENSOR_NAME = "Local\\GrappleIntentTestSensorArena"
SENSOR_SIGNAL = "Local\\GrappleIntentTestSensorSignal"

DATA_OFFSET = 64  # FlatBufferSensorArena.DataOffset


class FakeVideoArenaProducer:
    """Plays C# SharedMemoryArena + producer: header init, slot writes, publish.

    Mirrors SharedMemoryArena.InitializeIfNeeded / AcquireNextSlot / GetSpan /
    UpdatePublishedBuffer and the AutoReset frame signal event.
    """

    def __init__(self) -> None:
        self._shm = mmap.mmap(
            -1, TEST_VIDEO_CAPACITY, tagname=VIDEO_NAME, access=mmap.ACCESS_WRITE
        )
        self._write_head = 0
        # Header per ArenaHeader (C#): magic, slotCount, slotSize, writeHead,
        # publishedId=-1, protocolVersion=1, qpcFreq
        self._shm.seek(0)
        self._shm.write(
            struct.pack(
                VIDEO_HEADER_FORMAT,
                VIDEO_ARENA_MAGIC,
                TEST_SLOT_COUNT,
                TEST_SLOT_SIZE,
                0,
                -1,
                1,
                qpc_frequency(),
            )
        )
        # AutoReset event, like C#'s EventWaitHandle(false, AutoReset, name)
        self._event = arena_bridge._kernel32().CreateEventW(
            None, False, False, VIDEO_SIGNAL
        )

    def publish(self, frame: np.ndarray, timestamp: int) -> None:
        payload = frame.tobytes()
        buffer_id = self._write_head % TEST_SLOT_COUNT
        self._write_head += 1

        slot_offset = VIDEO_FIRST_SLOT_OFFSET + buffer_id * TEST_SLOT_SIZE
        self._shm.seek(slot_offset)
        self._shm.write(struct.pack("<qi", timestamp, len(payload)))
        self._shm.seek(slot_offset + VIDEO_METADATA_SIZE)
        self._shm.write(payload)

        # Volatile publish, then signal (C#: UpdatePublishedBuffer + Set)
        self._shm.seek(24)
        self._shm.write(struct.pack("<i", buffer_id))
        if self._event:
            arena_bridge._kernel32().SetEvent(self._event)

    def close(self) -> None:
        if self._event:
            arena_bridge._kernel32().CloseHandle(self._event)
            self._event = None
        self._shm.close()


def make_test_frame(index: int) -> np.ndarray:
    """Frame whose bytes encode its index — lets the stub extractor prove the
    video arena payload actually flowed through, not just the signal."""
    return np.full(
        (TEST_HEIGHT, TEST_WIDTH, 3), index % 251, dtype=np.uint8
    )


def read_sensor_frame_like_csharp(shm: mmap.mmap):
    """Replicates FlatBufferSensorArena.ReadLatestSensorFrame exactly:
    read BufferSize at offset 20, copy [64, 64+size), parse as SensorFrame.
    Returns (header_tuple, SensorFrame) or (header_tuple, None)."""
    header = struct.unpack(
        SENSOR_HEADER_FORMAT, shm[: struct.calcsize(SENSOR_HEADER_FORMAT)]
    )
    buffer_size = header[3]
    if buffer_size <= 0 or buffer_size > TEST_SENSOR_CAPACITY - DATA_OFFSET:
        return header, None

    buf = bytes(shm[DATA_OFFSET : DATA_OFFSET + buffer_size])
    FBHandState, FBSensorFrame = arena_bridge._import_generated()
    frame = FBSensorFrame.SensorFrame.GetRootAsSensorFrame(bytearray(buf), 0)
    return header, frame


@pytest.fixture
def patched_arenas(monkeypatch):
    """Point arena_bridge at test-specific names/sizes; restore afterwards."""
    monkeypatch.setattr(arena_bridge, "VIDEO_ARENA_NAME", VIDEO_NAME)
    monkeypatch.setattr(arena_bridge, "VIDEO_SIGNAL_NAME", VIDEO_SIGNAL)
    monkeypatch.setattr(arena_bridge, "VIDEO_ARENA_CAPACITY", TEST_VIDEO_CAPACITY)
    monkeypatch.setattr(arena_bridge, "SENSOR_ARENA_NAME", SENSOR_NAME)
    monkeypatch.setattr(arena_bridge, "SENSOR_SIGNAL_NAME", SENSOR_SIGNAL)
    monkeypatch.setattr(arena_bridge, "SENSOR_ARENA_SIZE", TEST_SENSOR_CAPACITY)
    yield


class TestVideoArenaBoundary:
    """VideoFrameReader against a byte-exact fake of the C# producer."""

    def test_reader_validates_header_and_geometry(self, patched_arenas):
        producer = FakeVideoArenaProducer()
        try:
            reader = VideoFrameReader(width=TEST_WIDTH, height=TEST_HEIGHT)
            assert reader.open()
            assert reader._slot_count == TEST_SLOT_COUNT
            assert reader._slot_size == TEST_SLOT_SIZE
            assert reader.qpc_frequency == qpc_frequency()
            reader.close()
        finally:
            producer.close()

    def test_frame_roundtrip_and_latest_wins(self, patched_arenas):
        producer = FakeVideoArenaProducer()
        reader = VideoFrameReader(width=TEST_WIDTH, height=TEST_HEIGHT)
        try:
            assert reader.open()

            # Nothing published yet
            frame, ts = reader.read_latest_frame()
            assert frame is None

            producer.publish(make_test_frame(1), timestamp=1000)
            frame, ts = reader.read_latest_frame()
            assert frame is not None
            assert ts == 1000
            np.testing.assert_array_equal(frame, make_test_frame(1))

            # Same published id → no new frame
            frame, _ = reader.read_latest_frame()
            assert frame is None

            # LIFO: two publishes without a read → reader sees only the latest
            producer.publish(make_test_frame(2), timestamp=2000)
            producer.publish(make_test_frame(3), timestamp=3000)
            frame, ts = reader.read_latest_frame()
            assert ts == 3000
            np.testing.assert_array_equal(frame, make_test_frame(3))
        finally:
            reader.close()
            producer.close()

    def test_event_signaling(self, patched_arenas):
        producer = FakeVideoArenaProducer()
        reader = VideoFrameReader(width=TEST_WIDTH, height=TEST_HEIGHT)
        try:
            assert reader.open()
            assert reader._event_handle is not None
            assert not reader.wait_for_frame(timeout_ms=10)  # nothing yet
            producer.publish(make_test_frame(1), timestamp=1)
            assert reader.wait_for_frame(timeout_ms=500)
        finally:
            reader.close()
            producer.close()


class TestSensorArenaBoundary:
    """SensorArenaWriter output verified with C# FlatBufferSensorArena semantics."""

    def test_header_discipline_preserves_csharp_init(self, patched_arenas):
        # C# FlatBufferSensorArena initializes the header first; the Python
        # writer must never clobber magic/version/frequency per frame.
        csharp_freq = 123456789
        shm = mmap.mmap(
            -1, TEST_SENSOR_CAPACITY, tagname=SENSOR_NAME, access=mmap.ACCESS_WRITE
        )
        shm.seek(0)
        shm.write(
            struct.pack(
                SENSOR_HEADER_FORMAT, ARENA_MAGIC, 0, PROTOCOL_VERSION, 0, csharp_freq
            )
        )
        writer = SensorArenaWriter()
        try:
            assert writer.open()
            for i in range(5):
                writer.write_sensor_frame(
                    x=0.1 * i, y=0.2, z=0.0, velocity_x=0.0, velocity_y=0.0,
                    gesture_id=0, confidence=1.0, timestamp=i,
                )

            header, frame = read_sensor_frame_like_csharp(shm)
            magic, sequence, version, buffer_size, freq = header
            assert magic == ARENA_MAGIC
            assert version == PROTOCOL_VERSION
            assert freq == csharp_freq  # untouched — C#'s value survives
            assert sequence == 5
            assert buffer_size > 0
            assert frame is not None
        finally:
            writer.close()
            shm.close()

    def test_sensor_frame_fields_parse_like_csharp(self, patched_arenas):
        shm = mmap.mmap(
            -1, TEST_SENSOR_CAPACITY, tagname=SENSOR_NAME, access=mmap.ACCESS_WRITE
        )
        writer = SensorArenaWriter()
        try:
            assert writer.open()
            writer.write_sensor_frame(
                x=0.25, y=0.75, z=0.05,
                velocity_x=1.5, velocity_y=-2.5,
                gesture_id=arena_bridge.GESTURE_PINCH,
                confidence=0.875, timestamp=987654321,
            )

            header, frame = read_sensor_frame_like_csharp(shm)
            assert frame is not None
            assert frame.Sequence() == 1
            assert frame.ProtocolVersion() == PROTOCOL_VERSION
            hand = frame.Hand()
            assert hand is not None
            assert hand.X() == pytest.approx(0.25)
            assert hand.Y() == pytest.approx(0.75)
            assert hand.Z() == pytest.approx(0.05)
            assert hand.VelocityX() == pytest.approx(1.5)
            assert hand.VelocityY() == pytest.approx(-2.5)
            assert hand.Gesture() == arena_bridge.GESTURE_PINCH
            assert hand.Confidence() == pytest.approx(0.875)
            assert hand.Timestamp() == 987654321
        finally:
            writer.close()
            shm.close()

    def test_sequence_monotonic(self, patched_arenas):
        shm = mmap.mmap(
            -1, TEST_SENSOR_CAPACITY, tagname=SENSOR_NAME, access=mmap.ACCESS_WRITE
        )
        writer = SensorArenaWriter()
        try:
            assert writer.open()
            last_seq = 0
            for i in range(20):
                writer.write_sensor_frame(
                    x=0.5, y=0.5, z=0.0, velocity_x=0.0, velocity_y=0.0,
                    gesture_id=0, confidence=1.0, timestamp=i,
                )
                header, frame = read_sensor_frame_like_csharp(shm)
                assert frame.Sequence() > last_seq
                last_seq = frame.Sequence()
            assert last_seq == 20
        finally:
            writer.close()
            shm.close()


class _StubLandmarkExtractor:
    """Deterministic LandmarkExtractor stand-in — derives the wrist position
    from the frame's encoded index, proving frame bytes traversed the arena."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def extract(self, frame_rgb: np.ndarray, time_s: float):
        from ..data.landmarks import FEATURE_DIMS, LANDMARK_DIMS, LandmarkFrame

        k = int(frame_rgb[0, 0, 0])
        x = 0.1 + 0.001 * k
        y = 0.2 + 0.002 * k
        features = np.zeros(FEATURE_DIMS, dtype=np.float32)
        features[0], features[1] = x, y
        return LandmarkFrame(
            features=features,
            wrist_x=x, wrist_y=y, wrist_z=0.0,
            velocity_x=0.0, velocity_y=0.0, velocity_z=0.0,
            handedness="Right", detection_confidence=1.0,
        )

    def close(self) -> None:
        pass


class TestFullRuntimeLoop:
    """Fake C# producer → GrappleIntentRuntime → sensor arena, read like C#."""

    def _run_loop(self, monkeypatch, onnx_path: str):
        from .. import runtime as runtime_mod
        from ..configs import GrappleIntentConfig

        monkeypatch.setattr(
            "GrappleIntent.data.landmarks.LandmarkExtractor", _StubLandmarkExtractor
        )

        producer = FakeVideoArenaProducer()
        # ACCESS_WRITE: the first creation of a named mapping fixes its page
        # protection on Windows — a read-only view here would block the
        # runtime's writable open. The test itself only reads.
        sensor_view = mmap.mmap(
            -1, TEST_SENSOR_CAPACITY, tagname=SENSOR_NAME, access=mmap.ACCESS_WRITE
        )

        rt = runtime_mod.GrappleIntentRuntime(
            GrappleIntentConfig(), onnx_path=onnx_path, wait_seconds=5.0
        )
        # The runtime constructs readers with production frame dims; swap in
        # one sized for the test arena before the loop opens it.
        rt._reader = VideoFrameReader(width=TEST_WIDTH, height=TEST_HEIGHT)

        loop = threading.Thread(target=rt.run, daemon=True)
        loop.start()

        samples: list[tuple[int, float, float, int, float, int]] = []
        try:
            deadline = time.monotonic() + 15.0
            k = 0
            last_seq = 0
            while len(samples) < 20 and time.monotonic() < deadline:
                producer.publish(make_test_frame(k), timestamp=1000 + k * 10)
                k += 1
                time.sleep(0.02)  # ~50fps

                header, frame = read_sensor_frame_like_csharp(sensor_view)
                if frame is None or frame.Sequence() == last_seq:
                    continue
                last_seq = frame.Sequence()
                hand = frame.Hand()
                samples.append((
                    frame.Sequence(), hand.X(), hand.Y(),
                    hand.Gesture(), hand.Confidence(), hand.Timestamp(),
                ))
        finally:
            rt.stop()
            loop.join(timeout=5.0)
            producer.close()
            sensor_view.close()

        assert not loop.is_alive(), "runtime loop failed to stop"
        return samples

    def test_e2e_passthrough(self, patched_arenas, monkeypatch, tmp_path):
        """No ONNX model → landmark passthrough. Locks the protocol path."""
        samples = self._run_loop(
            monkeypatch, onnx_path=str(tmp_path / "missing.onnx")
        )
        assert len(samples) >= 20, f"only {len(samples)} sensor frames observed"

        sequences = [s[0] for s in samples]
        assert sequences == sorted(sequences)
        assert len(set(sequences)) == len(sequences)  # strictly monotonic

        for seq, x, y, gesture, confidence, ts in samples:
            # Invert x → frame index, then y and timestamp must match it
            k = round((x - 0.1) / 0.001)
            assert 0 <= k < 251
            assert y == pytest.approx(0.2 + 0.002 * k, abs=1e-6)
            assert ts == 1000 + k * 10
            assert gesture == 0  # passthrough cannot classify
            assert 0.0 <= confidence <= 1.0

        # Frame indices seen by the model side must be non-decreasing (LIFO)
        ks = [round((s[1] - 0.1) / 0.001) for s in samples]
        assert ks == sorted(ks)

    def test_e2e_with_onnx_model(self, patched_arenas, monkeypatch):
        """With the trained reflexive model loaded, the loop must stay live
        and produce valid gesture ids at model-driven confidence."""
        onnx = Path("checkpoints/reflexive/mobilenetv3_cursor_v0.1_fp32.onnx")
        if not onnx.exists():
            pytest.skip("trained ONNX model not present — run export_onnx first")

        samples = self._run_loop(monkeypatch, onnx_path=str(onnx))
        assert len(samples) >= 20

        sequences = [s[0] for s in samples]
        assert sequences == sorted(sequences)
        for seq, x, y, gesture, confidence, ts in samples:
            assert 0 <= gesture <= 4
            assert 0.0 <= confidence <= 1.0
