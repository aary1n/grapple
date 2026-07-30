"""Shared memory IPC bridge to the C# pipeline.

From vla-architecture.md §2:
    - GrappleIntent writes to FlatBufferSensorArena using existing FlatBuffer schema
    - The C# side does NOT know the data came from a neural network
    - Arena magic: 0x4C505247 ("GRPL")
    - Sequence number: monotonically increasing
    - Timestamp: QPC ticks from original frame capture

Wire-format ground truth is GrappleDetector.py + docs/PROTOCOL.md. This module
mirrors GrappleDetector.py exactly (same generated FlatBuffer code, same header
discipline) so that GrappleIntent is a drop-in replacement:

    Sensor arena header (32 bytes, '<QqiiQ'):
        [0:8]   uint64  MagicNumber (0x4C505247)
        [8:16]  int64   SequenceNumber        ← updated per frame
        [16:20] int32   ProtocolVersion (2)
        [20:24] int32   BufferSize            ← updated per frame
        [24:32] uint64  TimestampFrequency (QPC)
    Payload (SensorFrame FlatBuffer) starts at offset 64.
    Magic/version/frequency are written ONCE at init if absent — never per frame.

    Video frame arena header (40 bytes, '<Qiiqiiq'):
        magic (0x31454C5050415247 "GRAPPLE1"), slot_count, slot_size,
        write_head, published_id (offset 24), _pad, qpc_frequency
    Slot i: metadata '<qi' (timestamp, payload_size) at 1024 + i×slot_size,
    frame bytes at metadata offset + 64. Geometry (slot_count, slot_size) is
    read from the header — never hardcoded.
"""

from __future__ import annotations

import ctypes
import json
import logging
import mmap
import os
import re
import struct
import sys
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── grapple_config.json (shared with C#) ─────────────────────────────────────


def _strip_json_comments(text: str) -> str:
    """Strip // line comments (C# JsonSerializerOptions allows them)."""
    return re.sub(r"//.*?$", "", text, flags=re.MULTILINE)


def _load_grapple_config() -> dict:
    """Load grapple_config.json, searching upward from this file's directory."""
    search_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        candidate = os.path.join(search_dir, "grapple_config.json")
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.loads(_strip_json_comments(f.read()))
        # Also check the GrappleV2 source tree at each level
        sibling = os.path.join(search_dir, "GrappleV2", "grapple_config.json")
        if os.path.exists(sibling):
            with open(sibling) as f:
                return json.loads(_strip_json_comments(f.read()))
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break
        search_dir = parent
    logger.warning("No grapple_config.json found — using protocol defaults")
    return {}


_CFG = _load_grapple_config()

# ── Constants (MUST MATCH C# / GrappleDetector.py) ───────────────────────────

_sensor_cfg = _CFG.get("arenas", {}).get("sensor", {})
SENSOR_ARENA_NAME = _sensor_cfg.get("mapName", "Local\\GrappleSensorArena")
SENSOR_SIGNAL_NAME = _sensor_cfg.get("signalName", "Local\\GrappleSensorSignal")
SENSOR_ARENA_SIZE = _sensor_cfg.get("capacityBytes", 8192)
ARENA_MAGIC = 0x4C505247  # "GRPL"
PROTOCOL_VERSION = 2
SENSOR_HEADER_FORMAT = "<QqiiQ"  # magic, sequence, version, bufferSize, qpcFreq
SENSOR_HEADER_SIZE = struct.calcsize(SENSOR_HEADER_FORMAT)  # 32
DATA_OFFSET = 64

_frame_cfg = _CFG.get("arenas", {}).get("frame", {})
_frame_signal_cfg = _CFG.get("arenas", {}).get("frameSignal", {})
VIDEO_ARENA_NAME = _frame_cfg.get("mapName", "Local\\GrappleMap")
VIDEO_SIGNAL_NAME = _frame_signal_cfg.get("signalName", "Local\\GrappleSignal")
VIDEO_ARENA_CAPACITY = _frame_cfg.get("capacityMB", 256) * 1024 * 1024
VIDEO_ARENA_MAGIC = 0x31454C5050415247  # "GRAPPLE1"
VIDEO_HEADER_FORMAT = "<Qiiqiiq"  # magic, slots, slotSize, writeHead, published, pad, freq
VIDEO_HEADER_SIZE = struct.calcsize(VIDEO_HEADER_FORMAT)  # 40
VIDEO_PUBLISHED_ID_OFFSET = 24
VIDEO_FIRST_SLOT_OFFSET = 1024  # Protocol constant
VIDEO_METADATA_FORMAT = "<qi"  # timestamp, payload_size
VIDEO_METADATA_SIZE = 64  # Protocol constant

_webcam_cfg = _CFG.get("webcam", {})
FRAME_WIDTH = _webcam_cfg.get("width", 1920)
FRAME_HEIGHT = _webcam_cfg.get("height", 1080)
FRAME_CHANNELS = 3  # Protocol constant (RGB24)

# Gesture enum mapping (matches GestureType in .fbs schema)
GESTURE_NONE = 0
GESTURE_POINT = 1
GESTURE_PINCH = 2
GESTURE_GRAB = 3
GESTURE_SWIPE = 4

# ── Win32 event plumbing ─────────────────────────────────────────────────────

SYNCHRONIZE = 0x00100000
EVENT_MODIFY_STATE = 0x0002
WAIT_OBJECT_0 = 0x0
WAIT_TIMEOUT = 0x102

_IS_WINDOWS = sys.platform == "win32"


def _kernel32():
    return ctypes.windll.kernel32  # type: ignore[attr-defined]


def qpc_frequency() -> int:
    """QueryPerformanceFrequency (ticks/sec). 0 on non-Windows."""
    if not _IS_WINDOWS:
        return 0
    freq = ctypes.c_int64()
    _kernel32().QueryPerformanceFrequency(ctypes.byref(freq))
    return freq.value


def qpc_now() -> int:
    """QueryPerformanceCounter ticks. 0 on non-Windows."""
    if not _IS_WINDOWS:
        return 0
    counter = ctypes.c_int64()
    _kernel32().QueryPerformanceCounter(ctypes.byref(counter))
    return counter.value


# ── Generated FlatBuffer modules (same code GrappleDetector.py uses) ─────────

_GENERATED_DIR = str(
    Path(__file__).resolve().parents[2] / "GrappleV2" / "tools" / "generated"
)


def _import_generated():
    """Import the flatc-generated protocol modules, extending sys.path if needed."""
    if _GENERATED_DIR not in sys.path:
        sys.path.insert(0, _GENERATED_DIR)
    from Grapple.Protocol import HandState as FBHandState  # noqa: N813
    from Grapple.Protocol import SensorFrame as FBSensorFrame  # noqa: N813

    return FBHandState, FBSensorFrame


class SensorArenaWriter:
    """Writes SensorFrame FlatBuffers to the shared memory arena.

    Drop-in replacement for GrappleDetector.py's FlatBuffer dual-write: same
    generated code, same header discipline. The C# MouseControllerNode reads
    from this arena identically regardless of which producer wrote the data.
    """

    def __init__(self) -> None:
        self._shm: mmap.mmap | None = None
        self._signal_handle: int | None = None
        self._sequence: int = 0
        self._lock = threading.Lock()
        self._builder = None  # Pre-allocated, reused across frames
        self._fb_hand = None
        self._fb_frame = None

    def open(self) -> bool:
        """Open (or create) the sensor arena and signal event.

        Returns True on success. The header is initialized once if the magic
        is absent; per-frame writes only ever touch sequence and bufferSize.
        """
        try:
            import flatbuffers

            self._fb_hand, self._fb_frame = _import_generated()
            self._builder = flatbuffers.Builder(512)
        except ImportError:
            logger.exception(
                "flatbuffers or generated protocol modules unavailable "
                "(expected at %s)", _GENERATED_DIR,
            )
            return False

        try:
            self._shm = mmap.mmap(
                -1, SENSOR_ARENA_SIZE, tagname=SENSOR_ARENA_NAME,
                access=mmap.ACCESS_WRITE,
            )
        except Exception:
            logger.exception("Could not open sensor arena %s", SENSOR_ARENA_NAME)
            return False

        # Initialize header once (idempotent — C# or a previous producer may
        # have written it already).
        self._shm.seek(0)
        existing_magic = struct.unpack("<Q", self._shm.read(8))[0]
        if existing_magic != ARENA_MAGIC:
            self._shm.seek(0)
            self._shm.write(
                struct.pack(
                    SENSOR_HEADER_FORMAT,
                    ARENA_MAGIC,
                    0,                  # sequence (updated per frame)
                    PROTOCOL_VERSION,
                    0,                  # bufferSize (updated per frame)
                    qpc_frequency(),
                )
            )

        if _IS_WINDOWS:
            try:
                # Create-or-open, matching GrappleDetector.py (CreateEventW
                # opens the existing event if the C# side made it first).
                handle = _kernel32().CreateEventW(None, False, False, SENSOR_SIGNAL_NAME)
                self._signal_handle = handle or None
                if not handle:
                    logger.warning("Could not create/open sensor signal event")
            except Exception:
                logger.exception("Sensor signal event setup failed")
                self._signal_handle = None
        else:
            logger.info("Win32 events not available — signal disabled")

        logger.info("Sensor arena opened: %s", SENSOR_ARENA_NAME)
        return True

    def write_sensor_frame(
        self,
        x: float,
        y: float,
        z: float,
        velocity_x: float,
        velocity_y: float,
        gesture_id: int,
        confidence: float,
        timestamp: int,
    ) -> None:
        """Write a SensorFrame to the shared memory arena.

        Args:
            x, y, z: Normalized hand position [0.0, 1.0]
            velocity_x, velocity_y: Hand velocity in normalized units/sec
            gesture_id: GestureType enum value
            confidence: Detection confidence [0.0, 1.0]
            timestamp: QPC ticks from the original frame capture
        """
        if self._shm is None or self._builder is None:
            return

        FBHandState, FBSensorFrame = self._fb_hand, self._fb_frame
        builder = self._builder

        try:
            builder.Clear()

            FBHandState.HandStateStart(builder)
            FBHandState.HandStateAddX(builder, x)
            FBHandState.HandStateAddY(builder, y)
            FBHandState.HandStateAddZ(builder, z)
            FBHandState.HandStateAddVelocityX(builder, velocity_x)
            FBHandState.HandStateAddVelocityY(builder, velocity_y)
            FBHandState.HandStateAddGesture(builder, gesture_id)
            FBHandState.HandStateAddConfidence(builder, confidence)
            FBHandState.HandStateAddTimestamp(builder, timestamp)
            hand_offset = FBHandState.HandStateEnd(builder)

            FBSensorFrame.SensorFrameStart(builder)
            FBSensorFrame.SensorFrameAddSequence(builder, self._sequence + 1)
            FBSensorFrame.SensorFrameAddHand(builder, hand_offset)
            FBSensorFrame.SensorFrameAddProtocolVersion(builder, PROTOCOL_VERSION)
            frame_offset = FBSensorFrame.SensorFrameEnd(builder)

            builder.Finish(frame_offset)
            # builder.Output() returns the builder's internal bytearray view;
            # bytes() materializes it in the correct order for the arena write.
            fb_bytes = bytes(builder.Output())

            with self._lock:
                self._sequence += 1

                self._shm.seek(DATA_OFFSET)
                self._shm.write(fb_bytes)

                # Update ONLY sequence and bufferSize — magic/version/freq are
                # init-time fields and must not be clobbered per frame.
                self._shm.seek(8)
                self._shm.write(struct.pack("<q", self._sequence))
                self._shm.seek(20)
                self._shm.write(struct.pack("<i", len(fb_bytes)))

            if self._signal_handle:
                try:
                    _kernel32().SetEvent(self._signal_handle)
                except Exception:
                    pass

        except Exception:
            logger.exception("Failed to write SensorFrame to arena")

    def close(self) -> None:
        """Close shared memory and signal handles."""
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        if self._signal_handle:
            try:
                _kernel32().CloseHandle(self._signal_handle)
            except Exception:
                pass
            self._signal_handle = None

    def __enter__(self) -> SensorArenaWriter:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class VideoFrameReader:
    """Reads video frames from the SharedMemoryArena ring buffer.

    Slot geometry (count, size) comes from the arena header written by the C#
    producer; frame dimensions come from grapple_config.json. Nothing about
    the arena layout is assumed beyond the protocol constants in PROTOCOL.md.
    """

    def __init__(self, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> None:
        self._shm: mmap.mmap | None = None
        self._event_handle: int | None = None
        self._last_read_index: int = -1
        self._slot_count: int = 0
        self._slot_size: int = 0
        self._qpc_freq: int = 0
        self._width = width
        self._height = height
        self._frame_size = width * height * FRAME_CHANNELS

    @property
    def qpc_frequency(self) -> int:
        """QPC ticks/sec as reported by the C# producer's arena header."""
        return self._qpc_freq

    def open(self) -> bool:
        """Open the video frame arena and validate its header."""
        try:
            self._shm = mmap.mmap(
                -1, VIDEO_ARENA_CAPACITY, tagname=VIDEO_ARENA_NAME,
                access=mmap.ACCESS_READ,
            )
        except Exception:
            logger.warning(
                "Could not open video frame arena %s — is the C# pipeline running?",
                VIDEO_ARENA_NAME,
            )
            return False

        header = struct.unpack(VIDEO_HEADER_FORMAT, self._shm[:VIDEO_HEADER_SIZE])
        magic, slot_count, slot_size, _write_head, _published, _pad, freq = header

        if magic != VIDEO_ARENA_MAGIC:
            logger.error(
                "Video arena magic mismatch: got 0x%X, expected 0x%X",
                magic, VIDEO_ARENA_MAGIC,
            )
            self._shm.close()
            self._shm = None
            return False

        self._slot_count = slot_count
        self._slot_size = slot_size
        self._qpc_freq = freq

        if _IS_WINDOWS:
            try:
                self._event_handle = _kernel32().OpenEventW(
                    SYNCHRONIZE | EVENT_MODIFY_STATE, False, VIDEO_SIGNAL_NAME
                ) or None
                if self._event_handle is None:
                    logger.warning(
                        "Could not open frame signal event %s — falling back to polling",
                        VIDEO_SIGNAL_NAME,
                    )
            except Exception:
                self._event_handle = None

        logger.info(
            "Video frame arena opened: %s (%d slots × %d bytes, qpc=%d)",
            VIDEO_ARENA_NAME, slot_count, slot_size, freq,
        )
        return True

    def wait_for_frame(self, timeout_ms: int = 100) -> bool:
        """Block until the producer signals a new frame (or timeout).

        Returns True if signaled, False on timeout or if events are unavailable
        (in which case callers should poll read_latest_frame directly).
        """
        if self._event_handle is None:
            return False
        result = _kernel32().WaitForSingleObject(self._event_handle, timeout_ms)
        return result == WAIT_OBJECT_0

    def read_latest_frame(self) -> tuple[np.ndarray | None, int]:
        """Read the most recent video frame.

        Returns:
            (frame_rgb, timestamp) — frame as (H, W, 3) uint8 array (zero-copy
            view into shared memory; consume before the slot is recycled),
            or (None, 0) if no new frame is available.
        """
        if self._shm is None:
            return None, 0

        try:
            self._shm.seek(VIDEO_PUBLISHED_ID_OFFSET)
            published_id = struct.unpack("<i", self._shm.read(4))[0]

            if published_id < 0 or published_id == self._last_read_index:
                return None, 0
            if published_id >= self._slot_count:
                logger.error("Published slot %d out of range", published_id)
                return None, 0

            slot_offset = VIDEO_FIRST_SLOT_OFFSET + published_id * self._slot_size

            self._shm.seek(slot_offset)
            timestamp, payload_size = struct.unpack(
                VIDEO_METADATA_FORMAT, self._shm.read(12)
            )
            if payload_size < self._frame_size:
                return None, 0

            frame = np.frombuffer(
                self._shm,
                dtype=np.uint8,
                count=self._frame_size,
                offset=slot_offset + VIDEO_METADATA_SIZE,
            ).reshape((self._height, self._width, FRAME_CHANNELS))

            self._last_read_index = published_id
            return frame, timestamp

        except Exception:
            logger.exception("Failed to read video frame")
            return None, 0

    def close(self) -> None:
        if self._event_handle:
            try:
                _kernel32().CloseHandle(self._event_handle)
            except Exception:
                pass
            self._event_handle = None
        if self._shm is not None:
            try:
                self._shm.close()
            except BufferError:
                # Zero-copy frame views from read_latest_frame() are still
                # referenced (e.g. the loop's last frame at shutdown); the
                # mapping is released when those arrays are collected.
                logger.warning(
                    "Video arena close deferred — frame views still referenced"
                )
            self._shm = None
