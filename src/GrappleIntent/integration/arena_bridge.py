"""Shared memory IPC bridge to the C# pipeline.

From vla-architecture.md §2:
    - GrappleIntent writes to FlatBufferSensorArena using existing FlatBuffer schema
    - The C# side does NOT know the data came from a neural network
    - Arena magic: 0x4C505247 ("GRPL")
    - Sequence number: monotonically increasing
    - Timestamp: QPC ticks from original frame capture

From PROTOCOL.md:
    FlatBuffer Arena Header (32 bytes):
        [0:8]   uint64  MagicNumber (0x4C505247)
        [8:16]  int64   SequenceNumber
        [16:20] int32   ProtocolVersion (2)
        [20:24] int32   BufferSize
        [24:32] int64   TimestampFrequency
    Data starts at offset 64.

This module mirrors the existing GrappleDetector.py FlatBuffer writing pattern
so that GrappleIntent is a drop-in replacement.
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import struct
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants from PROTOCOL.md ────────────────────────────────────────────────

SENSOR_ARENA_NAME = "Local\\GrappleSensorArena"
SENSOR_SIGNAL_NAME = "Local\\GrappleSensorSignal"
SENSOR_ARENA_SIZE = 8192  # 8KB
ARENA_MAGIC = 0x4C505247  # "GRPL"
PROTOCOL_VERSION = 2
HEADER_SIZE = 32
DATA_OFFSET = 64

# Video frame arena constants
VIDEO_ARENA_NAME = "Local\\GrappleMap"
VIDEO_ARENA_MAGIC = 0x31454C5050415247  # "GRAPPLE1"
VIDEO_HEADER_SIZE = 40
VIDEO_SLOT_SIZE = 8 * 1024 * 1024  # 8MB
VIDEO_NUM_SLOTS = 30
VIDEO_METADATA_OFFSET = 1024

# Gesture enum mapping (matches GestureType in .fbs schema)
GESTURE_NONE = 0
GESTURE_POINT = 1
GESTURE_PINCH = 2
GESTURE_GRAB = 3
GESTURE_SWIPE = 4


class SensorArenaWriter:
    """Writes SensorFrame FlatBuffers to the shared memory arena.

    This replaces GrappleDetector.py's FlatBuffer writing when GrappleIntent
    is the active sensor producer. The C# MouseControllerNode reads from
    this arena identically regardless of which producer wrote the data.
    """

    def __init__(self) -> None:
        self._shm: mmap.mmap | None = None
        self._signal_handle: int | None = None
        self._sequence: int = 0
        self._lock = threading.Lock()

    def open(self) -> bool:
        """Open the shared memory arena and signal event.

        Returns True if successful, False if the arena doesn't exist
        (C# process not running).
        """
        try:
            # Open existing memory-mapped file created by C#
            self._shm = mmap.mmap(-1, SENSOR_ARENA_SIZE, tagname=SENSOR_ARENA_NAME)

            # Open the signal event for notifying the C# consumer
            try:
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                self._signal_handle = kernel32.OpenEventW(0x2, False, SENSOR_SIGNAL_NAME)
                if self._signal_handle == 0:
                    logger.warning("Could not open signal event — C# may not be notified")
                    self._signal_handle = None
            except AttributeError:
                # Not on Windows (e.g., testing on Linux)
                logger.info("Win32 events not available — signal disabled")
                self._signal_handle = None

            logger.info("Sensor arena opened: %s", SENSOR_ARENA_NAME)
            return True

        except Exception:
            logger.warning("Could not open sensor arena — is the C# pipeline running?")
            return False

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
        handedness: str = "Right",
    ) -> None:
        """Write a SensorFrame to the shared memory arena.

        This builds the FlatBuffer in-place and updates the arena header.
        For now, we use struct.pack for the arena header and flatbuffers
        library for the payload (matching GrappleDetector.py pattern).

        Args:
            x, y, z: Normalized hand position [0.0, 1.0]
            velocity_x, velocity_y: Hand velocity in normalized units/sec
            gesture_id: GestureType enum value
            confidence: Detection confidence [0.0, 1.0]
            timestamp: QPC ticks from the original frame capture
            handedness: "Left" or "Right"
        """
        if self._shm is None:
            return

        try:
            import flatbuffers

            builder = flatbuffers.Builder(512)

            # Build HandState table
            # (Import paths depend on generated code location)
            # For now, build manually using the flatbuffers API
            handedness_offset = builder.CreateString(handedness)

            # HandState table (field order matters for FlatBuffer wire format)
            builder.StartObject(10)  # HandState has 10 fields
            builder.PrependFloat64Slot(0, x, 0.0)  # x
            builder.PrependFloat64Slot(1, y, 0.0)  # y
            builder.PrependFloat64Slot(2, z, 0.0)  # z
            builder.PrependFloat64Slot(3, velocity_x, 0.0)  # velocity_x
            builder.PrependFloat64Slot(4, velocity_y, 0.0)  # velocity_y
            builder.PrependInt32Slot(5, gesture_id, 0)  # gesture
            builder.PrependFloat32Slot(6, confidence, 0.0)  # confidence
            builder.PrependInt64Slot(7, timestamp, 0)  # timestamp
            builder.PrependUOffsetTRelativeSlot(8, handedness_offset, 0)  # handedness
            hand_offset = builder.EndObject()

            # Build SensorFrame root
            self._sequence += 1
            builder.StartObject(4)  # SensorFrame has 4 fields (sequence, hand, eye, telemetry, version)
            builder.PrependInt64Slot(0, self._sequence, 0)  # sequence
            builder.PrependUOffsetTRelativeSlot(1, hand_offset, 0)  # hand
            # eye and telemetry are null (not present)
            builder.PrependInt32Slot(4, PROTOCOL_VERSION, 2)  # protocol_version
            frame_offset = builder.EndObject()

            builder.Finish(frame_offset)
            buf = builder.Output()
            fb_bytes = bytes(buf)

            with self._lock:
                # Write arena header
                self._shm.seek(0)
                self._shm.write(
                    struct.pack(
                        "<QqiIq",
                        ARENA_MAGIC,  # MagicNumber
                        self._sequence,  # SequenceNumber
                        PROTOCOL_VERSION,  # ProtocolVersion
                        len(fb_bytes),  # BufferSize
                        0,  # TimestampFrequency (filled by C# typically)
                    )
                )

                # Write FlatBuffer payload at DATA_OFFSET
                self._shm.seek(DATA_OFFSET)
                self._shm.write(fb_bytes)

            # Signal the C# consumer
            if self._signal_handle:
                try:
                    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                    kernel32.SetEvent(self._signal_handle)
                except Exception:
                    pass

        except ImportError:
            logger.error("flatbuffers package not installed — cannot write SensorFrame")
        except Exception:
            logger.exception("Failed to write SensorFrame to arena")

    def close(self) -> None:
        """Close shared memory and signal handles."""
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        if self._signal_handle:
            try:
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                kernel32.CloseHandle(self._signal_handle)
            except Exception:
                pass
            self._signal_handle = None

    def __enter__(self) -> SensorArenaWriter:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class VideoFrameReader:
    """Reads video frames from the SharedMemoryArena (256MB ring buffer).

    Used by the semantic path to get screen/camera images for the
    dual-scale foveated input pipeline.

    From PROTOCOL.md:
        ArenaHeader at offset 0 (40 bytes)
        Frame metadata at offset 1024 + slotId × 8MB (64 bytes)
        Frame data follows metadata in each slot
    """

    def __init__(self) -> None:
        self._shm: mmap.mmap | None = None
        self._last_read_index: int = -1

    def open(self) -> bool:
        """Open the video frame arena."""
        try:
            total_size = VIDEO_HEADER_SIZE + VIDEO_NUM_SLOTS * VIDEO_SLOT_SIZE
            self._shm = mmap.mmap(-1, total_size, tagname=VIDEO_ARENA_NAME)
            logger.info("Video frame arena opened: %s", VIDEO_ARENA_NAME)
            return True
        except Exception:
            logger.warning("Could not open video frame arena")
            return False

    def read_latest_frame(self) -> tuple[np.ndarray | None, int]:
        """Read the most recent video frame.

        Returns:
            (frame_rgb, timestamp) — frame as (H, W, 3) uint8 array,
            or (None, 0) if no new frame available.
        """
        if self._shm is None:
            return None, 0

        try:
            # Read PublishedBufferId from header
            self._shm.seek(24)
            published_id = struct.unpack("<i", self._shm.read(4))[0]

            if published_id < 0 or published_id == self._last_read_index:
                return None, 0

            # Read frame metadata
            slot_offset = VIDEO_METADATA_OFFSET + published_id * VIDEO_SLOT_SIZE
            self._shm.seek(slot_offset)
            timestamp, payload_size = struct.unpack("<qi", self._shm.read(12))

            if payload_size <= 0:
                return None, 0

            # Read frame data (follows 64-byte metadata)
            data_offset = slot_offset + 64
            self._shm.seek(data_offset)
            raw = self._shm.read(payload_size)

            # Assuming 1920×1080 RGB
            expected_size = 1920 * 1080 * 3
            if len(raw) >= expected_size:
                frame = np.frombuffer(raw[:expected_size], dtype=np.uint8).reshape(
                    1080, 1920, 3
                )
                self._last_read_index = published_id
                return frame, timestamp

            return None, 0

        except Exception:
            logger.exception("Failed to read video frame")
            return None, 0

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()
            self._shm = None
