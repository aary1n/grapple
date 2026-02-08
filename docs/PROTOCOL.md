# Grapple Protocol V2 - Wire Format Specification

## Overview

Grapple Protocol V2 uses [FlatBuffers](https://flatbuffers.dev/) for zero-copy, schema-versioned IPC between sensor producers (Python/C++) and consumers (C#/.NET). This replaces the legacy `struct.pack` format with backward/forward compatible serialization.

**Protocol Version:** 2
**File Identifier:** `GRPL` (4 bytes)
**Schema:** `src/GrappleV2/schema/grapple_protocol.fbs`

---

## Shared Memory Arenas

Each sensor type uses a dedicated memory-mapped file (per ADR-003: separate arenas per sensor type).

| Arena | Map Name | Size | Signal Event | Update Rate | Status |
|-------|----------|------|-------------|-------------|--------|
| Sensor (Hand) | `Local\GrappleSensorArena` | 8KB | `Local\GrappleSensorSignal` | ~15Hz | Active |
| Eye Tracking | `Local\GrappleEyeResults` | 4KB | `Local\GrappleEyeSignal` | ~90Hz | Placeholder |
| Telemetry | `Local\GrappleTelemetry` | 4KB | `Local\GrappleTelemetrySignal` | ~10Hz | Placeholder |
| Video Frames | `Local\GrappleMap` | 256MB | `Local\GrappleSignal` | ~60Hz | Active (raw frames) |
| Legacy Hand | `Local\GrappleHandResults` | 4KB | `Local\GrappleHandSignal` | ~15Hz | Deprecated (dual-write) |

---

## Memory Layout

### Arena Header (32 bytes, offset 0)

All FlatBuffer arenas share this header format:

```
Offset  Size  Type     Field
──────  ────  ──────   ─────────────────────
0       8     uint64   MagicNumber (0x4C505247 = "GRPL")
8       8     int64    SequenceNumber (monotonic counter)
16      4     int32    ProtocolVersion (currently 2)
20      4     int32    BufferSize (bytes of FlatBuffer data)
24      8     int64    TimestampFrequency (QPC ticks/sec)
──────────────────────────────────────────────
Total: 32 bytes (8-byte aligned)
```

### Data Region (offset 64)

FlatBuffer-serialized payload starts at offset 64 (aligned). The `BufferSize` header field indicates the number of valid bytes.

```
[0..31]   Arena Header (32 bytes)
[32..63]  Reserved padding (alignment)
[64..]    FlatBuffer data (BufferSize bytes)
```

---

## FlatBuffers Schema

### SensorFrame (root type)

```
table SensorFrame {
  sequence: long;                    // Monotonic frame counter
  hand: HandState;                   // Nullable: only present if hand detected
  eye: EyeState;                     // Nullable: only present if eye tracker active
  telemetry: TelemetrySnapshot;      // Nullable: only present if metrics available
  protocol_version: int = 2;         // Schema version
}
```

### HandState

```
table HandState {
  x: double;              // Normalized X (0.0-1.0)
  y: double;              // Normalized Y (0.0-1.0)
  z: double;              // Depth (0=near, 1=far)
  velocity_x: double;     // Units/sec in normalized space
  velocity_y: double;     // Units/sec in normalized space
  gesture: GestureType;   // Enum: None=0, Point=1, Pinch=2
  confidence: float;      // MediaPipe confidence (0.0-1.0)
  timestamp: long;        // QPC ticks of source frame
  handedness: string;     // "Left" or "Right" (optional)
  landmarks: [double];    // 21x3 flattened array (optional)
}
```

### EyeState (placeholder)

```
table EyeState {
  gaze_x: double;              // Normalized screen X
  gaze_y: double;              // Normalized screen Y
  pupil_diameter_left: float;  // mm
  pupil_diameter_right: float; // mm
  confidence: float;           // Tracker confidence
  timestamp: long;             // QPC ticks
  fixation_duration_ms: int;   // Fixation duration
  saccade_velocity: float;     // Saccade speed
}
```

### TelemetrySnapshot (placeholder)

```
table TelemetrySnapshot {
  fps: float;
  latency_ms: float;
  dropped_frames: int;
  gc_gen0_collections: int;
  gc_gen1_collections: int;
  gc_gen2_collections: int;
  consecutive_drops: int;
  quality_degradation_active: bool;
  timestamp: long;
}
```

### GestureType Enum

```
enum GestureType : int {
  None = 0,
  Point = 1,
  Pinch = 2,
  Grab = 3,
  Swipe = 4
}
```

---

## Protocol Versioning

- The `protocol_version` field in `SensorFrame` tracks the schema version.
- Current version: **2** (initial FlatBuffers release).
- Version is also stored in the arena header for pre-validation before deserialization.

### Version Mismatch Handling

When a consumer detects a version mismatch:
1. Log a warning with expected vs actual version.
2. Continue operation if possible (FlatBuffers handles missing/new fields gracefully).
3. Recommend restarting processes to synchronize versions.

---

## Schema Evolution Rules

FlatBuffers supports forward/backward compatible schema changes:

### Safe Changes (no version bump required)
- Adding new fields to the **end** of a table (they get default values in old consumers)
- Adding new enum values (old consumers see the raw integer)

### Breaking Changes (requires version bump)
- Removing fields (use `deprecated` keyword instead)
- Changing field types
- Reordering fields
- Changing field IDs

### Policy
- Increment `protocol_version` default in schema for breaking changes.
- Old producers with new consumers: new fields read as defaults (safe).
- New producers with old consumers: new fields are ignored (safe).

---

## Migration: Legacy Format

### Legacy HandState (56 bytes)

The original format used `struct.pack`/`StructLayout` with explicit byte offsets:

```
Offset  Size  Type     Field
──────  ────  ──────   ─────
0       8     double   X
8       8     double   Y
16      8     double   Z
24      8     double   VX
32      8     double   VY
40      4     int32    GestureId
44      4     float    Confidence
48      8     int64    Timestamp
──────────────────────────────
Total: 56 bytes
Python format: '<dddddifq'
```

### Dual-Write Strategy

During migration, the Python detector writes to **both** arenas:
1. Legacy arena (`Local\GrappleHandResults`) - `struct.pack` format
2. Sensor arena (`Local\GrappleSensorArena`) - FlatBuffer `SensorFrame`

The C# `MouseControllerNode` reads from the FlatBuffer arena. The legacy arena remains for backward compatibility with older consumers.

---

## Code Generation

### Prerequisites
- FlatBuffers compiler (`flatc`) - version 25.x+
- Google.FlatBuffers NuGet package (25.2.10)
- Python `flatbuffers` package

### Regenerate Bindings

```bash
# From project root
cd src/GrappleV2/schema

# C# (single file)
flatc --csharp --gen-onefile grapple_protocol.fbs -o ../Grapple.Core/Generated/

# Python
flatc --python grapple_protocol.fbs -o ../tools/generated/
```

---

## Producer Implementation (Python)

```python
import flatbuffers
from Grapple.Protocol import HandState, SensorFrame

builder = flatbuffers.Builder(512)

# Build HandState
HandState.HandStateStart(builder)
HandState.HandStateAddX(builder, x)
HandState.HandStateAddY(builder, y)
HandState.HandStateAddGesture(builder, gesture_id)
HandState.HandStateAddConfidence(builder, confidence)
HandState.HandStateAddTimestamp(builder, timestamp)
hand_offset = HandState.HandStateEnd(builder)

# Build SensorFrame
SensorFrame.SensorFrameStart(builder)
SensorFrame.SensorFrameAddSequence(builder, sequence)
SensorFrame.SensorFrameAddHand(builder, hand_offset)
SensorFrame.SensorFrameAddProtocolVersion(builder, 2)
frame_offset = SensorFrame.SensorFrameEnd(builder)

builder.Finish(frame_offset)
buf = builder.Output()

# Write to shared memory at DATA_OFFSET
sensor_shm.seek(64)
sensor_shm.write(bytes(buf))
```

## Consumer Implementation (C#)

```csharp
using Grapple.Core;
using Grapple.Protocol;

using var arena = new FlatBufferSensorArena();

// Wait for data
arena.WaitForResult(100);

// Read FlatBuffer (pre-allocated buffer, near-zero alloc)
SensorFrame? frame = arena.ReadLatestSensorFrame();
if (frame?.Hand != null)
{
    var hand = frame.Value.Hand.Value;
    double x = hand.X;
    double y = hand.Y;
    GestureType gesture = hand.Gesture;
}
```

---

**Last Updated:** 2026-02-08
**Schema Version:** 2
