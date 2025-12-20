# Grapple - OS-Level Gesture Controller

## Project Manifest

**Version:** GrappleV2 (Clean-Room Implementation)
**Status:** M3 Complete - Full Pipeline Operational
**Target:** Enterprise CAD Integration (SolidWorks First)

---

## Architecture: "GrappleGraph"

Grapple is a **zero-GC, ultra-low-latency gesture-to-input pipeline** for professional CAD applications.

### Core Principles

1. **Zero Allocations in Hot Path**: No `new` keywords inside `Update()` loops. Gen 0 GC must remain at 0.
2. **LIFO Scheduling**: Latest frame wins. Drop backlog instantly if processing lags.
3. **Lock-Free IPC**: Interlocked operations only. No `lock`, no `Monitor`, no `SemaphoreSlim`.
4. **Type Safety**: No `object` or `dynamic`. Strict struct-based contracts.

### Latency Budget

**Target:** <30ms Motion-to-Photon (Hand Movement → Cursor Update)
**Measured:** ~15-20ms (current)

Breakdown:
- Webcam capture: <5ms
- Python inference: ~8-12ms (MediaPipe CPU)
- IPC transfer: <1ms (zero-copy shared memory)
- Cursor update: <2ms (120Hz extrapolation loop)

---

## Technology Stack

### C# (.NET 9)
- **Runtime:** Self-contained .NET 9 (desktop app)
- **Language Features:**
  - `unsafe` for pointer arithmetic
  - `readonly struct` for value semantics
  - `Span<byte>` for zero-copy buffer access
  - `Interlocked` for atomic operations
- **Dependencies:**
  - FlashCap (DirectShow webcam capture)
  - Microsoft.Extensions.Hosting (service orchestration)

### Python (3.12)
- **Purpose:** Vision sensor (MediaPipe inference sidecar)
- **Dependencies:**
  - `mediapipe==0.10.x` (hand landmark detection)
  - `numpy` (array operations)
- **IPC:** Zero-copy shared memory (memory-mapped files)

---

## System Components

### 1. The Data Plane (Memory & IPC)

#### SharedMemoryArena
- 256MB ring buffer for video frames (1920×1080 RGB)
- 30 slots × 8MB per slot
- Memory-mapped file: `Local\GrappleMap`
- **Zero-copy access** via `Span<byte> GetSpan(int bufferId)`

#### HandResultArena
- 4KB shared memory for hand tracking results
- 56-byte `HandState` struct (position + velocity + gesture + timestamp)
- Memory-mapped file: `Local\GrappleHandResults`
- Event-based signaling: `Local\GrappleHandSignal`

#### GraphPacket
- 16-byte value type representing a frame handle
- Fields: `BufferId` (int), `Timestamp` (long), `PayloadSize` (int)
- Passed by value between nodes

### 2. The Control Plane (The Governor)

#### AtomicMailbox
- Single-slot LIFO exchange
- `Publish(int id)`: Overwrites previous, returns old ID for recycling
- `Consume()`: Atomic swap with -1
- Lock-free: `Interlocked.Exchange` only
- Event-based consumer wakeup: `ManualResetEventSlim`

### 3. The Compute Plane (Nodes)

All nodes implement `IGraphNode` with `ValueTask StartAsync(CancellationToken)`.

#### WebcamCaptureNode (Producer)
- FlashCap 1920×1080 @ 60fps capture
- Direct write to `SharedMemoryArena`
- Publishes `GraphPacket` to downstream `AtomicMailbox`

#### GrappleDetector.py (Python Sidecar)
- MediaPipe Hands inference (CPU)
- Zero-copy frame consumption from shared memory
- Writes `HandState` to `HandResultArena`
- Pinch detection: Schmitt trigger + temporal debounce
- Landmark smoothing: 1€ Filter (cutoff=10, beta=2.1)

#### MouseControllerNode (Consumer)
- Reads `HandState` from `HandResultArena`
- 120Hz cursor update loop (decoupled from 15Hz inference)
- Velocity-based extrapolation for smooth motion
- 1€ Filter for cursor smoothing (cutoff=0.8, beta=0.02)
- F9 toggle for safety clutch
- Pinch-to-click gesture recognition

---

## Data Contracts (IPC Protocol)

### C# → Python (Video Frames)

**ArenaHeader** (40 bytes, offset 0):
```csharp
struct ArenaHeader {
    ulong MagicNumber;         // 0x31454C5050415247 ("GRAPPLE1")
    int SlotCount;             // 30
    int SlotSize;              // 8MB
    long WriteHeadIndex;       // Monotonic counter
    int PublishedBufferId;     // Latest frame for Python
    int _padding;
    long TimestampFrequency;   // Stopwatch.Frequency
}
```

**Frame Metadata** (64 bytes per slot, offset 1024 + slotId × 8MB):
```csharp
struct FrameMetadata {
    long Timestamp;      // QPC ticks
    int PayloadSize;     // Actual bytes used
    // 52 bytes padding
}
```

### Python → C# (Hand Data)

**HandResultHeader** (16 bytes, offset 0):
```python
struct HandResultHeader:
    magic: uint64        # 0x48414E4447525043 ("HANDGRPC")
    sequence: int64      # Monotonic counter
```

**HandState** (56 bytes, offset 64):
```python
# Format: '<dddddifq'
struct HandState:
    x: float64           # Normalized X (0.0-1.0)
    y: float64           # Normalized Y
    z: float64           # Normalized Z (depth)
    vx: float64          # Velocity X (units/sec)
    vy: float64          # Velocity Y
    gesture_id: int32    # 0=None, 1=Point, 2=Pinch
    confidence: float32  # MediaPipe confidence
    timestamp: int64     # Original frame QPC
```

**Alignment Verification:**
- C# `HandState` size: 56 bytes (verified via `Marshal.SizeOf<HandState>()`)
- Python pack format size: 56 bytes (verified via `struct.calcsize('<dddddifq')`)

---

## Build & Run

### Prerequisites
- .NET 9 SDK
- Python 3.12
- Visual C++ Redistributable (for FlashCap)

### Quick Start
```bash
# Build entire solution
dotnet build src/GrappleV2/GrappleGraphs.sln

# Run full pipeline (webcam → detector → mouse)
dotnet run --project src/GrappleV2/Grapple.SmokeTests -- --full

# Run individual tests
dotnet run --project src/GrappleV2/Grapple.SmokeTests -- --webcam
dotnet run --project src/GrappleV2/Grapple.SmokeTests -- --mouse

# Run Python detector standalone (for debugging)
python src/GrappleV2/tools/GrappleDetector.py
```

---

## Tuning Parameters

### Cursor Smoothing (C# MouseControllerNode)
```csharp
// 1€ Filter for cursor motion
minCutoff = 0.8;     // More responsive (lower = smoother but laggier)
beta = 0.02;         // Speed adaptation
dCutoff = 1.0;

// Other settings
Sensitivity = 1.3;   // Cursor speed multiplier
TargetUpdateHz = 120; // Cursor loop frequency
```

### Landmark Smoothing (Python GrappleDetector)
```python
# 1€ Filter for raw landmarks
LM_MIN_CUTOFF = 10    # High cutoff = responsive
LM_BETA = 2.1         # Speed-adaptive
LM_D_CUTOFF = 2.5

# Pinch detection
PINCH_THRESHOLD = 0.065   # Enter pinch
RELEASE_THRESHOLD = 0.12  # Exit pinch (hysteresis)
PINCH_DEBOUNCE = 2        # Frames to enter
RELEASE_DEBOUNCE = 3      # Frames to exit
```

### MediaPipe Configuration
```python
max_num_hands = 1
min_detection_confidence = 0.5
min_tracking_confidence = 0.4
```

---

## Known Limitations & Future Work

### Current Issues
- Drawing curved lines still somewhat choppy (frame dropping during drag)
- Single-hand mode only (left hand assumed)
- No multi-monitor/DPI scaling support
- No right-click or scroll gestures

### Roadmap (See ROADMAP.md)
- **0.9:** DPI scaling, multi-monitor, backpressure-aware queue
- **1.0:** gRPC over Named Pipes, Windows Service, SolidWorks add-in, MSIX installer
- **1.1:** AutoCAD/Inventor/NX plugins, auto-update service
- **1.2:** GPU inference (DirectML), <50ms p99 latency, security hardening

---

## Performance Telemetry

### Metrics Logged
- FPS (frames processed per second)
- Motion-to-Photon latency (hand → cursor)
- Dropped frames (mailbox overwrites)
- GC allocations (should be 0 in hot path)

### Validation
Run smoke tests to verify zero-GC:
```bash
dotnet run --project src/GrappleV2/Grapple.SmokeTests
```

Expected output:
```
[+] SUCCESS: Producer ran for 15 seconds with 0 Gen 0 collections
[+] SUCCESS: HandState is exactly 56 bytes
```

---

## File Structure

```
src/GrappleV2/
├── Grapple.Core/           # Zero-GC primitives
│   ├── SharedMemoryArena.cs
│   ├── AtomicMailbox.cs
│   ├── HandResultArena.cs
│   ├── HandState.cs
│   ├── GraphPacket.cs
│   ├── OneEuroFilter.cs
│   ├── PixelConverter.cs
│   └── Win32Input.cs
├── Grapple.Nodes/          # Graph nodes
│   ├── IGraphNode.cs
│   ├── WebcamCaptureNode.cs
│   ├── MouseControllerNode.cs
│   ├── NullSinkNode.cs
│   └── SyntheticCaptureNode.cs
├── Grapple.SmokeTests/     # Integration tests & runner
│   ├── Program.cs
│   ├── EndToEndTest.cs
│   ├── WebcamTest.cs
│   └── MouseControlTest.cs
├── Grapple.Service/        # (Future) Windows Service host
└── tools/
    ├── GrappleDetector.py  # Python vision sidecar
    └── debug_viewer.py     # Visualization tool
```

---

## Safety & Security

### Current Posture
- All processing on-device (no cloud dependency)
- F9 safety toggle (disable cursor control)
- No persistent storage of video frames
- Memory sanitization on arena slot recycling

### Future Hardening (1.2+)
- Least-privilege service account
- ACLs on shared memory handles
- Code signing for installers/executables
- Group Policy (ADMX) for IT admins
- Minidump integration for crash reporting

---

## Contributing

This is a clean-room V2 implementation. Key principles:

1. **Measure first.** Use Stopwatch, GC.GetTotalAllocatedBytes, and perf counters.
2. **No premature optimization.** Profile before tuning.
3. **Document tradeoffs.** Every tuning parameter should have a comment explaining "why".
4. **Test at boundaries.** Hand loss, camera disconnect, process crashes.

See `.claude/rules/` for coding standards.

---

## License

(TBD - proprietary for now)

---

**Last Updated:** 2025-12-20
**Architecture Version:** GrappleGraph V2