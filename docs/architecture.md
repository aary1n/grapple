# Grapple System Architecture

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
- Single-consumer enforcement: `RegisterConsumer()` / `UnregisterConsumer()` via `Interlocked.CompareExchange`

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
- Reads `SensorFrame` from `FlatBufferSensorArena` (FlatBuffer protocol v2)
- 120Hz cursor update loop (decoupled from 15Hz inference)
- Velocity-based extrapolation for smooth motion
- 1€ Filter for cursor smoothing (cutoff=0.8, beta=0.02)
- F9 toggle for safety clutch
- Pinch-to-click gesture recognition
- DPI-aware multi-monitor cursor via `SendInput` + `MOUSEEVENTF_VIRTUALDESK`
- Records end-to-end latency to `TelemetryCollector`

### 4. Observability

#### TelemetryCollector
- Lock-free counters: frames produced/dropped, consecutive drops, quality degradation
- Latency ring buffer (`double[256]`) with circular write index (bitwise AND masking)
- 10Hz `Timer` flush computes FPS, P50/P95/P99 via `stackalloc` + `Span<double>.Sort()`
- Writes FlatBuffer `TelemetrySnapshot` to `TelemetryArena`
- Python reader: `tools/telemetry_reader.py` (JSON-lines or human-readable output)

#### GrappleLogger
- Static JSON-lines logger to stdout (no external dependencies)
- Per-category throttle via `ConcurrentDictionary` (prevents log spam from hot loops)
- Configurable `MinLevel` (Debug/Info/Warning/Error/Silent) via `grapple_config.json`

#### DisplayInfo
- `readonly struct` with virtual desktop + primary screen dimensions
- `FromSystem()` factory for production; constructor injection for tests

---

## File Structure

```
src/GrappleIntent/                  # VLA research & model development
├── pyproject.toml                  # ML dependencies (torch, onnxruntime, transformers, peft, wandb)
├── configs/                        # Training/inference YAML configs
├── models/                         # Architecture definitions
│   ├── reflexive/                  # MobileNetV3 variants
│   ├── semantic/                   # VL-Transformer variants
│   └── adapters/                   # LoRA adapter definitions
├── training/                       # Training loops
├── inference/                      # Optimized inference + ONNX export
├── data/                           # Data loading + synthetic pipeline (Unity/Isaac)
├── evaluation/                     # Benchmarks, metrics, ablations
├── integration/                    # Bridge to C# IPC (FlatBuffer writers)
├── notebooks/                      # Exploration only (outputs gitignored)
└── tests/                          # Unit + integration tests

src/GrappleV2/
├── grapple_config.json         # Shared config (C# + Python read this)
├── schema/
│   └── grapple_protocol.fbs   # FlatBuffers schema
├── Grapple.Core/               # Zero-GC primitives
│   ├── SharedMemoryArena.cs
│   ├── AtomicMailbox.cs        # + RegisterConsumer/UnregisterConsumer
│   ├── HandResultArena.cs
│   ├── FlatBufferSensorArena.cs
│   ├── TelemetryArena.cs
│   ├── EyeResultArena.cs
│   ├── HandState.cs
│   ├── GraphPacket.cs
│   ├── OneEuroFilter.cs
│   ├── PixelConverter.cs
│   ├── Win32Input.cs           # + SendInput, VirtualDesktop, MoveMouseVirtual
│   ├── DisplayInfo.cs          # Virtual/primary screen abstraction
│   ├── GrappleConfig.cs        # All config classes
│   ├── GrappleLogger.cs        # JSON-lines structured logger
│   ├── TelemetryCollector.cs   # Lock-free metrics + 10Hz flush
│   └── Generated/
│       └── grapple_protocol_generated.cs
├── Grapple.Nodes/              # Graph nodes
│   ├── IGraphNode.cs
│   ├── WebcamCaptureNode.cs    # + TelemetryCollector integration
│   ├── MouseControllerNode.cs  # + SendInput, TelemetryCollector, GrappleLogger
│   ├── NullSinkNode.cs
│   └── SyntheticCaptureNode.cs
├── Grapple.Tests/              # 55 unit + integration tests
│   ├── ProtocolCompatibilityTests.cs
│   ├── ConfigTests.cs
│   ├── DisplayTests.cs
│   ├── AtomicMailboxTests.cs
│   ├── TelemetryTests.cs
│   └── IntegrationTests.cs
├── Grapple.SmokeTests/         # Pipeline runner
│   ├── Program.cs              # --full / --webcam / --mouse
│   ├── app.manifest            # PerMonitorV2 DPI awareness
│   └── ...
├── Grapple.Service/             # (Future) Windows Service host
│   └── app.manifest             # PerMonitorV2 DPI awareness
└── tools/
    ├── GrappleDetector.py       # Python vision sidecar
    ├── telemetry_reader.py      # LAM-readable telemetry consumer
    ├── debug_viewer.py          # Visualization tool
    ├── flatc.exe                # FlatBuffers compiler
    └── generated/               # Python FlatBuffer bindings
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
