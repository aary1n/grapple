# Grapple - OS-Level Gesture Controller

## Project Manifest

**Version:** GrappleV2 (Clean-Room Implementation) + GrappleIntent (VLA Research)
**Status:** M4 Complete - Production Hardened | GrappleIntent Phase 0 (Infrastructure)
**Target:** Enterprise CAD Integration (SolidWorks First) → Hierarchical VLA-Driven HCI

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

### Python (3.11.9)
- **Purpose:** Vision sensor (MediaPipe inference sidecar)
- **Dependencies:**
  - `mediapipe==0.10.x` (hand landmark detection)
  - `numpy` (array operations)
- **IPC:** Zero-copy shared memory (memory-mapped files)

### ML Stack (GrappleIntent)
- **Framework:** PyTorch ≥2.2 (training), ONNX Runtime ≥1.17 (inference)
- **Model Architectures:** `timm` (MobileNetV3), `transformers` (VL-Transformer)
- **Fine-tuning:** `peft` (LoRA adapters)
- **Quantization:** `autoawq` (INT4 with salient weight preservation)
- **Experiment Tracking:** Weights & Biases (`wandb`)
- **Runtimes:** ONNX Runtime CPU (default), DirectML (GPU), TensorRT (NVIDIA)
- **Project:** `src/GrappleIntent/pyproject.toml`

---

# GrappleIntent: Hierarchical Vision-Language-Action System

## Overview

GrappleIntent replaces rule-based gesture detection (MediaPipe + Schmitt trigger) with a **learned hierarchical VLA** that understands user intent, not just hand position.

```
Camera → SharedMemoryArena → [GrappleIntent] → FlatBufferSensorArena → MouseControllerNode
                                    │
                         ┌──────────┴──────────┐
                    Reflexive Path         Semantic Path
                    (MobileNetV3)         (VL-Transformer)
                     120Hz, ≤10ms          10Hz, ≤100ms
                    cursor control        intent parsing
```

## Dual Inference Paths

### Reflexive Path (Fast — Cursor Control)
- **Model:** Quantized MobileNetV3 backbone
- **Rate:** 120Hz (synced with cursor extrapolation loop)
- **Latency:** ≤10ms hard limit, ≤5ms target
- **Input:** Hand landmarks + velocity vector
- **Output:** Cursor delta (dx, dy) + gesture confidence
- **Quantization:** INT4-AWQ (top 1% attention heads in FP16, remainder INT4)

### Semantic Path (Slow — Intent Parsing)
- **Model:** Vision-Language Transformer
- **Rate:** 10Hz (decoupled from cursor)
- **Latency:** ≤100ms hard limit, ≤50ms target
- **Input:** `<ImagePatch>` + `<GazeVector>` + `<HandVelocity>` + `<UIContext>`
- **Output:** Intent classification + 2D Gaussian heatmap (probabilistic intent field)
- **Purpose:** Contextual actions ("select that button", "scroll this panel")

### Fusion
Reflexive runs continuously. Semantic overrides when: confidence > threshold AND prediction is fresh AND no active reflexive gesture.

## Novel Methods

### Probabilistic Intent Fields
Replaces deterministic coordinate mapping with `P(target | hand, gaze, ui) = Gaussian(μ, Σ) × Saliency(ui)`. Solves "Midas Touch" by requiring both high spatial confidence AND gesture confirmation.

### Speculative Trajectory Decoding
Adapted from LLM speculative decoding: batch-predict K=5 future cursor positions, verify against physics constraints (max velocity, screen bounds), commit valid prefix.

### IRL Error-Correction
Learns personalized reward functions from observed retry patterns (pinch → miss → release → retry). Per-user, on-device, updated asynchronously.

### LoRA Calibration
5-10 anchor gestures → fine-tune LoRA adapter (r=8) on frozen backbone → hot-swappable at runtime (≤500ms). Base model is never modified.

## Integration Contract

**Boundary:** GrappleIntent writes `SensorFrame` FlatBuffers to `FlatBufferSensorArena`. The C# pipeline is completely agnostic to the upstream model — it reads the same FlatBuffer format whether data comes from MediaPipe rules or a neural network.

- Arena magic: `0x4C505247` ("GRPL")
- Python/ONNX can allocate freely (separate process)
- C# reads with pre-allocated buffer (zero-GC preserved)

## Latency Budgets

```
Reflexive Path (120Hz, ≤10ms total):
├── Frame read:          <0.5ms
├── Preprocessing:       ≤1ms
├── MobileNetV3:         ≤5ms
├── Postprocessing:      ≤1ms
├── FlatBuffer write:    <0.5ms
└── Headroom:            ~2ms

Semantic Path (10Hz, ≤100ms total):
├── Frame + context:     ≤2ms
├── Token fusion:        ≤5ms
├── VL-Transformer:      ≤70ms
├── Intent field:        ≤10ms
├── FlatBuffer write:    <1ms
└── Headroom:            ~12ms
```

## Directory Structure

```
src/GrappleIntent/
├── pyproject.toml         # ML dependencies
├── configs/               # Training/inference YAML configs
├── models/                # Architecture definitions
│   ├── reflexive/         # MobileNetV3 variants
│   ├── semantic/          # VL-Transformer variants
│   └── adapters/          # LoRA adapter definitions
├── training/              # Training loops
├── inference/             # Optimized inference + ONNX export
├── data/                  # Data loading + synthetic pipeline
├── evaluation/            # Benchmarks, metrics, ablations
├── integration/           # Bridge to C# IPC (FlatBuffer writers)
├── notebooks/             # Exploration only
└── tests/                 # Unit + integration tests
```

## Rules & Commands

- **Rules:** `.claude/rules/ml-research.md` (experiment conventions), `.claude/rules/vla-architecture.md` (design patterns)
- **Commands:** `.claude/commands/research.md` (`/train`, `/eval`, `/benchmark`, `/convert-onnx`, `/quantize`, `/calibrate`, `/ablation`)
- **Tracking:** All experiments logged to W&B project `grapple-intent`

---

# MCP TOOLING PROTOCOLS

## Overview
Grapple development requires precise tool usage. **Never guess** library APIs, database schemas, or file structures. Use MCP tools to fetch authoritative information.

---

## 1. Context7 (Documentation Fetching)

### When to Use
**TRIGGER on these patterns:**
- Mentioning specific library versions (e.g., ".NET 9", "MediaPipe 0.10.x", "FlashCap")
- Working with C# language features (`Span<byte>`, `unsafe`, `Interlocked`, `readonly struct`)
- Python dependency operations (`numpy`, `mediapipe`)
- ML/AI library usage (`torch`, `transformers`, `onnxruntime`, `peft`, `autoawq`, `timm`)
- Questions about API signatures, best practices, or version-specific behavior

**ACTION:**
```
Explicitly call: use context7 to fetch docs for [Library + Version]
```

**Examples:**
- "I need to optimize Span<byte> usage" → `context7: .NET 9 Span documentation`
- "How do I configure FlashCap?" → `context7: FlashCap API reference`
- "MediaPipe hand landmark structure" → `context7: MediaPipe 0.10 hand tracking`
- "LoRA adapter fine-tuning" → `context7: PEFT LoRA configuration`
- "ONNX Runtime DirectML provider" → `context7: ONNX Runtime DirectML execution provider`
- "AWQ quantization" → `context7: AutoAWQ quantization API`
- "PyTorch model export" → `context7: PyTorch ONNX export`

**CONSTRAINT:**
- Do NOT infer API signatures from memory or old examples
- If uncertain about a method signature, property, or pattern → **fetch docs first**

**FALLBACK:**
- If Context7 fails or returns insufficient info → Ask user for official docs URL
- Then use `read` tool to fetch the specific page

---

## 2. Filesystem (Code Navigation & Analysis)

### When to Use
**TRIGGER on these patterns:**
- "Where is [component] implemented?"
- "Show me the current [service/class/module]"
- Before suggesting refactors or architectural changes
- When debugging requires seeing actual file structure

**ACTION:**
```
1. Use filesystem to list directories and locate relevant files
2. Read specific files to understand current implementation
3. Only then propose changes based on ACTUAL code, not assumptions
```

**Examples:**
- "How is the gesture pipeline structured?" → List directories → Read key files
- "Where does IPC happen?" → Search for memory-mapped file usage
- "Current camera capture implementation?" → Locate FlashCap integration files

**CONSTRAINT:**
- Do NOT suggest code changes without first reading existing implementation
- Do NOT assume project structure matches typical patterns

**PROHIBITED:**
- "I assume you have a Services folder..." ❌
- "Typically this would be in..." ❌
- CORRECT: "Let me check your current structure..." ✅

---

## 3. PostgreSQL (Database Schema Authority)

### When to Use
**TRIGGER on these patterns:**
- Designing database migrations
- Writing complex queries (JOINs, CTEs, aggregations)
- Performance optimization requiring index analysis
- Schema questions ("What columns does X have?")

**ACTION:**
```
1. Connect to postgresql via MCP
2. Introspect schema: \d table_name, \di (indexes), \df (functions)
3. Base ALL query/migration work on live schema
```

**Examples:**
- "Create a query for gesture history" → First: inspect gesture_events table schema
- "Add index for performance" → First: check existing indexes with \di
- "Migration to add field" → First: verify current table structure

**CONSTRAINT:**
- Do NOT infer schema from old migration files or code comments
- Do NOT assume column names, types, or constraints

**PROHIBITED:**
- "Based on typical schemas..." ❌
- "I'll assume the table has..." ❌
- CORRECT: "Let me check the actual schema..." ✅

---

## Tool Priority Matrix

| Scenario | Primary Tool | Secondary Tool | Fallback |
|----------|--------------|----------------|----------|
| API usage question | Context7 | User-provided URL + read | Documentation comment |
| Code structure question | Filesystem | - | Ask user |
| Database query | PostgreSQL introspection | - | Ask user for schema |
| Library version conflict | Context7 (specific version) | Release notes URL | Ask user |

---

## Anti-Patterns to Avoid

### ❌ DON'T:
```
"In .NET, you typically use Task.Run for async work..."
(Without checking .NET 9 specific patterns)
```

### ✅ DO:
```
"Let me fetch .NET 9 async best practices first."
[calls context7: .NET 9 Task and async/await patterns]
```

### ❌ DON'T:
```
"Your database probably has a users table with id, name, email..."
```

### ✅ DO:
```
"Let me inspect your database schema."
[calls postgresql: \dt, \d users]
```

### ❌ DON'T:
```
"I'll create the GestureProcessor class in Services/..."
```

### ✅ DO:
```
"Let me check your current project structure."
[calls filesystem: list directories, locate existing services]
```

---

## Zero-Tolerance Rules

1. **Never guess library APIs** - If you don't have Context7 docs, explicitly say "I need to fetch documentation for [X] before proceeding"

2. **Never assume schema** - If you need DB structure, say "I need to inspect the database schema first"

3. **Never fake file paths** - If you need to know where something is, say "Let me navigate your filesystem to locate [X]"

4. **When in doubt, fetch** - It's better to make 3 tool calls and be accurate than make 0 calls and be wrong

---

## Integration with Grapple Workflow

Since Grapple is performance-critical (zero-GC, ultra-low-latency), tool usage is especially important:

- **Before suggesting `Span<T>` usage** → Context7: .NET 9 Span best practices
- **Before modifying IPC layer** → Filesystem: read current shared memory implementation
- **Before query optimization** → PostgreSQL: check actual query plans with EXPLAIN ANALYZE
- **Before dependency updates** → Context7: check version compatibility for MediaPipe, FlashCap

---

## Quick Reference

```
Library API unclear?        → context7 [library] [version]
Don't know file location?   → filesystem list/read
Need schema info?           → postgresql \d [table]
Documentation link broken?  → read [url]
```

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

All tuning parameters are externalized in `src/GrappleV2/grapple_config.json` (loaded at startup by `GrappleConfigLoader`). Both C# and Python read from this file. Defaults are shown below.

### Cursor Smoothing (`cursor` section)
```json
{
  "updateHz": 120,
  "sensitivity": 1.3,
  "minConfidence": 0.5,
  "maxExtrapolationSec": 0.15,
  "velocityDecay": 0.95,
  "filter": { "minCutoff": 0.8, "beta": 0.02, "dCutoff": 1.0 }
}
```

### Landmark Smoothing (`landmarkFilter` section)
```json
{ "minCutoff": 10.0, "beta": 2.1, "dCutoff": 2.5 }
```

### Pinch Detection (`pinch` section)
```json
{
  "enterThreshold": 0.30, "exitThreshold": 0.70,
  "exitDebounceMs": 150,
  "enterConfirmFrames": 3, "exitConfirmFrames": 5,
  "ratioAlphaDown": 0.5, "ratioAlphaUp": 0.3,
  "noHandGraceFrames": 5, "minScaleThreshold": 0.001,
  "maxRatio": 0.9, "velocitySmooth": 0.3
}
```

### MediaPipe Configuration (`mediapipe` section)
```json
{ "maxHands": 1, "minDetectionConfidence": 0.5, "minTrackingConfidence": 0.4, "modelComplexity": 0 }
```

### Telemetry (`telemetryCollection` section)
```json
{ "enabled": true, "flushIntervalMs": 100, "maxLatencySamples": 256 }
```

### Logging (`logging` section)
```json
{ "minLevel": "Info" }
```

---

## Known Limitations & Future Work

### Current Issues
- Drawing curved lines still somewhat choppy (frame dropping during drag)
- Single-hand mode only (left hand assumed)
- No right-click or scroll gestures

### Roadmap (See ROADMAP.md)
- **0.9:** ~~DPI scaling, multi-monitor~~ DONE, ~~backpressure-aware queue~~ DONE, ~~observability~~ DONE. Remaining: health model, circuit breaker, crash handling.
- **1.0:** gRPC over Named Pipes, Windows Service, SolidWorks add-in, MSIX installer
- **1.1:** AutoCAD/Inventor/NX plugins, auto-update service
- **1.2:** GPU inference (DirectML), <50ms p99 latency, security hardening

---

## Performance Telemetry

### Real-Time Metrics (via TelemetryCollector → TelemetryArena)
- FPS (frames processed per second)
- End-to-end latency: P50, P95, P99 (ms)
- Total frames produced / dropped
- Consecutive drops + quality degradation flag
- GC Gen 0/1/2 collection counts
- Uptime (seconds)

### Reading Telemetry
```bash
# Human-readable (polls every 0.5s)
python src/GrappleV2/tools/telemetry_reader.py

# JSON-lines (for LAM / dashboard consumption)
python src/GrappleV2/tools/telemetry_reader.py --json

# Single snapshot
python src/GrappleV2/tools/telemetry_reader.py --once --json
```

### Validation
Run unit + integration tests (55 tests):
```bash
dotnet test src/GrappleV2/Grapple.Tests/Grapple.Tests.csproj
```

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

**Last Updated:** 2026-02-11
**Architecture Version:** GrappleGraph V2 (Phase 4 Complete) + GrappleIntent V0 (Infrastructure)