### GrappleGraph Architecture (V2)

#### Phase 1: The Data Plane (Memory & IPC)
*Strict Zero-GC Requirement (Gen 0) in Hot Path*

- [x] **Core Primitive:** Define `GraphPacket` struct (16 bytes).
    - [x] Fields: `BufferId` (int), `Timestamp` (long), `PayloadSize` (int).
    - [x] Must be `readonly struct` passed by value.
- [x] **Arena Allocator:** Implement `SharedMemoryArena`.
    - [x] Allocate 256MB slab via `MemoryMappedFile` (Windows).
    - [x] Implement Ring Buffer logic for slot management.
    - [x] Expose `Span<byte> GetBuffer(int bufferId)` for zero-copy access.
    - [ ] Implement `IPacketAllocator` interface (Rent/Return semantics).
- [x] **Hand Result Arena:** Implement `HandResultArena` for Python↔C# hand data transfer.
    - [x] 40-byte `HandState` struct (X, Y, Z, GestureId, Confidence, Timestamp).
    - [x] Shared memory IPC with event-based signaling.
- [ ] **Safety:** Implement "Lease Tracking" (Optional) to detect if a Node holds a buffer too long.

#### Phase 2: The Control Plane (The Governor)
*LIFO / Drop-Oldest Scheduling Policy*

- [x] **Concurrency:** Implement `AtomicMailbox` ("The Governor").
    - [x] Single-slot storage.
    - [x] `Publish(int id)`: Overwrites existing ID, returns old ID for recycling.
    - [x] `Consume()`: Atomic swap with -1.
    - [x] Use `Interlocked.Exchange`; strictly NO `lock` or `Monitor`.
    - [x] Event-based signaling (`ManualResetEventSlim`) for low-latency consumer wakeup.
- [x] **Scheduler:** Implement `IGraphNode` contract.
    - [x] Use `ValueTask` to prevent Task allocation overhead.
    - [x] Nodes should "pull" from Mailboxes or be triggered by Mailbox signals.

#### Phase 3: The Compute Plane (Nodes)

- [x] **Capture:** `WebcamCaptureNode` (Producer).
    - [x] Direct write to `Arena`.
    - [x] Pushes `GraphPacket` to Downstream Mailbox.
    - [x] 1920x1080 @ 60fps capture via DirectShow.
- [x] **Sink:** `NullSinkNode` (Consumer).
    - [x] Pulls from Mailbox via event-based signaling.
    - [x] Validates frame flow and measures latency.
- [x] **Processing:** `GrappleDetector.py` (Python Sidecar).
    - [x] MediaPipe Hands inference.
    - [x] Zero-copy frame consumption from shared memory.
    - [x] Writes hand state to `HandResultArena`.
    - [x] Pinch detection with Schmitt trigger + temporal debounce.
- [x] **Mouse Control:** `MouseControllerNode`.
    - [x] Reads hand data from `HandResultArena`.
    - [x] 1€ Filter for smooth cursor movement.
    - [x] Pinch-to-click gesture recognition.
    - [x] F9 toggle for safety clutch.
- [ ] **Fusion:** `ZipNode` (IMU Sync).
    - [ ] "Last Known Good" policy: Poll IMU buffer for timestamp closest to Frame time.
    - [ ] No blocking/waiting for perfect alignment.

#### Phase 4: Integration & Validation

- [x] **Repo Structure:** Create `src/GrappleV2` for clean-room implementation.
- [x] **Full Pipeline:** Single-click launch via `dotnet run -- --full`.
    - [x] Webcam → Python Detector → Mouse Controller.
    - [x] Automatic Python process management.
- [ ] **Benchmarking:** Create Side-by-Side Harness.
    - [ ] Run V1 (Old Queue) vs V2 (Arena/Mailbox).
    - [ ] Measure: Allocations per frame (Target: 0B), Motion-to-Photon Latency (Target: <10ms).
- [ ] **Telemetry:** Expose `Arena.Usage` and `Mailbox.Drops` to UI overlay.

### Milestones
- [x] **M1 (The Spine):** `Arena` + `Mailbox` + `CaptureNode` running. Video frames flow to null sink with 0 GC.
- [x] **M2 (The Brain):** Python `GrappleDetector` integrated. Hand coordinates extracted via shared memory.
- [x] **M3 (The Reflex):** `MouseControllerNode` + Pinch-to-Click. Full loop closed.
- [x] **M4 (Hardening):** Shared Memory IPC working between C# and Python processes.

### Acceptance Criteria
- [x] **Zero Allocations:** No `new` keywords inside the `Run()` loop.
- [x] **LIFO Behavior:** If Detector lags, frames are dropped instantly (Mailbox overwrite).
- [x] **Type Safety:** No `object` or `dynamic`. Strict struct usage.

---

### Usability Issues (Addressed)

#### Click Detection
- [x] Schmitt trigger with hysteresis (PINCH < 0.065, RELEASE > 0.12).
- [x] Temporal debounce (2 frames to enter, 3 frames to exit).
- [x] State-dependent counter logic to prevent click spam.
- [x] EMA smoothing on pinch distance (α=0.3).

#### Cursor Stability
- [x] 1€ Filter with tuned parameters (minCutoff=0.4, beta=0.01).
- [x] 1.3x sensitivity multiplier (center-anchored).
- [x] Teleport protection during drag (reject >200px jumps).
- [x] Motion interpolation for smooth dragging.

#### Hand Detection
- [x] MediaPipe with lower thresholds (detection=0.5, tracking=0.4).
- [x] Single-hand mode (max_num_hands=1).

### Known Issues / Future Work

- [ ] Drawing curved lines still somewhat choppy (frame dropping during drag).
- [ ] Consider buffering frames during active drag instead of dropping.
- [ ] Multi-monitor support and DPI scaling.
- [ ] Right-click gesture (e.g., two-finger pinch or hold duration).
- [ ] Scroll gesture (e.g., open palm swipe).
- [ ] Visual feedback overlay (cursor ring, click indicator).
