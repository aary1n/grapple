### GrappleGraph Architecture (V2)

#### Phase 1: The Data Plane (Memory & IPC)
*Strict Zero-GC Requirement (Gen 0) in Hot Path*

- [x] **Core Primitive:** Define `GraphPacket` struct (16 bytes).
    - [x] Fields: `BufferId` (int), `Timestamp` (long), `PayloadSize` (int).
    - [x] Must be `readonly struct` passed by value.
- [x] **Arena Allocator:** Implement `SharedMemoryArena`.
    - [x] Allocate 256MB slab via `MemoryMappedFile` (Windows) or `NativeMemory` (Unix).
    - [x] Implement Ring Buffer logic for slot management.
    - [x] Expose `Span<byte> GetBuffer(int bufferId)` for zero-copy access.
    - [ ] Implement `IPacketAllocator` interface (Rent/Return semantics).
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

- [x] **Capture:** `CaptureNode` (Producer).
    - [x] Direct write to `Arena`.
    - [x] Pushes `GraphPacket` to Downstream Mailbox.
- [x] **Sink:** `NullSinkNode` (Consumer).
    - [x] Pulls from Mailbox via event-based signaling.
    - [x] Validates frame flow and measures latency.
- [ ] **Processing:** `DetectNode` (Consumer/Producer).
    - [ ] Wrapper for MediaPipe/ONNX.
    - [ ] Reads input `BufferId`, writes result to output `BufferId`.
- [ ] **Fusion:** `ZipNode` (IMU Sync).
    - [ ] "Last Known Good" policy: Poll IMU buffer for timestamp closest to Frame time.
    - [ ] No blocking/waiting for perfect alignment.

#### Phase 4: Integration & Validation

- [x] **Repo Structure:** Create `src/GrappleV2` for clean-room implementation.
- [ ] **Benchmarking:** Create Side-by-Side Harness.
    - [ ] Run V1 (Old Queue) vs V2 (Arena/Mailbox).
    - [ ] Measure: Allocations per frame (Target: 0B), Motion-to-Photon Latency (Target: <10ms).
- [ ] **Telemetry:** Expose `Arena.Usage` and `Mailbox.Drops` to UI overlay.

### Milestones
- [x] **M1 (The Spine):** `Arena` + `Mailbox` + `CaptureNode` running. Video frames flow to null sink with 0 GC.
- [ ] **M2 (The Brain):** `DetectNode` integrated. Hand coordinates extracted.
- [ ] **M3 (The Reflex):** `ZipNode` + Mouse Input. Full loop closed.
- [ ] **M4 (Hardening):** Shared Memory IPC exposed to Python/External processes.

### Acceptance Criteria
- [x] **Zero Allocations:** No `new` keywords inside the `Run()` loop.
- [x] **LIFO Behavior:** If Detector lags, frames are dropped instantly (Mailbox overwrite).
- [x] **Type Safety:** No `object` or `dynamic`. Strict struct usage.