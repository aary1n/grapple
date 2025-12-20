# C# Core Rules - Grapple Zero-GC Architecture

## Mission-Critical Constraints

These rules are **NON-NEGOTIABLE** for the Grapple hot path. Violating them breaks the latency budget.

---

## 1. Zero Allocations in Hot Path

**Rule:** NO `new` keywords inside `Update()`, `Run()`, or any method called at >10Hz frequency.

### Forbidden Patterns
```csharp
// ❌ BAD: Allocates on every frame
public void ProcessFrame()
{
    var buffer = new byte[1920 * 1080 * 3];  // ALLOCATES!
    var packet = new GraphPacket(...);       // Boxes if not struct!
}

// ❌ BAD: String allocations
Console.WriteLine($"Frame {frameId}");       // Allocates string!

// ❌ BAD: LINQ allocates enumerators
var recent = frames.Where(f => f.Age < 100).ToList();
```

### Correct Patterns
```csharp
// ✅ GOOD: Pre-allocated buffer reuse
private readonly byte[] _buffer = new byte[1920 * 1080 * 3];

public void ProcessFrame()
{
    // Use existing buffer
    Span<byte> target = _buffer;
}

// ✅ GOOD: Struct copy (stack-only, no heap)
public GraphPacket AcquireSlot()
{
    return new GraphPacket(bufferId, timestamp, size);  // OK for structs!
}

// ✅ GOOD: Span-based zero-copy
public Span<byte> GetFrameData(int bufferId)
{
    return new Span<byte>(_basePtr + offset, size);
}
```

### Verification
Run smoke tests and check GC metrics:
```csharp
long gcBefore = GC.GetTotalAllocatedBytes(precise: true);
// ... run hot path ...
long gcAfter = GC.GetTotalAllocatedBytes(precise: true);
Assert.AreEqual(0, gcAfter - gcBefore, "Hot path allocated bytes!");
```

---

## 2. Unsafe Pointer Arithmetic

**Rule:** Use `unsafe` for direct memory access in arenas. Bounds-check manually.

### Requirements
- Enable `<AllowUnsafeBlocks>true</AllowUnsafeBlocks>` in `.csproj`
- Always validate `bufferId` range before pointer math
- Use `[MethodImpl(MethodImplOptions.AggressiveInlining)]` for hot accessors

### Pattern
```csharp
public unsafe Span<byte> GetSpan(int bufferId)
{
    // Bounds check (critical!)
    if ((uint)bufferId >= (uint)_headerPtr->SlotCount)
    {
        throw new IndexOutOfRangeException();
    }

    // Pointer arithmetic
    long byteOffset = FirstSlotOffset + ((long)bufferId * _headerPtr->SlotSize) + MetadataSize;
    return new Span<byte>(_basePtr + byteOffset, _headerPtr->SlotSize - MetadataSize);
}
```

### Safety Notes
- Always use `(uint)` cast for bounds checks (handles negative values correctly)
- Never dereference pointers outside the mapped region
- Release pointers on `Dispose()`: `_accessor.SafeMemoryMappedViewHandle.ReleasePointer()`

---

## 3. Lock-Free Concurrency

**Rule:** Use `Interlocked` operations ONLY. No `lock`, `Monitor`, `SemaphoreSlim`, `Mutex`.

### Forbidden Patterns
```csharp
// ❌ BAD: Blocking lock
private readonly object _lock = new object();
public void Publish(int id)
{
    lock (_lock) { _currentId = id; }  // BLOCKS!
}

// ❌ BAD: Monitor (same as lock)
Monitor.Enter(_lock);

// ❌ BAD: Async locks allocate
await _semaphore.WaitAsync();
```

### Correct Patterns
```csharp
// ✅ GOOD: Atomic exchange (LIFO mailbox)
public int Publish(int newId)
{
    return Interlocked.Exchange(ref _currentId, newId);
}

// ✅ GOOD: Atomic read
public int Consume()
{
    return Interlocked.Exchange(ref _currentId, -1);
}

// ✅ GOOD: Monotonic counter
long nextIndex = Interlocked.Increment(ref _writeHead) - 1;

// ✅ GOOD: Volatile write (ordering guarantee)
Volatile.Write(ref _headerPtr->PublishedBufferId, bufferId);
```

### Event-Based Signaling
Allowed for consumer wakeup (not in hot path):
```csharp
private readonly ManualResetEventSlim _signal = new(false);

// Producer (non-blocking)
_signal.Set();

// Consumer (blocking wait is OK here)
_signal.Wait(cancellationToken);
```

---

## 4. Struct Layout & Value Semantics

**Rule:** Use `readonly struct` for data packets. Explicit layout for IPC contracts.

### Packet Structs
```csharp
[StructLayout(LayoutKind.Explicit, Size = 16)]
public readonly struct GraphPacket
{
    [FieldOffset(0)]
    public readonly int BufferId;

    [FieldOffset(4)]
    public readonly int PayloadSize;

    [FieldOffset(8)]
    public readonly long Timestamp;

    public GraphPacket(int bufferId, long timestamp, int payloadSize)
    {
        BufferId = bufferId;
        Timestamp = timestamp;
        PayloadSize = payloadSize;
    }
}
```

### IPC Structs
**CRITICAL:** Must match Python `struct.pack` format byte-for-byte!

```csharp
// C# Side (56 bytes)
[StructLayout(LayoutKind.Explicit, Size = 56)]
public readonly struct HandState
{
    [FieldOffset(0)]  public readonly double X;   // 8 bytes
    [FieldOffset(8)]  public readonly double Y;   // 8 bytes
    [FieldOffset(16)] public readonly double Z;   // 8 bytes
    [FieldOffset(24)] public readonly double VX;  // 8 bytes
    [FieldOffset(32)] public readonly double VY;  // 8 bytes
    [FieldOffset(40)] public readonly int GestureId;  // 4 bytes
    [FieldOffset(44)] public readonly float Confidence; // 4 bytes
    [FieldOffset(48)] public readonly long Timestamp;   // 8 bytes
}

// Python Side (MUST MATCH!)
# Format: '<dddddifq' = 56 bytes
struct.pack('<dddddifq', x, y, z, vx, vy, gesture_id, confidence, timestamp)
```

**Validation:**
```csharp
int size = Marshal.SizeOf<HandState>();
Assert.AreEqual(56, size, "HandState size mismatch!");
```

---

## 5. No Boxing / Dynamic Dispatch

**Rule:** Avoid `object`, `dynamic`, interface dispatch in hot path.

### Forbidden
```csharp
// ❌ BAD: Boxing
object boxed = myStruct;  // Allocates!

// ❌ BAD: Interface dispatch (virtual call overhead)
IGraphNode node = GetNode();
await node.ProcessAsync();  // Indirect call

// ❌ BAD: Dynamic
dynamic config = GetConfig();
config.Value = 123;  // Slow reflection
```

### Allowed
```csharp
// ✅ GOOD: Direct struct usage
GraphPacket packet = arena.AcquireNextSlot(timestamp, size);

// ✅ GOOD: Generic constraints (devirtualized at compile time)
public T Process<T>(T input) where T : struct
{
    return input;
}

// ✅ GOOD: Interface dispatch OK for startup/shutdown (not hot path)
public ValueTask StartAsync(CancellationToken ct)  // Called once
{
    // Interface is fine here
}
```

---

## 6. Memory Safety in Arenas

**Rule:** Never return pointers that outlive the arena. Use `Span<byte>` for safety.

### Correct Pattern
```csharp
// ✅ GOOD: Span prevents pointer escape
public void WriteFrame(int bufferId, ReadOnlySpan<byte> data)
{
    Span<byte> target = GetSpan(bufferId);  // Safe: Span is stack-bound
    data.CopyTo(target);
}

// ✅ GOOD: Return Span, not pointer
public Span<byte> GetSpan(int bufferId)
{
    return new Span<byte>(_basePtr + offset, size);
}
```

### Forbidden Pattern
```csharp
// ❌ BAD: Returning raw pointer (caller could use after Dispose!)
public unsafe byte* GetPointer(int bufferId)
{
    return _basePtr + offset;  // DANGER!
}
```

---

## 7. Dispose Pattern

**Rule:** Always implement `IDisposable` for types holding unmanaged resources.

### Pattern
```csharp
public class SharedMemoryArena : IDisposable
{
    private bool _disposed;

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (_basePtr != null)
            {
                _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
            }

            if (disposing)
            {
                _accessor?.Dispose();
                _mmf?.Dispose();
            }

            _disposed = true;
        }
    }

    ~SharedMemoryArena()
    {
        Dispose(false);
    }
}
```

---

## 8. Nullable Reference Types

**Rule:** Enable `<Nullable>enable</Nullable>`. Treat warnings as errors.

```csharp
// ✅ GOOD: Explicit nullability
public HandState? TryReadHandState()
{
    if (!_arena.IsValid())
        return null;

    return _arena.ReadHandState();
}

// ✅ GOOD: Non-nullable fields
private readonly SharedMemoryArena _arena = null!;  // Initialized in ctor

public MyClass()
{
    _arena = new SharedMemoryArena();  // Set before any use
}
```

---

## 9. Async/Await

**Rule:** Use `ValueTask` instead of `Task` for hot paths (avoids allocation if synchronous).

```csharp
// ✅ GOOD: ValueTask for IGraphNode contract
public ValueTask StartAsync(CancellationToken ct)
{
    // If starting synchronously
    DoSyncSetup();
    return ValueTask.CompletedTask;  // No allocation
}

// ❌ BAD: Task allocates even if sync
public Task StartAsync(CancellationToken ct)
{
    DoSyncSetup();
    return Task.CompletedTask;  // Allocates Task object
}
```

**Exception:** Long-running background threads use `Task.Factory.StartNew` with `TaskCreationOptions.LongRunning`.

---

## 10. Constants & Configuration

**Rule:** Use `const` or `readonly` fields. NO runtime config in hot path.

```csharp
// ✅ GOOD: Compile-time constant
private const int TargetSlotSize = 8 * 1024 * 1024;

// ✅ GOOD: Readonly field (set once in constructor)
private readonly double _minCutoff;

public OneEuroFilter(double minCutoff)
{
    _minCutoff = minCutoff;  // OK in ctor
}

// ❌ BAD: Reading config file in hot path
public void ProcessFrame()
{
    var config = File.ReadAllText("config.json");  // SLOW!
}
```

---

## 11. Logging in Hot Path

**Rule:** NO string interpolation, NO Console.WriteLine in loops.

### Forbidden
```csharp
// ❌ BAD: Allocates string on every frame
Console.WriteLine($"Frame {frameId} at {timestamp}");
```

### Allowed
```csharp
// ✅ GOOD: Throttled logging (e.g., every 100 frames)
if (frameId % 100 == 0)
{
    Console.WriteLine($"Processed 100 frames");
}

// ✅ GOOD: Metrics without strings
Interlocked.Increment(ref _frameCount);

// ✅ GOOD: Structured logging (outside hot path)
_logger.LogInformation("Pipeline started");
```

---

## 12. Exception Handling

**Rule:** Avoid exceptions in hot path (they allocate stack traces). Validate upfront.

```csharp
// ✅ GOOD: Early validation
public Span<byte> GetSpan(int bufferId)
{
    if ((uint)bufferId >= (uint)_slotCount)
    {
        throw new ArgumentOutOfRangeException(nameof(bufferId));  // OK: happens BEFORE loop
    }
    // ...
}

// ❌ BAD: Try-catch in loop
for (int i = 0; i < 1000; i++)
{
    try
    {
        ProcessFrame(i);
    }
    catch (Exception ex)  // Allocates on throw!
    {
        Log(ex);
    }
}
```

---

## Verification Checklist

Before committing changes to `Grapple.Core` or `Grapple.Nodes`:

- [ ] Run smoke tests: `dotnet run --project Grapple.SmokeTests`
- [ ] Verify 0 Gen 0 GC collections during 15-second run
- [ ] Check `Marshal.SizeOf<HandState>() == 56`
- [ ] No `new` keywords in methods called >10Hz
- [ ] No `lock`, `Monitor`, `SemaphoreSlim` in hot path
- [ ] All `unsafe` code has bounds checks
- [ ] All `IDisposable` types have finalizers
- [ ] All structs use `readonly struct` where applicable

---

**Last Updated:** 2025-12-20
**Enforced By:** Code review + automated tests
