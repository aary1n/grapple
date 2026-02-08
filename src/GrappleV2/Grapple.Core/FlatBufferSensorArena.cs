using System;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Google.FlatBuffers;
using Grapple.Protocol;

namespace Grapple.Core
{
    /// <summary>
    /// Header structure for FlatBuffer-based sensor arena.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct FlatBufferArenaHeader
    {
        public ulong MagicNumber;      // Offset 0,  8 bytes - "GRPL" file identifier
        public long SequenceNumber;    // Offset 8,  8 bytes (monotonic counter)
        public int ProtocolVersion;    // Offset 16, 4 bytes (FlatBuffers schema version)
        public int BufferSize;         // Offset 20, 4 bytes (size of serialized FlatBuffer)
        public long TimestampFrequency;// Offset 24, 8 bytes (Stopwatch.Frequency)
        // Total: 32 bytes (8-byte aligned)
    }

    /// <summary>
    /// Shared memory reader for FlatBuffer-serialized sensor data.
    /// Zero-copy deserialization via ByteBuffer wrapper.
    /// Replaces HandResultArena with versioned multi-modal protocol.
    /// </summary>
    public unsafe class FlatBufferSensorArena : IDisposable
    {
        // Configuration
        private const string MapName = "Local\\GrappleSensorArena";
        private const string SignalName = "Local\\GrappleSensorSignal";
        private const long MapCapacity = 8192; // 8KB (supports full SensorFrame with landmarks)
        private const int DataOffset = 64; // FlatBuffer data starts here (aligned)

        // "GRPL" file identifier as ulong (little-endian)
        private const ulong MagicSignature = 0x4C505247; // "GRPL" in ASCII

        // Protocol version (must match schema file_identifier)
        private const int CurrentProtocolVersion = 2;

        private readonly MemoryMappedFile _mmf;
        private readonly MemoryMappedViewAccessor _accessor;
        private readonly byte* _basePtr;
        private readonly FlatBufferArenaHeader* _headerPtr;
        private readonly EventWaitHandle _signal;
        private bool _disposed;

        // Pre-allocated read buffer to avoid per-frame allocations
        // Inference reader runs at ~15Hz (not 120Hz hot path) but we still minimize GC pressure
        private byte[] _readBuffer;
        private ByteBuffer _byteBuffer;

        public FlatBufferSensorArena()
        {
            // Create or open the named memory mapped file
            _mmf = MemoryMappedFile.CreateOrOpen(
                MapName,
                MapCapacity,
                MemoryMappedFileAccess.ReadWrite);

            // Create a view for the entire map
            _accessor = _mmf.CreateViewAccessor(0, MapCapacity, MemoryMappedFileAccess.ReadWrite);

            // Acquire the raw pointer
            byte* ptr = null;
            _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
            _basePtr = ptr;

            // Map the header struct to the beginning of the memory
            _headerPtr = (FlatBufferArenaHeader*)_basePtr;

            // Create or open the signal event (AutoReset)
            _signal = new EventWaitHandle(false, EventResetMode.AutoReset, SignalName);

            // Pre-allocate read buffer (8KB max, reused across reads)
            _readBuffer = new byte[MapCapacity - DataOffset];
            _byteBuffer = new ByteBuffer(_readBuffer);

            InitializeIfNeeded();
        }

        private void InitializeIfNeeded()
        {
            // Check if already initialized by inspecting the MagicNumber
            if (_headerPtr->MagicNumber != MagicSignature)
            {
                _headerPtr->SequenceNumber = 0;
                _headerPtr->ProtocolVersion = CurrentProtocolVersion;
                _headerPtr->BufferSize = 0;
                _headerPtr->TimestampFrequency = System.Diagnostics.Stopwatch.Frequency;
                // Set magic number last to indicate valid header
                _headerPtr->MagicNumber = MagicSignature;
            }
            else
            {
                // Arena already initialized - verify protocol version
                if (_headerPtr->ProtocolVersion != CurrentProtocolVersion)
                {
                    Console.WriteLine($"[SensorArena] WARNING: Protocol version mismatch! Expected {CurrentProtocolVersion}, found {_headerPtr->ProtocolVersion}");
                    Console.WriteLine($"[SensorArena] FlatBuffer schema may be incompatible. Regenerate bindings and restart processes.");
                }
            }
        }

        /// <summary>
        /// Reads the current sequence number (for change detection).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long GetSequenceNumber()
        {
            return Volatile.Read(ref _headerPtr->SequenceNumber);
        }

        /// <summary>
        /// Gets the QueryPerformanceCounter frequency for timestamp conversions.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long GetTimestampFrequency()
        {
            return Volatile.Read(ref _headerPtr->TimestampFrequency);
        }

        /// <summary>
        /// Reads the latest SensorFrame from shared memory using pre-allocated buffer.
        /// Returns null if buffer is invalid or empty.
        /// Near-zero allocation: reuses internal byte[] and ByteBuffer across calls.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public SensorFrame? ReadLatestSensorFrame()
        {
            int bufferSize = Volatile.Read(ref _headerPtr->BufferSize);
            if (bufferSize <= 0 || bufferSize > (MapCapacity - DataOffset))
            {
                return null; // Invalid buffer size
            }

            // Copy from shared memory into pre-allocated buffer (no heap allocation)
            Marshal.Copy((IntPtr)(_basePtr + DataOffset), _readBuffer, 0, bufferSize);

            // Wrap pre-allocated buffer (ByteBuffer constructor is lightweight)
            _byteBuffer = new ByteBuffer(_readBuffer);

            // Deserialize FlatBuffer (zero-copy access to buffer internals)
            return SensorFrame.GetRootAsSensorFrame(_byteBuffer);
        }

        /// <summary>
        /// Writes a SensorFrame to shared memory.
        /// Used by producer (Python detector) - exposed for C# testing.
        /// </summary>
        public void WriteSensorFrame(FlatBufferBuilder builder)
        {
            // Get serialized buffer
            byte[] buffer = builder.SizedByteArray();
            int bufferSize = buffer.Length;

            if (bufferSize > (MapCapacity - DataOffset))
            {
                throw new ArgumentException($"FlatBuffer too large: {bufferSize} bytes (max {MapCapacity - DataOffset})");
            }

            // Write buffer to shared memory
            Marshal.Copy(buffer, 0, (IntPtr)(_basePtr + DataOffset), bufferSize);

            // Update header atomically
            Volatile.Write(ref _headerPtr->BufferSize, bufferSize);
            Interlocked.Increment(ref _headerPtr->SequenceNumber);

            // Signal waiting consumers
            _signal.Set();
        }

        /// <summary>
        /// Blocks until Python signals new data is available.
        /// </summary>
        /// <param name="timeoutMs">Maximum time to wait in milliseconds.</param>
        /// <returns>True if signaled, false if timed out.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool WaitForResult(int timeoutMs)
        {
            return _signal.WaitOne(timeoutMs);
        }

        /// <summary>
        /// Blocks until Python signals new data or cancellation is requested.
        /// </summary>
        /// <param name="timeoutMs">Maximum time to wait in milliseconds.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>True if signaled, false if timed out or cancelled.</returns>
        public bool WaitForResult(int timeoutMs, CancellationToken ct)
        {
            if (ct == default)
            {
                return _signal.WaitOne(timeoutMs);
            }

            int result = WaitHandle.WaitAny(new[] { _signal, ct.WaitHandle }, timeoutMs);

            // WaitAny returns index of signaled handle, or WaitHandle.WaitTimeout (258)
            // Index 0 = _signal, Index 1 = cancellation
            return result == 0;
        }

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
                    _signal?.Dispose();
                    _accessor?.Dispose();
                    _mmf?.Dispose();
                }

                _disposed = true;
            }
        }

        ~FlatBufferSensorArena()
        {
            Dispose(false);
        }
    }
}
