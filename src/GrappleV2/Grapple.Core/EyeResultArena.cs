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
    /// Shared memory arena for eye tracking results (Tobii/Gazepoint).
    /// Phase 2 placeholder: no active producer yet.
    /// Follows same pattern as FlatBufferSensorArena but dedicated to eye data
    /// per ADR-003 (separate arenas per sensor type for decoupled update rates).
    /// Eye tracking updates at ~90Hz vs hand tracking at ~15Hz.
    /// </summary>
    public unsafe class EyeResultArena : IDisposable
    {
        private const string MapName = "Local\\GrappleEyeResults";
        private const string SignalName = "Local\\GrappleEyeSignal";
        private const long MapCapacity = 4096; // 4KB
        private const int DataOffset = 64;

        // Reuses FlatBuffer arena header format
        private const ulong MagicSignature = 0x4C505247; // "GRPL"
        private const int CurrentProtocolVersion = 2;

        private readonly MemoryMappedFile _mmf;
        private readonly MemoryMappedViewAccessor _accessor;
        private readonly byte* _basePtr;
        private readonly FlatBufferArenaHeader* _headerPtr;
        private readonly EventWaitHandle _signal;
        private bool _disposed;

        private byte[] _readBuffer;
        private ByteBuffer _byteBuffer;

        public EyeResultArena()
        {
            _mmf = MemoryMappedFile.CreateOrOpen(
                MapName,
                MapCapacity,
                MemoryMappedFileAccess.ReadWrite);

            _accessor = _mmf.CreateViewAccessor(0, MapCapacity, MemoryMappedFileAccess.ReadWrite);

            byte* ptr = null;
            _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
            _basePtr = ptr;

            _headerPtr = (FlatBufferArenaHeader*)_basePtr;

            _signal = new EventWaitHandle(false, EventResetMode.AutoReset, SignalName);

            _readBuffer = new byte[MapCapacity - DataOffset];
            _byteBuffer = new ByteBuffer(_readBuffer);

            InitializeIfNeeded();
        }

        private void InitializeIfNeeded()
        {
            if (_headerPtr->MagicNumber != MagicSignature)
            {
                _headerPtr->SequenceNumber = 0;
                _headerPtr->ProtocolVersion = CurrentProtocolVersion;
                _headerPtr->BufferSize = 0;
                _headerPtr->TimestampFrequency = System.Diagnostics.Stopwatch.Frequency;
                _headerPtr->MagicNumber = MagicSignature;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long GetSequenceNumber()
        {
            return Volatile.Read(ref _headerPtr->SequenceNumber);
        }

        /// <summary>
        /// Reads the latest EyeState from shared memory.
        /// Returns null if no eye data has been written.
        /// </summary>
        public EyeState? ReadLatestEyeState()
        {
            int bufferSize = Volatile.Read(ref _headerPtr->BufferSize);
            if (bufferSize <= 0 || bufferSize > (MapCapacity - DataOffset))
            {
                return null;
            }

            Marshal.Copy((IntPtr)(_basePtr + DataOffset), _readBuffer, 0, bufferSize);
            _byteBuffer = new ByteBuffer(_readBuffer);

            return EyeState.GetRootAsEyeState(_byteBuffer);
        }

        /// <summary>
        /// Writes an EyeState to shared memory.
        /// For future eye tracker producers.
        /// </summary>
        public void WriteEyeState(FlatBufferBuilder builder)
        {
            byte[] buffer = builder.SizedByteArray();
            int bufferSize = buffer.Length;

            if (bufferSize > (MapCapacity - DataOffset))
            {
                throw new ArgumentException($"FlatBuffer too large: {bufferSize} bytes (max {MapCapacity - DataOffset})");
            }

            Marshal.Copy(buffer, 0, (IntPtr)(_basePtr + DataOffset), bufferSize);

            Volatile.Write(ref _headerPtr->BufferSize, bufferSize);
            Interlocked.Increment(ref _headerPtr->SequenceNumber);

            _signal.Set();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool WaitForResult(int timeoutMs)
        {
            return _signal.WaitOne(timeoutMs);
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

        ~EyeResultArena()
        {
            Dispose(false);
        }
    }
}
