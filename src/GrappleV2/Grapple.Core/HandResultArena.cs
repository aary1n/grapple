using System;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace Grapple.Core
{
    /// <summary>
    /// Header structure for the hand results shared memory region.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct HandResultHeader
    {
        public ulong MagicNumber;      // Offset 0,  8 bytes
        public long SequenceNumber;    // Offset 8,  8 bytes (monotonic counter)
        // Total: 16 bytes
    }

    /// <summary>
    /// Shared memory reader for hand tracking results from Python.
    /// Python writes HandState structs, C# reads them.
    /// </summary>
    public unsafe class HandResultArena : IDisposable
    {
        // Configuration
        private const string MapName = "Local\\GrappleHandResults";
        private const string SignalName = "Local\\GrappleHandSignal";
        private const long MapCapacity = 4096; // 4KB
        private const int DataOffset = 64; // HandState starts here (aligned)

        // "HANDGRPC" in hex (little-endian)
        private const ulong MagicSignature = 0x48414E4447525043;

        private readonly MemoryMappedFile _mmf;
        private readonly MemoryMappedViewAccessor _accessor;
        private readonly byte* _basePtr;
        private readonly HandResultHeader* _headerPtr;
        private readonly EventWaitHandle _signal;
        private bool _disposed;

        public HandResultArena()
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
            _headerPtr = (HandResultHeader*)_basePtr;

            // Create or open the signal event (AutoReset)
            _signal = new EventWaitHandle(false, EventResetMode.AutoReset, SignalName);

            InitializeIfNeeded();
        }

        private void InitializeIfNeeded()
        {
            // Check if already initialized by inspecting the MagicNumber
            // Idempotent: Python may also initialize, both write same values
            if (_headerPtr->MagicNumber != MagicSignature)
            {
                _headerPtr->SequenceNumber = 0;
                // Set magic number last to indicate valid header
                _headerPtr->MagicNumber = MagicSignature;
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
        /// Reads the latest HandState from shared memory.
        /// Zero allocation - returns struct by value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public HandState ReadLatest()
        {
            // Read HandState from data offset
            HandState* statePtr = (HandState*)(_basePtr + DataOffset);
            return *statePtr;
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

        ~HandResultArena()
        {
            Dispose(false);
        }
    }
}

