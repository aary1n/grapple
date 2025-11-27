using System;
using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace Grapple.Core
{
    [StructLayout(LayoutKind.Sequential)]
    public struct ArenaHeader
    {
        public ulong MagicNumber;         // Offset 0,  8 bytes
        public int SlotCount;             // Offset 8,  4 bytes
        public int SlotSize;              // Offset 12, 4 bytes
        public long WriteHeadIndex;       // Offset 16, 8 bytes
        public int PublishedBufferId;     // Offset 24, 4 bytes  (for IPC consumers)
        public int _padding;              // Offset 28, 4 bytes  (alignment)
        public long TimestampFrequency;   // Offset 32, 8 bytes  (Stopwatch.Frequency for Python)
        // Total: 40 bytes (8-byte aligned)
    }

    public unsafe class SharedMemoryArena : IDisposable
    {
        // Configuration
        private const string MapName = "Local\\GrappleMap";
        private const long MapCapacity = 256 * 1024 * 1024; // 256 MB
        private const int HeaderReservedSize = 1024;
        private const int FirstSlotOffset = 1024; // Aligned to 64 bytes (1024 is multiple of 64)
        private const int TargetSlotSize = 8 * 1024 * 1024; // 8 MB
        private const int MetadataSize = 64; // Reserved space for in-band metadata
        
        // "GRAPPLE1" in hex - used to verify memory initialization
        private const ulong MagicSignature = 0x31454C5050415247; 

        private readonly MemoryMappedFile _mmf;
        private readonly MemoryMappedViewAccessor _accessor;
        private readonly byte* _basePtr;
        private readonly ArenaHeader* _headerPtr;
        private bool _disposed;

        public SharedMemoryArena()
        {
            // Create or open the named memory mapped file.
            // "Local\" prefix makes it visible in current session.
            _mmf = MemoryMappedFile.CreateOrOpen(
                MapName, 
                MapCapacity, 
                MemoryMappedFileAccess.ReadWrite);

            // Create a view for the entire map
            _accessor = _mmf.CreateViewAccessor(0, MapCapacity, MemoryMappedFileAccess.ReadWrite);

            // Acquire the raw pointer. 
            // SafeMemoryMappedViewHandle pins the memory (or rather, holds the handle that keeps the mapping valid).
            byte* ptr = null;
            _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
            _basePtr = ptr;
            
            // Map the header struct to the beginning of the memory
            _headerPtr = (ArenaHeader*)_basePtr;

            InitializeIfNeeded();
        }

        private void InitializeIfNeeded()
        {
            // Check if already initialized by inspecting the MagicNumber
            // We use a simple check. For a robust multi-process race-free init, 
            // we would need a named Mutex, but for this primitive we assume one writer/initializer 
            // or that the race is benign (re-writing same values).
            if (_headerPtr->MagicNumber != MagicSignature)
            {
                // Calculate geometry
                int availableMemory = (int)(MapCapacity - FirstSlotOffset);
                int slotCount = availableMemory / TargetSlotSize; // ~30 slots

                // Write header fields
                _headerPtr->SlotCount = slotCount;
                _headerPtr->SlotSize = TargetSlotSize;
                _headerPtr->WriteHeadIndex = 0;
                _headerPtr->PublishedBufferId = -1;  // No frame published yet
                _headerPtr->_padding = 0;
                _headerPtr->TimestampFrequency = Stopwatch.Frequency;

                // Set magic number last to indicate valid header
                // Use Volatile Write to ensure ordering if needed, but here simple assignment suffices
                // as we are likely the only process starting up right now.
                _headerPtr->MagicNumber = MagicSignature;
            }
        }

        /// <summary>
        /// Allocator Logic: Acquires the next slot in the ring buffer.
        /// Zero allocation.
        /// </summary>
        public GraphPacket AcquireNextSlot(long timestamp, int payloadSize)
        {
            if (payloadSize > _headerPtr->SlotSize - MetadataSize)
            {
                throw new ArgumentOutOfRangeException(nameof(payloadSize), "Payload too large for slot (metadata space included).");
            }

            // Monotonic counter increment
            long nextIndex = Interlocked.Increment(ref _headerPtr->WriteHeadIndex) - 1;

            // Math Safety: Handle wrap-around and ensure positive index
            // Using ulong cast handles the modulo logic correctly for ring buffer behavior
            int bufferId = (int)((ulong)nextIndex % (ulong)_headerPtr->SlotCount);

            // Write metadata to the slot
            WriteFrameMetadata(bufferId, timestamp, payloadSize);

            // Create the handle (struct copy, no heap alloc)
            return new GraphPacket(bufferId, timestamp, payloadSize);
        }

        /// <summary>
        /// Access: Returns a Span covering the PAYLOAD of the slot for the given bufferId.
        /// Skips the first 64 bytes (Metadata).
        /// Strict Zero-Alloc.
        /// </summary>
        public Span<byte> GetSpan(int bufferId)
        {
            // Validate BufferId (Safety check)
            if ((uint)bufferId >= (uint)_headerPtr->SlotCount)
            {
                 throw new IndexOutOfRangeException();
            }

            // Calculate absolute start address
            // Math: start = basePtr + headerOffset + (bufferId * slotSize) + MetadataSize
            // MetadataSize is 64 bytes.
            long byteOffset = FirstSlotOffset + ((long)bufferId * _headerPtr->SlotSize) + MetadataSize;
            
            // Create Span from pointer
            return new Span<byte>(_basePtr + byteOffset, _headerPtr->SlotSize - MetadataSize);
        }

        /// <summary>
        /// Writes metadata to the reserved 64 bytes at the start of the slot.
        /// </summary>
        public void WriteFrameMetadata(int bufferId, long timestamp, int payloadSize)
        {
             if ((uint)bufferId >= (uint)_headerPtr->SlotCount)
            {
                 throw new IndexOutOfRangeException();
            }

            long byteOffset = FirstSlotOffset + ((long)bufferId * _headerPtr->SlotSize);
            byte* slotStart = _basePtr + byteOffset;

            // Layout:
            // Bytes 0-7: long Timestamp
            // Bytes 8-11: int PayloadSize
            // Bytes 12-63: Padding
            
            *(long*)(slotStart) = timestamp;
            *(int*)(slotStart + 8) = payloadSize;
        }

        /// <summary>
        /// Reads metadata from the reserved 64 bytes at the start of the slot.
        /// </summary>
        public GraphPacket ReadGraphPacket(int bufferId)
        {
            if ((uint)bufferId >= (uint)_headerPtr->SlotCount)
            {
                 throw new IndexOutOfRangeException();
            }

            long byteOffset = FirstSlotOffset + ((long)bufferId * _headerPtr->SlotSize);
            byte* slotStart = _basePtr + byteOffset;

            long timestamp = *(long*)(slotStart);
            int payloadSize = *(int*)(slotStart + 8);

            return new GraphPacket(bufferId, timestamp, payloadSize);
        }

        /// <summary>
        /// Updates the header to indicate which buffer contains the latest frame.
        /// Called by producer after writing frame data.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void UpdatePublishedBuffer(int bufferId)
        {
            Volatile.Write(ref _headerPtr->PublishedBufferId, bufferId);
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
}
