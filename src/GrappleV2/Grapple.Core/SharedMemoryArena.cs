using System;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Threading;

namespace Grapple.Core
{
    [StructLayout(LayoutKind.Sequential)]
    public struct ArenaHeader
    {
        public ulong MagicNumber;       // 8 bytes
        public int SlotCount;           // 4 bytes
        public int SlotSize;            // 4 bytes
        public long WriteHeadIndex;     // 8 bytes
        // Padding might occur here depending on alignment, but explicit fields sum to 24.
        // We reserved 1KB, so we have plenty of space.
    }

    public unsafe class SharedMemoryArena : IDisposable
    {
        // Configuration
        private const string MapName = "Local\\GrappleMap";
        private const long MapCapacity = 256 * 1024 * 1024; // 256 MB
        private const int HeaderReservedSize = 1024;
        private const int FirstSlotOffset = 1024; // Aligned to 64 bytes (1024 is multiple of 64)
        private const int TargetSlotSize = 8 * 1024 * 1024; // 8 MB
        
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
            // "Global\" prefix makes it visible across all sessions.
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
            if (payloadSize > _headerPtr->SlotSize)
            {
                throw new ArgumentOutOfRangeException(nameof(payloadSize), "Payload too large for slot.");
            }

            // Monotonic counter increment
            long nextIndex = Interlocked.Increment(ref _headerPtr->WriteHeadIndex) - 1;

            // Math Safety: Handle wrap-around and ensure positive index
            // Using ulong cast handles the modulo logic correctly for ring buffer behavior
            int bufferId = (int)((ulong)nextIndex % (ulong)_headerPtr->SlotCount);

            // Create the handle (struct copy, no heap alloc)
            return new GraphPacket(bufferId, timestamp, payloadSize);
        }

        /// <summary>
        /// Access: Returns a Span covering the entire slot for the given bufferId.
        /// Strict Zero-Alloc.
        /// </summary>
        public Span<byte> GetSpan(int bufferId)
        {
            // Calculate absolute start address
            // Math: start = basePtr + headerOffset (which is FirstSlotOffset effectively for the data start) + (bufferId * slotSize)
            // Note: Header reserves 1024 bytes. First slot starts at 1024.
            
            // Validate BufferId (Safety check)
            if ((uint)bufferId >= (uint)_headerPtr->SlotCount)
            {
                 throw new IndexOutOfRangeException();
            }

            long byteOffset = FirstSlotOffset + ((long)bufferId * _headerPtr->SlotSize);
            
            // Create Span from pointer
            return new Span<byte>(_basePtr + byteOffset, _headerPtr->SlotSize);
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

