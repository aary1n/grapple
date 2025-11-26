using System.Runtime.InteropServices;

namespace Grapple.Core
{
    /// <summary>
    /// The Handle: A 16-byte struct representing a frame in the shared memory arena.
    /// Pure data carrier. No methods, no references.
    /// </summary>
    [StructLayout(LayoutKind.Explicit, Size = 16)]
    public readonly struct GraphPacket
    {
        /// <summary>
        /// The logical index of the slot in the arena.
        /// </summary>
        [FieldOffset(0)]
        public readonly int BufferId;

        /// <summary>
        /// Actual bytes used in the frame.
        /// Placed at offset 4 to optimize packing (4+4+8 = 16).
        /// </summary>
        [FieldOffset(4)]
        public readonly int PayloadSize;

        /// <summary>
        /// QPC ticks timestamp.
        /// </summary>
        [FieldOffset(8)]
        public readonly long Timestamp;

        public GraphPacket(int bufferId, long timestamp, int payloadSize)
        {
            BufferId = bufferId;
            Timestamp = timestamp;
            PayloadSize = payloadSize;
        }
    }
}

