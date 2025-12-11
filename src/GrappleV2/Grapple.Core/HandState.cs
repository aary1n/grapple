using System.Runtime.InteropServices;

namespace Grapple.Core
{
    /// <summary>
    /// The Hand Data Contract: A 40-byte struct representing hand tracking results.
    /// Written by Python detector, read by C# consumer.
    /// Pure data carrier. No methods, no references.
    /// </summary>
    /// <remarks>
    /// Python struct format: '&lt;dddifq' (little-endian: 3×double, int, float, long)
    /// </remarks>
    [StructLayout(LayoutKind.Explicit, Size = 40)]
    public readonly struct HandState
    {
        /// <summary>
        /// Normalized X coordinate (0.0–1.0) of index finger tip.
        /// </summary>
        [FieldOffset(0)]
        public readonly double X;

        /// <summary>
        /// Normalized Y coordinate (0.0–1.0) of index finger tip.
        /// </summary>
        [FieldOffset(8)]
        public readonly double Y;

        /// <summary>
        /// Normalized Z coordinate (depth) of index finger tip.
        /// </summary>
        [FieldOffset(16)]
        public readonly double Z;

        /// <summary>
        /// Gesture identifier: 0=None, 1=IndexPoint, 2=Fist, 3=OpenPalm.
        /// </summary>
        [FieldOffset(24)]
        public readonly int GestureId;

        /// <summary>
        /// MediaPipe detection confidence (0.0–1.0).
        /// </summary>
        [FieldOffset(28)]
        public readonly float Confidence;

        /// <summary>
        /// Original frame QPC timestamp for round-trip latency calculation.
        /// </summary>
        [FieldOffset(32)]
        public readonly long Timestamp;

        public HandState(double x, double y, double z, int gestureId, float confidence, long timestamp)
        {
            X = x;
            Y = y;
            Z = z;
            GestureId = gestureId;
            Confidence = confidence;
            Timestamp = timestamp;
        }
    }
}

