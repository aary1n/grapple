using System.Runtime.InteropServices;

namespace Grapple.Core
{
    /// <summary>
    /// The Hand Data Contract: A 56-byte struct representing hand tracking results.
    /// Written by Python detector, read by C# consumer.
    /// Pure data carrier. No methods, no references.
    /// </summary>
    /// <remarks>
    /// Python struct format: '&lt;dddddifq' (little-endian: 5×double, int, float, long)
    /// Includes velocity (vx, vy) for cursor extrapolation between inference frames.
    /// </remarks>
    [StructLayout(LayoutKind.Explicit, Size = 56)]
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
        /// Velocity X component (units/sec in normalized space).
        /// Used for cursor extrapolation between inference frames.
        /// </summary>
        [FieldOffset(24)]
        public readonly double VX;

        /// <summary>
        /// Velocity Y component (units/sec in normalized space).
        /// Used for cursor extrapolation between inference frames.
        /// </summary>
        [FieldOffset(32)]
        public readonly double VY;

        /// <summary>
        /// Gesture identifier: 0=None, 1=IndexPoint, 2=Pinch.
        /// </summary>
        [FieldOffset(40)]
        public readonly int GestureId;

        /// <summary>
        /// MediaPipe detection confidence (0.0–1.0).
        /// </summary>
        [FieldOffset(44)]
        public readonly float Confidence;

        /// <summary>
        /// Original frame QPC timestamp for round-trip latency calculation.
        /// </summary>
        [FieldOffset(48)]
        public readonly long Timestamp;

        public HandState(double x, double y, double z, double vx, double vy, int gestureId, float confidence, long timestamp)
        {
            X = x;
            Y = y;
            Z = z;
            VX = vx;
            VY = vy;
            GestureId = gestureId;
            Confidence = confidence;
            Timestamp = timestamp;
        }
    }
}
