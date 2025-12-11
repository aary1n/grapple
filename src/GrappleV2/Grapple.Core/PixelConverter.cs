using System;
using System.Runtime.CompilerServices;

namespace Grapple.Core
{
    /// <summary>
    /// High-performance pixel format conversion utilities.
    /// Used to convert webcam BGR frames to RGB for MediaPipe.
    /// </summary>
    public static class PixelConverter
    {
        /// <summary>
        /// Swaps Blue and Red channels from BGR to RGB format.
        /// BGR (webcam/DirectShow) → RGB (MediaPipe)
        /// </summary>
        /// <param name="input">Source BGR24 pixel data.</param>
        /// <param name="output">Destination RGB24 buffer (can be same as input for in-place).</param>
        /// <remarks>
        /// Performance target: &lt; 2ms for 1920×1080 frame (6.2MB).
        /// Uses unsafe pointer arithmetic for maximum throughput.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void BgrToRgb(ReadOnlySpan<byte> input, Span<byte> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            if (input.Length % 3 != 0)
            {
                throw new ArgumentException("Buffer length must be divisible by 3 (BGR24 format).");
            }

            int pixelCount = input.Length / 3;

            fixed (byte* pInput = input)
            fixed (byte* pOutput = output)
            {
                byte* src = pInput;
                byte* dst = pOutput;

                // Process 4 pixels at a time (12 bytes) for better cache utilization
                int unrolledCount = pixelCount / 4;
                int remainder = pixelCount % 4;

                for (int i = 0; i < unrolledCount; i++)
                {
                    // Pixel 0: BGR -> RGB
                    byte b0 = src[0];
                    byte g0 = src[1];
                    byte r0 = src[2];
                    dst[0] = r0;
                    dst[1] = g0;
                    dst[2] = b0;

                    // Pixel 1: BGR -> RGB
                    byte b1 = src[3];
                    byte g1 = src[4];
                    byte r1 = src[5];
                    dst[3] = r1;
                    dst[4] = g1;
                    dst[5] = b1;

                    // Pixel 2: BGR -> RGB
                    byte b2 = src[6];
                    byte g2 = src[7];
                    byte r2 = src[8];
                    dst[6] = r2;
                    dst[7] = g2;
                    dst[8] = b2;

                    // Pixel 3: BGR -> RGB
                    byte b3 = src[9];
                    byte g3 = src[10];
                    byte r3 = src[11];
                    dst[9] = r3;
                    dst[10] = g3;
                    dst[11] = b3;

                    src += 12;
                    dst += 12;
                }

                // Handle remaining pixels
                for (int i = 0; i < remainder; i++)
                {
                    byte b = src[0];
                    byte g = src[1];
                    byte r = src[2];
                    dst[0] = r;
                    dst[1] = g;
                    dst[2] = b;

                    src += 3;
                    dst += 3;
                }
            }
        }

        /// <summary>
        /// Swaps Blue and Red channels in-place.
        /// </summary>
        /// <param name="buffer">BGR24 pixel data to convert to RGB24 in-place.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void BgrToRgbInPlace(Span<byte> buffer)
        {
            BgrToRgb(buffer, buffer);
        }
    }
}

