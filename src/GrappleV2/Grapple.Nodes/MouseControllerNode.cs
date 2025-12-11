using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Grapple.Core;

namespace Grapple.Nodes
{
    /// <summary>
    /// Reads hand tracking data from HandResultArena and moves the Windows cursor.
    /// Uses 1€ Filter for smooth, responsive cursor movement.
    /// </summary>
    public class MouseControllerNode : IGraphNode, IDisposable
    {
        private readonly HandResultArena _arena;
        private readonly OneEuroFilter _filterX;
        private readonly OneEuroFilter _filterY;

        // Configuration
        private const float MinConfidence = 0.5f;   // Skip if detection uncertain
        private const int WaitTimeoutMs = 100;      // Allow cancellation check

        // Telemetry
        private long _frameCount = 0;
        private bool _disposed = false;

        /// <summary>
        /// Creates a new mouse controller node.
        /// </summary>
        /// <param name="minCutoff">Minimum cutoff frequency. Lower = smoother but more lag.</param>
        /// <param name="beta">Speed coefficient. Higher = more responsive to fast movements.</param>
        public MouseControllerNode(double minCutoff = 1.0, double beta = 0.007)
        {
            _arena = new HandResultArena();
            _filterX = new OneEuroFilter(minCutoff, beta, dCutoff: 1.0);
            _filterY = new OneEuroFilter(minCutoff, beta, dCutoff: 1.0);
        }

        public ValueTask StartAsync(CancellationToken ct)
        {
            // Start the loop on a dedicated thread
            Task.Factory.StartNew(() => RunLoop(ct),
                ct,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);

            // Return completed ValueTask immediately (non-blocking)
            return ValueTask.CompletedTask;
        }

        private void RunLoop(CancellationToken ct)
        {
            Console.WriteLine("[Mouse] Controller started...");
            Console.WriteLine($"[Mouse] Screen: {Win32Input.ScreenWidth}x{Win32Input.ScreenHeight}");

            long lastSeq = -1;
            int noHandFrames = 0;

            try
            {
                while (!ct.IsCancellationRequested)
                {
                    // 1. Wait for signal (with timeout for cancellation responsiveness)
                    if (!_arena.WaitForResult(WaitTimeoutMs, ct))
                    {
                        continue;
                    }

                    // 2. Read state
                    HandState state = _arena.ReadLatest();
                    long seq = _arena.GetSequenceNumber();

                    // 3. Skip if stale (same sequence = already processed)
                    if (seq == lastSeq)
                    {
                        continue;
                    }
                    lastSeq = seq;

                    // 4. Skip if no hand or low confidence
                    if (state.GestureId == 0 || state.Confidence < MinConfidence)
                    {
                        noHandFrames++;
                        
                        // Reset filters after prolonged tracking loss (1 second @ ~20fps)
                        if (noHandFrames > 20)
                        {
                            _filterX.Reset();
                            _filterY.Reset();
                            noHandFrames = 0;
                        }
                        continue;
                    }

                    noHandFrames = 0;

                    // 5. Convert timestamp to seconds for filter
                    double timestampSec = state.Timestamp / (double)Stopwatch.Frequency;

                    // 6. Apply smoothing
                    double smoothX = _filterX.Filter(state.X, timestampSec);
                    double smoothY = _filterY.Filter(state.Y, timestampSec);

                    // 7. MIRROR X-axis (webcam is mirrored!)
                    // MediaPipe X=0 is left side of frame = YOUR right hand
                    // To make it intuitive: move right → cursor moves right
                    smoothX = 1.0 - smoothX;

                    // 8. Map to screen coordinates
                    int screenX = (int)(smoothX * Win32Input.ScreenWidth);
                    int screenY = (int)(smoothY * Win32Input.ScreenHeight);

                    // 9. Clamp to screen bounds
                    screenX = Math.Clamp(screenX, 0, Win32Input.ScreenWidth - 1);
                    screenY = Math.Clamp(screenY, 0, Win32Input.ScreenHeight - 1);

                    // 10. Move cursor
                    try
                    {
                        Win32Input.MoveMouse(screenX, screenY);
                    }
                    catch (Exception ex)
                    {
                        // Log but don't crash
                        Console.WriteLine($"[Mouse] SendInput error: {ex.Message}");
                    }

                    // 11. Telemetry (every 60 frames)
                    _frameCount++;
                    if (_frameCount % 60 == 0)
                    {
                        Console.WriteLine($"[Mouse] Frames: {_frameCount} | Pos: ({screenX}, {screenY}) | Conf: {state.Confidence:F2}");
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Expected during graceful shutdown
            }

            Console.WriteLine($"[Mouse] Controller stopped. Total frames: {_frameCount}");
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _arena?.Dispose();
                _disposed = true;
            }
        }
    }
}

