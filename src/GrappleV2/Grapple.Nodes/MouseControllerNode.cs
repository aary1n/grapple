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

        // === TUNING PARAMETERS ===
        
        // Confidence threshold
        private const float MinConfidence = 0.5f;
        private const int WaitTimeoutMs = 100;

        // Sensitivity: simple linear multiplier (1.0 = 1:1 mapping)
        // Values > 1.0 = more sensitive, < 1.0 = less sensitive
        private const double Sensitivity = 1.3;  // Slight boost, but not crazy
        
        // Motion Interpolation (for smooth dragging)
        private const int InterpolationSteps = 4;

        // Telemetry
        private long _frameCount = 0;
        private bool _disposed = false;

        // Click state
        private bool _isLeftDown = false;

        // Motion tracking for interpolation
        private int _lastScreenX = 0;
        private int _lastScreenY = 0;
        private bool _hasLastPosition = false;

        // Safety clutch state
        private bool _isActive = false;
        private bool _wasToggleKeyDown = false;

        /// <summary>
        /// Creates a new mouse controller node with tuned filter parameters.
        /// </summary>
        public MouseControllerNode()
        {
            _arena = new HandResultArena();
            
            // HEAVILY SMOOTHED filter settings to eliminate jitter
            // minCutoff: Lower = smoother but laggier. 0.5 is quite smooth.
            // beta: Controls speed-adaptive smoothing. 0.007 is subtle.
            // dCutoff: Derivative smoothing. 1.0 is standard.
            double minCutoff = 0.4;   // Very smooth for slow movements
            double beta = 0.01;       // Slight speed adaptation
            double dCutoff = 1.0;
            
            _filterX = new OneEuroFilter(minCutoff, beta, dCutoff);
            _filterY = new OneEuroFilter(minCutoff, beta, dCutoff);
        }

        public ValueTask StartAsync(CancellationToken ct)
        {
            Task.Factory.StartNew(() => RunLoop(ct),
                ct,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);

            return ValueTask.CompletedTask;
        }

        private void RunLoop(CancellationToken ct)
        {
            Console.WriteLine("[Mouse] Controller started...");
            Console.WriteLine($"[Mouse] Screen: {Win32Input.ScreenWidth}x{Win32Input.ScreenHeight}");
            Console.WriteLine($"[Mouse] Sensitivity: {Sensitivity:F1}x");
            Console.WriteLine("[Mouse] *** PAUSED *** (Press F9 to activate)");
            Console.Beep(440, 200);

            long lastSeq = -1;
            int noHandFrames = 0;

            try
            {
                while (!ct.IsCancellationRequested)
                {
                    // 0. Check safety toggle (F9)
                    bool isToggleKeyDown = Win32Input.IsKeyDown(Win32Input.VK_F9);
                    
                    if (isToggleKeyDown && !_wasToggleKeyDown)
                    {
                        _isActive = !_isActive;
                        
                        if (!_isActive && _isLeftDown)
                        {
                            Win32Input.LeftUp();
                            _isLeftDown = false;
                            Console.WriteLine("[Mouse] Left Up (paused - safety release)");
                        }
                        
                        _hasLastPosition = false;
                        _filterX.Reset();
                        _filterY.Reset();
                        
                        if (_isActive)
                        {
                            Console.Beep(880, 100);
                            Console.WriteLine("[Mouse] *** ACTIVE *** (F9 to pause)");
                        }
                        else
                        {
                            Console.Beep(440, 200);
                            Console.WriteLine("[Mouse] *** PAUSED *** (F9 to activate)");
                        }
                    }
                    _wasToggleKeyDown = isToggleKeyDown;

                    // 1. Wait for signal
                    if (!_arena.WaitForResult(WaitTimeoutMs, ct))
                    {
                        continue;
                    }

                    // 2. Read state
                    HandState state = _arena.ReadLatest();
                    long seq = _arena.GetSequenceNumber();

                    // 3. Skip if stale
                    if (seq == lastSeq)
                    {
                        continue;
                    }
                    lastSeq = seq;

                    // GATE: Skip if paused
                    if (!_isActive)
                    {
                        continue;
                    }

                    // 4. Handle no hand or low confidence
                    if (state.GestureId == 0 || state.Confidence < MinConfidence)
                    {
                        noHandFrames++;
                        
                        if (_isLeftDown)
                        {
                            Win32Input.LeftUp();
                            _isLeftDown = false;
                            Console.WriteLine("[Mouse] Left Up (hand lost - safety release)");
                        }
                        
                        if (noHandFrames % 30 == 0)
                        {
                            Console.WriteLine($"[Mouse] No hand detected (skipped {noHandFrames} frames)");
                        }
                        
                        if (noHandFrames > 20)
                        {
                            _filterX.Reset();
                            _filterY.Reset();
                            _hasLastPosition = false;
                        }
                        continue;
                    }

                    noHandFrames = 0;

                    // 5. Convert timestamp to seconds for filter
                    double timestampSec = state.Timestamp / (double)Stopwatch.Frequency;

                    // 6. Apply smoothing filter FIRST (this is critical for stability)
                    double smoothX = _filterX.Filter(state.X, timestampSec);
                    double smoothY = _filterY.Filter(state.Y, timestampSec);

                    // 7. MIRROR X-axis (webcam is mirrored)
                    smoothX = 1.0 - smoothX;
                    // Keep Y as-is (MediaPipe Y=0 is top, same as screen)
                    // Actually, test showed we need to invert Y too
                    smoothY = 1.0 - smoothY;

                    // 8. Apply sensitivity (center-anchored for natural feel)
                    double centerX = 0.5;
                    double centerY = 0.5;
                    double scaledX = centerX + (smoothX - centerX) * Sensitivity;
                    double scaledY = centerY + (smoothY - centerY) * Sensitivity;

                    // 9. Map to screen coordinates
                    int screenX = (int)(scaledX * Win32Input.ScreenWidth);
                    int screenY = (int)(scaledY * Win32Input.ScreenHeight);

                    // 10. Clamp to screen bounds
                    screenX = Math.Clamp(screenX, 0, Win32Input.ScreenWidth - 1);
                    screenY = Math.Clamp(screenY, 0, Win32Input.ScreenHeight - 1);

                    // 11. Motion Interpolation during drag
                    if (_isLeftDown && _hasLastPosition)
                    {
                        int deltaX = screenX - _lastScreenX;
                        int deltaY = screenY - _lastScreenY;
                        double distance = Math.Sqrt(deltaX * deltaX + deltaY * deltaY);
                        
                        if (distance > 15)
                        {
                            int steps = Math.Min(InterpolationSteps, (int)(distance / 8));
                            for (int i = 1; i < steps; i++)
                            {
                                double t = (double)i / steps;
                                int interpX = _lastScreenX + (int)(deltaX * t);
                                int interpY = _lastScreenY + (int)(deltaY * t);
                                Win32Input.MoveMouse(interpX, interpY);
                            }
                        }
                    }

                    // 12. Move cursor to final position
                    bool success = Win32Input.MoveMouse(screenX, screenY);
                    
                    _lastScreenX = screenX;
                    _lastScreenY = screenY;
                    _hasLastPosition = true;
                    
                    if (_frameCount == 0 && success)
                    {
                        Console.WriteLine($"[Mouse] First cursor move to ({screenX}, {screenY})");
                    }

                    // 13. Handle click state machine
                    if (state.GestureId == 2 && !_isLeftDown)
                    {
                        Win32Input.LeftDown();
                        _isLeftDown = true;
                        Console.WriteLine($"[Mouse] Left Down at ({screenX}, {screenY})");
                    }
                    else if (state.GestureId != 2 && _isLeftDown)
                    {
                        Win32Input.LeftUp();
                        _isLeftDown = false;
                        Console.WriteLine($"[Mouse] Left Up at ({screenX}, {screenY})");
                    }

                    // 14. Telemetry
                    _frameCount++;
                    if (_frameCount % 60 == 0)
                    {
                        string clickState = _isLeftDown ? "DOWN" : "UP";
                        Console.WriteLine($"[Mouse] Frames: {_frameCount} | Pos: ({screenX}, {screenY}) | " +
                                        $"Gesture: {state.GestureId} | Click: {clickState}");
                    }
                }
            }
            catch (OperationCanceledException)
            {
            }
            finally
            {
                if (_isLeftDown)
                {
                    Win32Input.LeftUp();
                    _isLeftDown = false;
                    Console.WriteLine("[Mouse] Left Up (shutdown - safety release)");
                }
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
