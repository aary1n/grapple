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
    /// Features adaptive gain (mouse acceleration) and motion interpolation for drag operations.
    /// </summary>
    public class MouseControllerNode : IGraphNode, IDisposable
    {
        private readonly HandResultArena _arena;
        private readonly OneEuroFilter _filterX;
        private readonly OneEuroFilter _filterY;

        // === TUNING PARAMETERS ===
        
        // Confidence threshold
        private const float MinConfidence = 0.5f;   // Skip if detection uncertain
        private const int WaitTimeoutMs = 100;      // Allow cancellation check

        // Adaptive Gain (Mouse Acceleration)
        // Small movements = precise (1x), fast movements = amplified (up to MaxGain)
        private const double BaseGain = 1.2;          // Minimum gain (slightly above 1 for comfort)
        private const double MaxGain = 4.0;           // Maximum gain for fast movements
        private const double GainVelocityThreshold = 0.02;  // Velocity at which gain starts increasing (normalized units/sec)
        private const double GainVelocityMax = 0.15;        // Velocity at which gain reaches maximum
        
        // Motion Interpolation (for smooth dragging)
        private const int InterpolationSteps = 5;    // Number of intermediate points when dragging
        private const double InterpolationThreshold = 0.02; // Min distance to trigger interpolation (normalized)
        
        // Active Zone (use only central portion of camera frame for easier reach)
        // Values define the usable region: [ZoneMin, ZoneMax] maps to full screen
        private const double ZoneMinX = 0.20;  // Left 20% is dead zone
        private const double ZoneMaxX = 0.80;  // Right 20% is dead zone  
        private const double ZoneMinY = 0.15;  // Top 15% is dead zone
        private const double ZoneMaxY = 0.85;  // Bottom 15% is dead zone

        // Telemetry
        private long _frameCount = 0;
        private bool _disposed = false;

        // Click state
        private bool _isLeftDown = false;

        // Motion tracking for interpolation and velocity
        private double _lastRawX = 0.5;
        private double _lastRawY = 0.5;
        private double _lastTimestamp = 0;
        private int _lastScreenX = 0;
        private int _lastScreenY = 0;
        private bool _hasLastPosition = false;

        // Safety clutch state
        private bool _isActive = false;           // DEFAULT TO FALSE (safe on startup!)
        private bool _wasToggleKeyDown = false;   // For edge detection

        /// <summary>
        /// Creates a new mouse controller node.
        /// </summary>
        /// <param name="minCutoff">Minimum cutoff frequency. Lower = smoother but more lag.</param>
        /// <param name="beta">Speed coefficient. Higher = more responsive to fast movements.</param>
        public MouseControllerNode(double minCutoff = 0.8, double beta = 0.04)
        {
            _arena = new HandResultArena();
            // Tuned for more responsiveness: lower minCutoff, higher beta
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

        /// <summary>
        /// Calculates adaptive gain based on movement velocity.
        /// Slow movements get precision (low gain), fast movements get acceleration (high gain).
        /// </summary>
        private double CalculateAdaptiveGain(double velocity)
        {
            if (velocity <= GainVelocityThreshold)
                return BaseGain;
            
            if (velocity >= GainVelocityMax)
                return MaxGain;
            
            // Linear interpolation between base and max gain
            double t = (velocity - GainVelocityThreshold) / (GainVelocityMax - GainVelocityThreshold);
            return BaseGain + t * (MaxGain - BaseGain);
        }

        /// <summary>
        /// Maps a coordinate from the active zone to full [0, 1] range.
        /// </summary>
        private static double MapFromActiveZone(double value, double zoneMin, double zoneMax)
        {
            // Clamp to zone bounds first
            value = Math.Clamp(value, zoneMin, zoneMax);
            // Map [zoneMin, zoneMax] -> [0, 1]
            return (value - zoneMin) / (zoneMax - zoneMin);
        }

        private void RunLoop(CancellationToken ct)
        {
            Console.WriteLine("[Mouse] Controller started...");
            Console.WriteLine($"[Mouse] Screen: {Win32Input.ScreenWidth}x{Win32Input.ScreenHeight}");
            Console.WriteLine($"[Mouse] Adaptive Gain: {BaseGain:F1}x - {MaxGain:F1}x");
            Console.WriteLine($"[Mouse] Active Zone: X[{ZoneMinX:P0}-{ZoneMaxX:P0}] Y[{ZoneMinY:P0}-{ZoneMaxY:P0}]");
            Console.WriteLine("[Mouse] *** PAUSED *** (Press F9 to activate)");
            Console.Beep(440, 200);  // Low beep to indicate paused state

            long lastSeq = -1;
            int noHandFrames = 0;

            try
            {
                while (!ct.IsCancellationRequested)
                {
                    // 0. Check safety toggle (F9) - MUST be first for responsiveness
                    bool isToggleKeyDown = Win32Input.IsKeyDown(Win32Input.VK_F9);
                    
                    if (isToggleKeyDown && !_wasToggleKeyDown)
                    {
                        // Rising edge detected - toggle state
                        _isActive = !_isActive;
                        
                        // Force release any held click when pausing (prevent stuck drags)
                        if (!_isActive && _isLeftDown)
                        {
                            Win32Input.LeftUp();
                            _isLeftDown = false;
                            Console.WriteLine("[Mouse] Left Up (paused - safety release)");
                        }
                        
                        // Reset position tracking when toggling
                        _hasLastPosition = false;
                        
                        // Audio feedback (different tones for on/off)
                        if (_isActive)
                        {
                            Console.Beep(880, 100);  // High beep = ACTIVE
                            Console.WriteLine("[Mouse] *** ACTIVE *** (F9 to pause)");
                        }
                        else
                        {
                            Console.Beep(440, 200);  // Low beep = PAUSED
                            Console.WriteLine("[Mouse] *** PAUSED *** (F9 to activate)");
                        }
                    }
                    _wasToggleKeyDown = isToggleKeyDown;

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

                    // GATE: Skip all mouse actions if paused
                    if (!_isActive)
                    {
                        continue;
                    }

                    // 4. Handle no hand or low confidence
                    if (state.GestureId == 0 || state.Confidence < MinConfidence)
                    {
                        noHandFrames++;
                        
                        // SAFETY: Release mouse button if hand lost while clicking
                        if (_isLeftDown)
                        {
                            Win32Input.LeftUp();
                            _isLeftDown = false;
                            Console.WriteLine("[Mouse] Left Up (hand lost - safety release)");
                        }
                        
                        // Log skipped frames periodically
                        if (noHandFrames % 30 == 0)
                        {
                            Console.WriteLine($"[Mouse] No hand detected (skipped {noHandFrames} frames)");
                        }
                        
                        // Reset filters and position tracking after prolonged tracking loss
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

                    // 6. Calculate velocity for adaptive gain (before filtering)
                    double velocity = 0;
                    double dt = timestampSec - _lastTimestamp;
                    if (_hasLastPosition && dt > 0 && dt < 0.5) // Sanity check: < 500ms
                    {
                        double dx = state.X - _lastRawX;
                        double dy = state.Y - _lastRawY;
                        velocity = Math.Sqrt(dx * dx + dy * dy) / dt;
                    }
                    _lastRawX = state.X;
                    _lastRawY = state.Y;
                    _lastTimestamp = timestampSec;

                    // 7. Apply smoothing filter
                    double smoothX = _filterX.Filter(state.X, timestampSec);
                    double smoothY = _filterY.Filter(state.Y, timestampSec);

                    // 8. MIRROR X-axis (webcam is mirrored!)
                    smoothX = 1.0 - smoothX;
                    // INVERT Y-axis  
                    smoothY = 1.0 - smoothY;

                    // 9. Map from active zone to full range
                    double zoneX = MapFromActiveZone(smoothX, ZoneMinX, ZoneMaxX);
                    double zoneY = MapFromActiveZone(smoothY, ZoneMinY, ZoneMaxY);

                    // 10. Apply adaptive gain (center-anchored)
                    double gain = CalculateAdaptiveGain(velocity);
                    double centerX = 0.5;
                    double centerY = 0.5;
                    double gainedX = centerX + (zoneX - centerX) * gain;
                    double gainedY = centerY + (zoneY - centerY) * gain;

                    // 11. Map to screen coordinates
                    int screenX = (int)(gainedX * Win32Input.ScreenWidth);
                    int screenY = (int)(gainedY * Win32Input.ScreenHeight);

                    // 12. Clamp to screen bounds
                    screenX = Math.Clamp(screenX, 0, Win32Input.ScreenWidth - 1);
                    screenY = Math.Clamp(screenY, 0, Win32Input.ScreenHeight - 1);

                    // 13. Motion Interpolation for smooth dragging
                    // When dragging and there's a significant jump, interpolate intermediate points
                    if (_isLeftDown && _hasLastPosition)
                    {
                        int deltaX = screenX - _lastScreenX;
                        int deltaY = screenY - _lastScreenY;
                        double distance = Math.Sqrt(deltaX * deltaX + deltaY * deltaY);
                        
                        // Only interpolate for significant movements
                        if (distance > 20) // More than 20 pixels
                        {
                            int steps = Math.Min(InterpolationSteps, (int)(distance / 10));
                            for (int i = 1; i < steps; i++)
                            {
                                double t = (double)i / steps;
                                int interpX = _lastScreenX + (int)(deltaX * t);
                                int interpY = _lastScreenY + (int)(deltaY * t);
                                Win32Input.MoveMouse(interpX, interpY);
                                // Small delay to ensure the OS registers the movement
                                // Thread.Sleep(1) would be too slow, so we just do rapid fire
                            }
                        }
                    }

                    // 14. Move cursor to final position
                    bool success = Win32Input.MoveMouse(screenX, screenY);
                    
                    // Update last position
                    _lastScreenX = screenX;
                    _lastScreenY = screenY;
                    _hasLastPosition = true;
                    
                    // Log first success or any failures
                    if (_frameCount == 0 && success)
                    {
                        Console.WriteLine($"[Mouse] First cursor move SUCCESS to ({screenX}, {screenY})");
                    }
                    else if (!success)
                    {
                        Console.WriteLine($"[Mouse] SetCursorPos FAILED for ({screenX}, {screenY})");
                    }

                    // 15. Handle click state machine
                    // GestureId: 0=None, 1=Point, 2=Pinch
                    if (state.GestureId == 2 && !_isLeftDown)
                    {
                        // Pinch started → Mouse down
                        Win32Input.LeftDown();
                        _isLeftDown = true;
                        Console.WriteLine($"[Mouse] Left Down at ({screenX}, {screenY})");
                    }
                    else if (state.GestureId != 2 && _isLeftDown)
                    {
                        // Pinch released → Mouse up
                        Win32Input.LeftUp();
                        _isLeftDown = false;
                        Console.WriteLine($"[Mouse] Left Up at ({screenX}, {screenY})");
                    }

                    // 16. Telemetry (every 30 frames for less spam)
                    _frameCount++;
                    if (_frameCount % 30 == 0)
                    {
                        string clickState = _isLeftDown ? "DOWN" : "UP";
                        string activeState = _isActive ? "ACTIVE" : "PAUSED";
                        Console.WriteLine($"[Mouse] Frames: {_frameCount} | Screen: ({screenX}, {screenY}) | " +
                                        $"Gain: {gain:F1}x | Gesture: {state.GestureId} | Click: {clickState}");
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Expected during graceful shutdown
            }
            finally
            {
                // SAFETY: Always release mouse button on shutdown
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
