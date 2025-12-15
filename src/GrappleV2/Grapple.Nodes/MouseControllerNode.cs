using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Grapple.Core;

namespace Grapple.Nodes
{
    /// <summary>
    /// Reads hand tracking data from HandResultArena and moves the Windows cursor.
    /// Uses velocity-based extrapolation for smooth 120Hz cursor updates,
    /// even though Python inference runs at ~15Hz.
    /// </summary>
    public class MouseControllerNode : IGraphNode, IDisposable
    {
        private readonly HandResultArena _arena;
        private readonly OneEuroFilter _filterX;
        private readonly OneEuroFilter _filterY;

        // === TUNING PARAMETERS ===
        
        // Confidence threshold
        private const float MinConfidence = 0.5f;

        // Cursor update rate (decoupled from inference rate!)
        private const int TargetUpdateHz = 120;
        private const int UpdateIntervalMs = 1000 / TargetUpdateHz; // ~8ms

        // Sensitivity: simple linear multiplier (1.0 = 1:1 mapping)
        private const double Sensitivity = 1.3;

        // Extrapolation limits (prevent runaway prediction)
        private const double MaxExtrapolationSec = 0.15; // Max 150ms of prediction
        private const double VelocityDecay = 0.95; // Decay velocity when no new data

        // Telemetry
        private long _frameCount = 0;
        private bool _disposed = false;

        // Click state
        private bool _isLeftDown = false;

        // Motion tracking
        private int _lastScreenX = 0;
        private int _lastScreenY = 0;
        private bool _hasLastPosition = false;

        // Safety clutch state
        private bool _isActive = false;
        private bool _wasToggleKeyDown = false;

        // Extrapolation state (updated when new inference arrives)
        private double _baseX = 0.5;
        private double _baseY = 0.5;
        private double _velocityX = 0.0;
        private double _velocityY = 0.0;
        private long _lastInferenceQpc = 0;
        private int _lastGestureId = 0;
        private float _lastConfidence = 0f;
        private readonly object _stateLock = new object();

        /// <summary>
        /// Creates a new mouse controller node with tuned filter parameters.
        /// </summary>
        public MouseControllerNode()
        {
            _arena = new HandResultArena();
            
            // Responsive filter settings for extrapolated input
            // We can be more aggressive since extrapolation smooths the input
            double minCutoff = 0.8;   // More responsive
            double beta = 0.02;       // Moderate speed adaptation
            double dCutoff = 1.0;
            
            _filterX = new OneEuroFilter(minCutoff, beta, dCutoff);
            _filterY = new OneEuroFilter(minCutoff, beta, dCutoff);
        }

        public ValueTask StartAsync(CancellationToken ct)
        {
            // Start the inference reader thread
            Task.Factory.StartNew(() => InferenceReaderLoop(ct),
                ct,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);

            // Start the high-frequency cursor update thread
            Task.Factory.StartNew(() => CursorUpdateLoop(ct),
                ct,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);

            return ValueTask.CompletedTask;
        }

        /// <summary>
        /// Reads inference results from Python as they arrive (non-blocking on cursor).
        /// Updates the extrapolation base state.
        /// </summary>
        private void InferenceReaderLoop(CancellationToken ct)
        {
            Console.WriteLine("[Mouse] Inference reader started...");
            long lastSeq = -1;

            try
            {
                while (!ct.IsCancellationRequested)
                {
                    // Wait for new inference (blocking here is fine - separate thread)
                    if (!_arena.WaitForResult(100, ct))
                    {
                        continue;
                    }

                    HandState state = _arena.ReadLatest();
                    long seq = _arena.GetSequenceNumber();

                    if (seq == lastSeq)
                    {
                        continue;
                    }
                    lastSeq = seq;

                    // Update extrapolation base state (thread-safe)
                    lock (_stateLock)
                    {
                        _baseX = state.X;
                        _baseY = state.Y;
                        _velocityX = state.VX;
                        _velocityY = state.VY;
                        _lastInferenceQpc = Stopwatch.GetTimestamp();
                        _lastGestureId = state.GestureId;
                        _lastConfidence = state.Confidence;
                    }
                }
            }
            catch (OperationCanceledException) { }

            Console.WriteLine("[Mouse] Inference reader stopped.");
        }

        /// <summary>
        /// High-frequency cursor update loop (~120Hz).
        /// Extrapolates position between inference updates using velocity.
        /// </summary>
        private void CursorUpdateLoop(CancellationToken ct)
        {
            Console.WriteLine("[Mouse] Cursor controller started...");
            Console.WriteLine($"[Mouse] Screen: {Win32Input.ScreenWidth}x{Win32Input.ScreenHeight}");
            Console.WriteLine($"[Mouse] Sensitivity: {Sensitivity:F1}x");
            Console.WriteLine($"[Mouse] Target update rate: {TargetUpdateHz}Hz");
            Console.WriteLine("[Mouse] *** PAUSED *** (Press F9 to activate)");
            Console.Beep(440, 200);

            int noHandFrames = 0;
            var stopwatch = Stopwatch.StartNew();

            try
            {
                while (!ct.IsCancellationRequested)
                {
                    var loopStart = stopwatch.ElapsedMilliseconds;

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

                    // Skip if paused
                    if (!_isActive)
                    {
                        Thread.Sleep(UpdateIntervalMs);
                        continue;
                    }

                    // 1. Get current extrapolation state (thread-safe)
                    double baseX, baseY, velX, velY;
                    long inferenceQpc;
                    int gestureId;
                    float confidence;

                    lock (_stateLock)
                    {
                        baseX = _baseX;
                        baseY = _baseY;
                        velX = _velocityX;
                        velY = _velocityY;
                        inferenceQpc = _lastInferenceQpc;
                        gestureId = _lastGestureId;
                        confidence = _lastConfidence;
                    }

                    // 2. Handle no hand or low confidence
                    if (gestureId == 0 || confidence < MinConfidence)
                    {
                        noHandFrames++;
                        
                        if (_isLeftDown)
                        {
                            Win32Input.LeftUp();
                            _isLeftDown = false;
                            Console.WriteLine("[Mouse] Left Up (hand lost - safety release)");
                        }
                        
                        if (noHandFrames % 120 == 0)
                        {
                            Console.WriteLine($"[Mouse] No hand detected (skipped {noHandFrames} frames)");
                        }
                        
                        if (noHandFrames > 60)
                        {
                            _filterX.Reset();
                            _filterY.Reset();
                            _hasLastPosition = false;
                        }

                        Thread.Sleep(UpdateIntervalMs);
                        continue;
                    }

                    noHandFrames = 0;

                    // 3. Calculate time since last inference
                    long currentQpc = Stopwatch.GetTimestamp();
                    double timeSinceInference = (currentQpc - inferenceQpc) / (double)Stopwatch.Frequency;
                    
                    // Clamp extrapolation time to prevent runaway
                    timeSinceInference = Math.Min(timeSinceInference, MaxExtrapolationSec);

                    // 4. EXTRAPOLATE position using velocity
                    double extrapolatedX = baseX + velX * timeSinceInference;
                    double extrapolatedY = baseY + velY * timeSinceInference;

                    // 5. Apply smoothing filter
                    double timestampSec = currentQpc / (double)Stopwatch.Frequency;
                    double smoothX = _filterX.Filter(extrapolatedX, timestampSec);
                    double smoothY = _filterY.Filter(extrapolatedY, timestampSec);

                    // 6. MIRROR X-axis and invert Y (webcam is mirrored)
                    smoothX = 1.0 - smoothX;
                    smoothY = 1.0 - smoothY;

                    // 7. Apply sensitivity (center-anchored)
                    double centerX = 0.5;
                    double centerY = 0.5;
                    double scaledX = centerX + (smoothX - centerX) * Sensitivity;
                    double scaledY = centerY + (smoothY - centerY) * Sensitivity;

                    // 8. Map to screen coordinates
                    int screenX = (int)(scaledX * Win32Input.ScreenWidth);
                    int screenY = (int)(scaledY * Win32Input.ScreenHeight);

                    // 9. Clamp to screen bounds
                    screenX = Math.Clamp(screenX, 0, Win32Input.ScreenWidth - 1);
                    screenY = Math.Clamp(screenY, 0, Win32Input.ScreenHeight - 1);

                    // 10. Move cursor
                    bool success = Win32Input.MoveMouse(screenX, screenY);
                    
                    _lastScreenX = screenX;
                    _lastScreenY = screenY;
                    _hasLastPosition = true;
                    
                    if (_frameCount == 0 && success)
                    {
                        Console.WriteLine($"[Mouse] First cursor move to ({screenX}, {screenY})");
                    }

                    // 11. Handle click state machine
                    if (gestureId == 2 && !_isLeftDown)
                    {
                        Win32Input.LeftDown();
                        _isLeftDown = true;
                        Console.WriteLine($"[Mouse] Left Down at ({screenX}, {screenY})");
                    }
                    else if (gestureId != 2 && _isLeftDown)
                    {
                        Win32Input.LeftUp();
                        _isLeftDown = false;
                        Console.WriteLine($"[Mouse] Left Up at ({screenX}, {screenY})");
                    }

                    // 12. Telemetry
                    _frameCount++;
                    if (_frameCount % 300 == 0) // Every ~2.5 sec at 120Hz
                    {
                        string clickState = _isLeftDown ? "DOWN" : "UP";
                        Console.WriteLine($"[Mouse] Frames: {_frameCount} | Pos: ({screenX}, {screenY}) | " +
                                        $"Gesture: {gestureId} | Click: {clickState} | " +
                                        $"Extrap: {timeSinceInference * 1000:F0}ms");
                    }

                    // 13. Sleep to maintain target rate
                    var elapsed = stopwatch.ElapsedMilliseconds - loopStart;
                    var sleepTime = Math.Max(1, UpdateIntervalMs - (int)elapsed);
                    Thread.Sleep(sleepTime);
                }
            }
            catch (OperationCanceledException) { }
            finally
            {
                if (_isLeftDown)
                {
                    Win32Input.LeftUp();
                    _isLeftDown = false;
                    Console.WriteLine("[Mouse] Left Up (shutdown - safety release)");
                }
            }

            Console.WriteLine($"[Mouse] Cursor controller stopped. Total frames: {_frameCount}");
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

