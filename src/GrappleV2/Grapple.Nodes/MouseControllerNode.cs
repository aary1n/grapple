using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Grapple.Core;
using Grapple.Protocol;

namespace Grapple.Nodes
{
    /// <summary>
    /// Lock-free extrapolation state for double-buffering between inference (15Hz) and cursor (120Hz) threads.
    /// Padded to 64 bytes (cache line) to prevent false sharing.
    /// </summary>
    [StructLayout(LayoutKind.Explicit, Size = 64)]
    internal struct ExtrapolationState
    {
        [FieldOffset(0)]  public double BaseX;
        [FieldOffset(8)]  public double BaseY;
        [FieldOffset(16)] public double VelocityX;
        [FieldOffset(24)] public double VelocityY;
        [FieldOffset(32)] public long LastInferenceQpc;
        [FieldOffset(40)] public int GestureId;
        [FieldOffset(44)] public float Confidence;
        [FieldOffset(48)] public long SequenceNumber;
        // 8 bytes padding to reach 64 bytes (cache line alignment)
    }

    /// <summary>
    /// Reads hand tracking data from HandResultArena and moves the Windows cursor.
    /// Uses velocity-based extrapolation for smooth 120Hz cursor updates,
    /// even though Python inference runs at ~15Hz.
    /// </summary>
    public class MouseControllerNode : IGraphNode, IDisposable
    {
        private readonly FlatBufferSensorArena _sensorArena;
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
        private long _startTimestamp = 0;
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

        // Lock-free double-buffered extrapolation state
        // Writer (inference thread) writes to inactive slot, then atomically swaps
        // Reader (cursor thread) always reads from stable slot (no tearing, no blocking)
        private ExtrapolationState _state0;
        private ExtrapolationState _state1;
        private int _currentSlot;  // 0 or 1 (atomic updates via Interlocked.Exchange)

        // Diagnostic tracking
        private long _lastLoggedSeq = -1;

        /// <summary>
        /// Creates a new mouse controller node with tuned filter parameters.
        /// </summary>
        public MouseControllerNode()
        {
            _sensorArena = new FlatBufferSensorArena();
            
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
            Console.WriteLine("[Mouse] Inference reader started (FlatBuffer protocol v2)...");
            long lastSeq = -1;

            try
            {
                while (!ct.IsCancellationRequested)
                {
                    // Wait for new inference (blocking here is fine - separate thread)
                    if (!_sensorArena.WaitForResult(100, ct))
                    {
                        continue;
                    }

                    long seq = _sensorArena.GetSequenceNumber();
                    if (seq == lastSeq)
                    {
                        continue;
                    }
                    lastSeq = seq;

                    // Read FlatBuffer SensorFrame (pre-allocated buffer, near-zero alloc)
                    SensorFrame? frame = _sensorArena.ReadLatestSensorFrame();
                    if (frame == null)
                    {
                        continue;
                    }

                    // Extract HandState from SensorFrame
                    Grapple.Protocol.HandState? hand = frame.Value.Hand;
                    if (hand == null)
                    {
                        continue;
                    }

                    var h = hand.Value;

                    // Map FlatBuffer GestureType to legacy int gesture IDs
                    int gestureId = (int)h.Gesture;

                    // Build legacy HandState for the double-buffer update
                    var legacyState = new Core.HandState(
                        h.X, h.Y, h.Z,
                        h.VelocityX, h.VelocityY,
                        gestureId,
                        h.Confidence,
                        h.Timestamp
                    );

                    // Update extrapolation base state (lock-free double-buffering)
                    UpdateExtrapolationState(legacyState, seq);
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
            Console.WriteLine($"[Mouse] Controller started ({Win32Input.ScreenWidth}x{Win32Input.ScreenHeight}, {Sensitivity:F1}x sensitivity, {TargetUpdateHz}Hz)");
            Console.WriteLine("[Mouse] *** PAUSED *** (Press F9 to activate)");
            Console.Beep(440, 200);

            int noHandFrames = 0;
            var stopwatch = Stopwatch.StartNew();
            _startTimestamp = Stopwatch.GetTimestamp();

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
                        }

                        _hasLastPosition = false;
                        _filterX.Reset();
                        _filterY.Reset();

                        if (_isActive)
                        {
                            Console.Beep(880, 100);
                            Console.WriteLine("\n[Mouse] *** ACTIVE *** (F9 to pause)");
                        }
                        else
                        {
                            Console.Beep(440, 200);
                            Console.WriteLine("\n[Mouse] *** PAUSED *** (F9 to activate)");
                        }
                    }
                    _wasToggleKeyDown = isToggleKeyDown;

                    // Skip if paused
                    if (!_isActive)
                    {
                        Thread.Sleep(UpdateIntervalMs);
                        continue;
                    }

                    // 1. Get current extrapolation state (lock-free, stable read)
                    int readSlot = _currentSlot;  // Atomic read (always 0 or 1)
                    ref readonly ExtrapolationState state = ref (readSlot == 0 ? ref _state0 : ref _state1);

                    double baseX = state.BaseX;
                    double baseY = state.BaseY;
                    double velX = state.VelocityX;
                    double velY = state.VelocityY;
                    long inferenceQpc = state.LastInferenceQpc;
                    int gestureId = state.GestureId;
                    float confidence = state.Confidence;
                    long currentSeq = state.SequenceNumber;

                    // Shared diagnostic variable (used in both no-hand and normal paths)
                    long seqGap = currentSeq - _lastLoggedSeq - 1;
                    if (_lastLoggedSeq == -1) seqGap = 0;
                    _lastLoggedSeq = currentSeq;

                    // 2. Handle no hand or low confidence
                    if (gestureId == 0 || confidence < MinConfidence)
                    {
                        noHandFrames++;

                        string noHandAction = "NONE";
                        if (_isLeftDown)
                        {
                            Win32Input.LeftUp();
                            _isLeftDown = false;
                            noHandAction = "SAFETY_UP";
                            Console.WriteLine("[Mouse] Left Up (hand lost - safety release)");
                        }

                        // === DIAGNOSTIC LOGGING (No Hand) ===
                        Console.WriteLine($"CS\t{_frameCount}\t{gestureId}\tUP\t{noHandAction}\t0.00\t{currentSeq}\t{seqGap}");

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
                    
                    // 11. Handle click state machine
                    string action = "MOVE";
                    if (gestureId == 2 && !_isLeftDown)
                    {
                        Win32Input.LeftDown();
                        _isLeftDown = true;
                        action = "DOWN";
                        Console.WriteLine($"\n[+] Pinch DOWN at ({screenX}, {screenY})");
                    }
                    else if (gestureId != 2 && _isLeftDown)
                    {
                        Win32Input.LeftUp();
                        _isLeftDown = false;
                        action = "UP";
                        Console.WriteLine($"\n[-] Pinch UP at ({screenX}, {screenY})");
                    }

                    // === DIAGNOSTIC LOGGING ===
                    string diagClickState = _isLeftDown ? "DOWN" : "UP";
                    Console.WriteLine($"CS\t{_frameCount}\t{gestureId}\t{diagClickState}\t{action}\t{timeSinceInference * 1000:F2}\t{currentSeq}\t{seqGap}");

                    // 12. Update status line every 30 frames (~250ms at 120Hz)
                    _frameCount++;
                    if (_frameCount % 30 == 0)
                    {
                        double elapsedSec = (Stopwatch.GetTimestamp() - _startTimestamp) / (double)Stopwatch.Frequency;
                        double actualHz = _frameCount / elapsedSec;
                        string gestureStr = gestureId switch { 0 => "None", 1 => "Point", 2 => "Pinch", _ => "?" };
                        string clickState = _isLeftDown ? "DOWN" : "UP";

                        Console.Write($"\r[Mouse] Hz: {actualHz:F0} | Pos: ({screenX:D4}, {screenY:D4}) | " +
                                     $"Gesture: {gestureStr,-5} | Click: {clickState,-4} | " +
                                     $"Extrap: {timeSinceInference * 1000:F0}ms     ");
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
                }
            }

            Console.WriteLine($"\n[Mouse] Stopped. Total frames: {_frameCount}");
        }

        /// <summary>
        /// Lock-free writer: Updates extrapolation state using double-buffering pattern.
        /// Writer writes to inactive slot, then atomically swaps to make it visible.
        /// This ensures the 120Hz reader never blocks or sees torn reads.
        /// </summary>
        private void UpdateExtrapolationState(Core.HandState state, long sequenceNumber)
        {
            int readSlot = _currentSlot;  // Read current active slot
            int writeSlot = 1 - readSlot;  // Flip to inactive slot

            // Write to inactive buffer (no contention with reader)
            ref ExtrapolationState bufferToWrite = ref (writeSlot == 0 ? ref _state0 : ref _state1);
            bufferToWrite.BaseX = state.X;
            bufferToWrite.BaseY = state.Y;
            bufferToWrite.VelocityX = state.VX;
            bufferToWrite.VelocityY = state.VY;
            bufferToWrite.LastInferenceQpc = Stopwatch.GetTimestamp();
            bufferToWrite.GestureId = state.GestureId;
            bufferToWrite.Confidence = state.Confidence;
            bufferToWrite.SequenceNumber = sequenceNumber;

            // Atomic swap to make new buffer visible (single interlocked operation)
            Interlocked.Exchange(ref _currentSlot, writeSlot);
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _sensorArena?.Dispose();
                _disposed = true;
            }
        }
    }
}

