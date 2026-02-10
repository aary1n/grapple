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
        private readonly TelemetryCollector? _telemetry;
        private readonly OneEuroFilter _filterX;
        private readonly OneEuroFilter _filterY;

        // === TUNING PARAMETERS (loaded from GrappleConfig at startup) ===

        private readonly float _minConfidence;
        private readonly int _targetUpdateHz;
        private readonly int _updateIntervalMs;
        private readonly double _sensitivity;
        private readonly double _maxExtrapolationSec;
        private readonly double _velocityDecay;

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
        /// Creates a new mouse controller node with default parameters.
        /// </summary>
        public MouseControllerNode()
            : this(new CursorConfig(), new SmallArenaConfig
            {
                MapName = "Local\\GrappleSensorArena",
                SignalName = "Local\\GrappleSensorSignal",
                CapacityBytes = 8192
            }, null) { }

        /// <summary>
        /// Creates a new mouse controller node with config-driven parameters.
        /// </summary>
        public MouseControllerNode(CursorConfig cursorConfig, SmallArenaConfig sensorArenaConfig, TelemetryCollector? telemetry = null)
        {
            _sensorArena = new FlatBufferSensorArena(sensorArenaConfig);
            _telemetry = telemetry;

            _minConfidence = cursorConfig.MinConfidence;
            _targetUpdateHz = cursorConfig.UpdateHz;
            _updateIntervalMs = 1000 / cursorConfig.UpdateHz;
            _sensitivity = cursorConfig.Sensitivity;
            _maxExtrapolationSec = cursorConfig.MaxExtrapolationSec;
            _velocityDecay = cursorConfig.VelocityDecay;

            _filterX = new OneEuroFilter(
                cursorConfig.Filter.MinCutoff,
                cursorConfig.Filter.Beta,
                cursorConfig.Filter.DCutoff);
            _filterY = new OneEuroFilter(
                cursorConfig.Filter.MinCutoff,
                cursorConfig.Filter.Beta,
                cursorConfig.Filter.DCutoff);
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
            GrappleLogger.Info("Mouse", "Inference reader started (FlatBuffer protocol v2)");
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

                    // Record end-to-end latency (inference timestamp → now)
                    if (_telemetry != null && h.Timestamp > 0)
                    {
                        long nowQpc = Stopwatch.GetTimestamp();
                        double latencyMs = (nowQpc - h.Timestamp) / (double)Stopwatch.Frequency * 1000.0;
                        _telemetry.RecordLatency(latencyMs);
                    }

                    // Update extrapolation base state (lock-free double-buffering)
                    UpdateExtrapolationState(legacyState, seq);
                }
            }
            catch (OperationCanceledException) { }

            GrappleLogger.Info("Mouse", "Inference reader stopped.");
        }

        /// <summary>
        /// High-frequency cursor update loop (~120Hz).
        /// Extrapolates position between inference updates using velocity.
        /// </summary>
        private void CursorUpdateLoop(CancellationToken ct)
        {
            GrappleLogger.Info("Mouse", $"Controller started (Virtual: {Win32Input.VirtualScreenWidth}x{Win32Input.VirtualScreenHeight}, Primary: {Win32Input.ScreenWidth}x{Win32Input.ScreenHeight}, {_sensitivity:F1}x sens, {_targetUpdateHz}Hz)");
            GrappleLogger.Info("Mouse", "PAUSED (Press F9 to activate)");
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
                            Win32Input.LeftUpSendInput();
                            _isLeftDown = false;
                        }

                        _hasLastPosition = false;
                        _filterX.Reset();
                        _filterY.Reset();

                        if (_isActive)
                        {
                            Console.Beep(880, 100);
                            GrappleLogger.Info("Mouse", "ACTIVE (F9 to pause)");
                        }
                        else
                        {
                            Console.Beep(440, 200);
                            GrappleLogger.Info("Mouse", "PAUSED (F9 to activate)");
                        }
                    }
                    _wasToggleKeyDown = isToggleKeyDown;

                    // Skip if paused
                    if (!_isActive)
                    {
                        Thread.Sleep(_updateIntervalMs);
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
                    if (gestureId == 0 || confidence < _minConfidence)
                    {
                        noHandFrames++;

                        string noHandAction = "NONE";
                        if (_isLeftDown)
                        {
                            Win32Input.LeftUpSendInput();
                            _isLeftDown = false;
                            noHandAction = "SAFETY_UP";
                            GrappleLogger.Warning("Mouse", "Left Up (hand lost - safety release)");
                        }

                        // === DIAGNOSTIC LOGGING (No Hand) ===
                        Console.WriteLine($"CS\t{_frameCount}\t{gestureId}\tUP\t{noHandAction}\t0.00\t{currentSeq}\t{seqGap}");

                        if (noHandFrames % 120 == 0)
                        {
                            GrappleLogger.DebugThrottled("Mouse", "nohand", $"No hand detected (skipped {noHandFrames} frames)", 2000);
                        }

                        if (noHandFrames > 60)
                        {
                            _filterX.Reset();
                            _filterY.Reset();
                            _hasLastPosition = false;
                        }

                        Thread.Sleep(_updateIntervalMs);
                        continue;
                    }

                    noHandFrames = 0;

                    // 3. Calculate time since last inference
                    long currentQpc = Stopwatch.GetTimestamp();
                    double timeSinceInference = (currentQpc - inferenceQpc) / (double)Stopwatch.Frequency;
                    
                    // Clamp extrapolation time to prevent runaway
                    timeSinceInference = Math.Min(timeSinceInference, _maxExtrapolationSec);

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
                    double scaledX = centerX + (smoothX - centerX) * _sensitivity;
                    double scaledY = centerY + (smoothY - centerY) * _sensitivity;

                    // 8. Clamp to normalized bounds and move cursor via SendInput (DPI-aware, multi-monitor)
                    double clampedX = Math.Clamp(scaledX, 0.0, 1.0);
                    double clampedY = Math.Clamp(scaledY, 0.0, 1.0);
                    Win32Input.MoveMouseVirtual(clampedX, clampedY);

                    // Compute approximate pixel coords for diagnostics/display
                    int screenX = (int)(clampedX * Win32Input.VirtualScreenWidth) + Win32Input.VirtualScreenLeft;
                    int screenY = (int)(clampedY * Win32Input.VirtualScreenHeight) + Win32Input.VirtualScreenTop;
                    _lastScreenX = screenX;
                    _lastScreenY = screenY;
                    _hasLastPosition = true;

                    // 9. Handle click state machine (SendInput-based)
                    string action = "MOVE";
                    if (gestureId == 2 && !_isLeftDown)
                    {
                        Win32Input.LeftDownSendInput();
                        _isLeftDown = true;
                        action = "DOWN";
                        GrappleLogger.Info("Mouse", $"Pinch DOWN at ({screenX}, {screenY})");
                    }
                    else if (gestureId != 2 && _isLeftDown)
                    {
                        Win32Input.LeftUpSendInput();
                        _isLeftDown = false;
                        action = "UP";
                        GrappleLogger.Info("Mouse", $"Pinch UP at ({screenX}, {screenY})");
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
                    var sleepTime = Math.Max(1, _updateIntervalMs - (int)elapsed);
                    Thread.Sleep(sleepTime);
                }
            }
            catch (OperationCanceledException) { }
            finally
            {
                if (_isLeftDown)
                {
                    Win32Input.LeftUpSendInput();
                    _isLeftDown = false;
                }
            }

            GrappleLogger.Info("Mouse", $"Stopped. Total frames: {_frameCount}");
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

