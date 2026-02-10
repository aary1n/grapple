using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FlashCap;
using Grapple.Core;

namespace Grapple.Nodes
{
    /// <summary>
    /// Real webcam capture node using FlashCap.
    /// Captures frames from the system's default webcam and writes them to the shared memory arena.
    /// </summary>
    public class WebcamCaptureNode : IGraphNode, IAsyncDisposable
    {
        private readonly SharedMemoryArena _arena;
        private readonly AtomicMailbox _mailbox;
        private readonly TelemetryCollector? _telemetry;

        // Configurable (loaded from GrappleConfig at startup)
        private readonly int _targetWidth;
        private readonly int _targetHeight;
        private readonly int _frameSize;
        private readonly int _backpressureThreshold;

        // Protocol constant (not configurable)
        private const int BytesPerPixel = 3;

        private CaptureDevice? _captureDevice;
        private CancellationToken _cancellationToken;

        private long _generatedFrames = 0;
        private long _droppedFrames = 0;
        private long _skippedFrames = 0;
        private long _startTimestamp = 0;

        // Backpressure detection (CV-4 fix)
        private int _consecutiveDrops = 0;
        private bool _qualityDegradationMode = false;

        public WebcamCaptureNode(SharedMemoryArena arena, AtomicMailbox mailbox)
            : this(arena, mailbox, new WebcamConfig(), null) { }

        public WebcamCaptureNode(SharedMemoryArena arena, AtomicMailbox mailbox, WebcamConfig config, TelemetryCollector? telemetry = null)
        {
            _arena = arena;
            _mailbox = mailbox;
            _telemetry = telemetry;
            _targetWidth = config.Width;
            _targetHeight = config.Height;
            _frameSize = _targetWidth * _targetHeight * BytesPerPixel;
            _backpressureThreshold = config.BackpressureThreshold;
        }

        public async ValueTask StartAsync(CancellationToken ct)
        {
            _cancellationToken = ct;

            GrappleLogger.Info("Webcam", "Enumerating capture devices...");

            // 1. Enumerate available devices
            var captureDevices = new CaptureDevices();
            var devices = captureDevices.EnumerateDescriptors();

            if (!devices.Any())
            {
                throw new InvalidOperationException("No webcam devices found!");
            }

            // 2. Select first available device
            var device = devices.First();
            GrappleLogger.Info("Webcam", $"Using device: {device.Name}");

            // 3. Find a suitable characteristic (1920x1080)
            var characteristic = FindBestCharacteristic(device);

            if (characteristic == null)
            {
                // List available resolutions for debugging
                GrappleLogger.Warning("Webcam", "Available characteristics:");
                foreach (var c in device.Characteristics)
                {
                    GrappleLogger.Warning("Webcam", $"{c.Width}x{c.Height} @ {c.FramesPerSecond:F1} FPS - {c.PixelFormat}");
                }
                throw new InvalidOperationException(
                    $"No 1920x1080 characteristic found! Python requires exactly 1920x1080. " +
                    $"Your webcam may not support this resolution.");
            }

            GrappleLogger.Info("Webcam", $"Selected: {characteristic.Width}x{characteristic.Height} @ {characteristic.FramesPerSecond:F1} FPS - {characteristic.PixelFormat}");

            // 4. Open device with async callback
            _captureDevice = await device.OpenAsync(
                characteristic,
                OnFrameArrivedAsync);

            // 5. Start capture
            _startTimestamp = Stopwatch.GetTimestamp();
            await _captureDevice.StartAsync();

            GrappleLogger.Info("Webcam", $"Capture started ({_targetWidth}x{_targetHeight} @ {characteristic.FramesPerSecond}fps)");
        }

        private VideoCharacteristics? FindBestCharacteristic(CaptureDeviceDescriptor device)
        {
            // Priority: 1920x1080, prefer higher FPS, prefer MJPEG (better quality at same bandwidth)
            var candidates = device.Characteristics
                .Where(c => c.Width == _targetWidth && c.Height == _targetHeight)
                .OrderByDescending(c => c.FramesPerSecond)
                .ThenByDescending(c => c.PixelFormat.ToString().Contains("MJPEG") ? 1 : 0)
                .ToList();

            return candidates.FirstOrDefault();
        }

        private async Task OnFrameArrivedAsync(PixelBufferScope bufferScope)
        {
            // Check cancellation
            if (_cancellationToken.IsCancellationRequested)
            {
                return;
            }

            try
            {
                // 1. Get QPC timestamp IMMEDIATELY for accurate latency measurement
                long timestamp = Stopwatch.GetTimestamp();

                // 2. Get the decoded image data from FlashCap
                var imageData = bufferScope.Buffer.ReferImage();

                // 3. Validate frame size (allow >= _frameSize, extra bytes are typically padding)
                if (imageData.Array == null || imageData.Count < _frameSize)
                {
                    // Frame too small - skip this frame
                    Interlocked.Increment(ref _skippedFrames);
                    
                    if (_skippedFrames == 1 || _skippedFrames % 100 == 0)
                    {
                        GrappleLogger.Warning("Webcam", $"Frame too small. Expected >= {_frameSize}, got {imageData.Count}. Skipped: {_skippedFrames}");
                    }
                    return;
                }

                // Log padding info once
                if (_generatedFrames == 0 && imageData.Count > _frameSize)
                {
                    GrappleLogger.Debug("Webcam", $"Frame has {imageData.Count - _frameSize} bytes padding (ignored)");
                }

                // 4. Acquire arena slot
                GraphPacket packet = _arena.AcquireNextSlot(timestamp, _frameSize);
                Span<byte> arenaSpan = _arena.GetSpan(packet.BufferId);

                // 5. Convert BGR to RGB directly into arena
                // FlashCap decodes MJPEG/YUY2 to BGR24
                // Only use first _frameSize bytes from input (ignore padding)
                // Slice output to match input size (arena slot is larger than frame)
                ReadOnlySpan<byte> inputSpan = new ReadOnlySpan<byte>(imageData.Array, imageData.Offset, _frameSize);
                Span<byte> outputSpan = arenaSpan.Slice(0, _frameSize);
                PixelConverter.BgrToRgb(inputSpan, outputSpan);

                // 6. Publish to mailbox
                int droppedId = _mailbox.Publish(packet.BufferId);
                _arena.UpdatePublishedBuffer(packet.BufferId);

                // 7. Backpressure detection (CV-4 fix)
                if (droppedId != -1)
                {
                    Interlocked.Increment(ref _droppedFrames);
                    _consecutiveDrops++;
                    _telemetry?.RecordFrameDropped();

                    if (_consecutiveDrops >= _backpressureThreshold && !_qualityDegradationMode)
                    {
                        _qualityDegradationMode = true;
                        _telemetry?.SetQualityDegradation(true);
                        GrappleLogger.Warning("Webcam", $"BACKPRESSURE DETECTED: {_consecutiveDrops} consecutive drops. Quality degradation mode ACTIVE.");
                    }
                }
                else
                {
                    // Frame was consumed - reset consecutive drop counter
                    if (_consecutiveDrops > 0)
                    {
                        if (_qualityDegradationMode)
                        {
                            _qualityDegradationMode = false;
                            _telemetry?.SetQualityDegradation(false);
                            GrappleLogger.Info("Webcam", "Quality degradation mode DISABLED. Consumer catching up.");
                        }
                        _consecutiveDrops = 0;
                    }
                }

                _telemetry?.SetConsecutiveDrops(_consecutiveDrops);

                // 8. Telemetry
                long frames = Interlocked.Increment(ref _generatedFrames);
                _telemetry?.RecordFrameProduced();

                // Update status line every 10 frames (more responsive, less spam)
                if (frames % 10 == 0)
                {
                    double elapsedSec = (Stopwatch.GetTimestamp() - _startTimestamp) / (double)Stopwatch.Frequency;
                    double fps = frames / elapsedSec;
                    string degradationFlag = _qualityDegradationMode ? " [DEGRADED]" : "";
                    Console.Write($"\r[Webcam] FPS: {fps:F1} | Frames: {frames} | Drops: {_droppedFrames} | ConsecDrops: {_consecutiveDrops}{degradationFlag}          ");
                }
            }
            catch (Exception ex)
            {
                GrappleLogger.Error("Webcam", $"Frame callback error: {ex.Message}");
            }

            await Task.CompletedTask; // Satisfy async signature
        }

        public async ValueTask DisposeAsync()
        {
            if (_captureDevice != null)
            {
                GrappleLogger.Info("Webcam", "Stopping capture...");
                await _captureDevice.StopAsync();
                await _captureDevice.DisposeAsync();
                _captureDevice = null;
                GrappleLogger.Info("Webcam", $"Stopped. Total frames: {_generatedFrames}, Drops: {_droppedFrames}, Skips: {_skippedFrames}");
            }
        }
    }
}

