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
            : this(arena, mailbox, new WebcamConfig()) { }

        public WebcamCaptureNode(SharedMemoryArena arena, AtomicMailbox mailbox, WebcamConfig config)
        {
            _arena = arena;
            _mailbox = mailbox;
            _targetWidth = config.Width;
            _targetHeight = config.Height;
            _frameSize = _targetWidth * _targetHeight * BytesPerPixel;
            _backpressureThreshold = config.BackpressureThreshold;
        }

        public async ValueTask StartAsync(CancellationToken ct)
        {
            _cancellationToken = ct;

            Console.WriteLine("[Webcam] Enumerating capture devices...");

            // 1. Enumerate available devices
            var captureDevices = new CaptureDevices();
            var devices = captureDevices.EnumerateDescriptors();

            if (!devices.Any())
            {
                throw new InvalidOperationException("No webcam devices found!");
            }

            // 2. Select first available device
            var device = devices.First();
            Console.WriteLine($"[Webcam] Using device: {device.Name}");

            // 3. Find a suitable characteristic (1920x1080)
            var characteristic = FindBestCharacteristic(device);

            if (characteristic == null)
            {
                // List available resolutions for debugging
                Console.WriteLine("[Webcam] Available characteristics:");
                foreach (var c in device.Characteristics)
                {
                    Console.WriteLine($"    {c.Width}x{c.Height} @ {c.FramesPerSecond:F1} FPS - {c.PixelFormat}");
                }
                throw new InvalidOperationException(
                    $"No 1920x1080 characteristic found! Python requires exactly 1920x1080. " +
                    $"Your webcam may not support this resolution.");
            }

            Console.WriteLine($"[Webcam] Selected: {characteristic.Width}x{characteristic.Height} @ {characteristic.FramesPerSecond:F1} FPS - {characteristic.PixelFormat}");

            // 4. Open device with async callback
            _captureDevice = await device.OpenAsync(
                characteristic,
                OnFrameArrivedAsync);

            // 5. Start capture
            _startTimestamp = Stopwatch.GetTimestamp();
            await _captureDevice.StartAsync();

            Console.WriteLine($"[Webcam] Capture started ({_targetWidth}x{_targetHeight} @ {characteristic.FramesPerSecond}fps)");
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
                        Console.WriteLine($"[Webcam] WARNING: Frame too small. Expected >= {_frameSize}, got {imageData.Count}. Skipped: {_skippedFrames}");
                    }
                    return;
                }

                // Log padding info once
                if (_generatedFrames == 0 && imageData.Count > _frameSize)
                {
                    Console.WriteLine($"[Webcam] Note: Frame has {imageData.Count - _frameSize} bytes padding (ignored)");
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

                    if (_consecutiveDrops >= _backpressureThreshold && !_qualityDegradationMode)
                    {
                        _qualityDegradationMode = true;
                        Console.WriteLine($"\n[Webcam] *** BACKPRESSURE DETECTED *** Sustained lag detected ({_consecutiveDrops} consecutive drops)");
                        Console.WriteLine($"[Webcam] Quality degradation mode ACTIVE. Consumer cannot keep up with 60fps.");
                        Console.WriteLine($"[Webcam] Consider: Lower resolution, skip frames, or optimize consumer pipeline.");
                        // TODO Phase 4: Implement adaptive quality (lower resolution to 720p, skip every other frame)
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
                            Console.WriteLine($"\n[Webcam] Quality degradation mode DISABLED. Consumer catching up.");
                        }
                        _consecutiveDrops = 0;
                    }
                }

                // 8. Telemetry
                long frames = Interlocked.Increment(ref _generatedFrames);

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
                Console.WriteLine($"\n[Webcam] ERROR: {ex.Message}");
            }

            await Task.CompletedTask; // Satisfy async signature
        }

        public async ValueTask DisposeAsync()
        {
            if (_captureDevice != null)
            {
                Console.WriteLine("\n[Webcam] Stopping capture...");
                await _captureDevice.StopAsync();
                await _captureDevice.DisposeAsync();
                _captureDevice = null;
                Console.WriteLine($"[Webcam] Stopped. Total frames: {_generatedFrames}, Drops: {_droppedFrames}, Skips: {_skippedFrames}");
            }
        }
    }
}

