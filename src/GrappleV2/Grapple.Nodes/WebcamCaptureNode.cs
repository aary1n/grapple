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

        // Frame dimensions - MUST match Python's hardcoded values
        private const int TargetWidth = 1920;
        private const int TargetHeight = 1080;
        private const int BytesPerPixel = 3;
        private const int FrameSize = TargetWidth * TargetHeight * BytesPerPixel; // 6,220,800 bytes

        private CaptureDevice? _captureDevice;
        private CancellationToken _cancellationToken;

        private long _generatedFrames = 0;
        private long _droppedFrames = 0;
        private long _skippedFrames = 0;

        public WebcamCaptureNode(SharedMemoryArena arena, AtomicMailbox mailbox)
        {
            _arena = arena;
            _mailbox = mailbox;
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
            Console.WriteLine("[Webcam] Starting capture...");
            await _captureDevice.StartAsync();

            Console.WriteLine($"[Webcam] Capture active. Feeding {TargetWidth}x{TargetHeight} frames to arena.");
        }

        private VideoCharacteristics? FindBestCharacteristic(CaptureDeviceDescriptor device)
        {
            // Priority: 1920x1080, prefer higher FPS, prefer MJPEG (better quality at same bandwidth)
            var candidates = device.Characteristics
                .Where(c => c.Width == TargetWidth && c.Height == TargetHeight)
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

                // 3. Validate frame size (allow >= FrameSize, extra bytes are typically padding)
                if (imageData.Array == null || imageData.Count < FrameSize)
                {
                    // Frame too small - skip this frame
                    Interlocked.Increment(ref _skippedFrames);
                    
                    if (_skippedFrames == 1 || _skippedFrames % 100 == 0)
                    {
                        Console.WriteLine($"[Webcam] WARNING: Frame too small. Expected >= {FrameSize}, got {imageData.Count}. Skipped: {_skippedFrames}");
                    }
                    return;
                }

                // Log padding info once
                if (_generatedFrames == 0 && imageData.Count > FrameSize)
                {
                    Console.WriteLine($"[Webcam] Note: Frame has {imageData.Count - FrameSize} bytes padding (ignored)");
                }

                // 4. Acquire arena slot
                GraphPacket packet = _arena.AcquireNextSlot(timestamp, FrameSize);
                Span<byte> arenaSpan = _arena.GetSpan(packet.BufferId);

                // 5. Convert BGR to RGB directly into arena
                // FlashCap decodes MJPEG/YUY2 to BGR24
                // Only use first FrameSize bytes (ignore padding)
                ReadOnlySpan<byte> inputSpan = new ReadOnlySpan<byte>(imageData.Array, imageData.Offset, FrameSize);
                PixelConverter.BgrToRgb(inputSpan, arenaSpan);

                // 6. Publish to mailbox
                int droppedId = _mailbox.Publish(packet.BufferId);
                _arena.UpdatePublishedBuffer(packet.BufferId);

                // 7. Telemetry
                long frames = Interlocked.Increment(ref _generatedFrames);
                if (droppedId != -1)
                {
                    Interlocked.Increment(ref _droppedFrames);
                }

                // Log every 30 frames (~1 second at 30fps webcam)
                if (frames % 30 == 0)
                {
                    Console.WriteLine($"[Webcam] Frames: {frames} | Drops: {_droppedFrames}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Webcam] ERROR in frame handler: {ex.Message}");
            }

            await Task.CompletedTask; // Satisfy async signature
        }

        public async ValueTask DisposeAsync()
        {
            if (_captureDevice != null)
            {
                Console.WriteLine("[Webcam] Stopping capture...");
                await _captureDevice.StopAsync();
                await _captureDevice.DisposeAsync();
                _captureDevice = null;
                Console.WriteLine($"[Webcam] Stopped. Total frames: {_generatedFrames}, Drops: {_droppedFrames}, Skips: {_skippedFrames}");
            }
        }
    }
}

