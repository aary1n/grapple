using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Grapple.Core;

namespace Grapple.Nodes
{
    /// <summary>
    /// A test producer that simulates a 60 FPS 1080p camera.
    /// Generates a moving vertical bar test pattern.
    /// </summary>
    public class SyntheticCaptureNode : IGraphNode
    {
        private readonly SharedMemoryArena _arena;
        private readonly AtomicMailbox _mailbox;
        
        // Constants for 1080p RGB
        private const int Width = 1920;
        private const int Height = 1080;
        private const int BytesPerPixel = 3;
        private const int Stride = Width * BytesPerPixel;
        private const int FrameSize = Stride * Height;
        
        // Timing constants
        private const double TargetFps = 60.0;
        private const long TicksPerSecond = 10_000_000; // Stopwatch frequency on Windows is usually high-res, but we use Ticks
        // However, Stopwatch.GetTimestamp() returns ticks based on Stopwatch.Frequency.
        // We should normalize to QPC ticks or just use Frequency directly.
        // Let's use Stopwatch.Frequency for accurate calculations.
        
        private long _droppedFrames = 0;
        private long _generatedFrames = 0;

        public SyntheticCaptureNode(SharedMemoryArena arena, AtomicMailbox mailbox)
        {
            _arena = arena;
            _mailbox = mailbox;
        }

        public Task StartAsync(CancellationToken ct)
        {
            return Task.Factory.StartNew(() => RunLoop(ct), 
                ct, 
                TaskCreationOptions.LongRunning, 
                TaskScheduler.Default);
        }

        private void RunLoop(CancellationToken ct)
        {
            Console.WriteLine($"[SyntheticCaptureNode] Starting capture at {TargetFps} FPS...");
            
            long frameIntervalTicks = (long)(Stopwatch.Frequency / TargetFps);
            long nextFrameTime = Stopwatch.GetTimestamp();
            
            // Reusable variables to avoid closure allocations if possible (though local vars are fine)
            int frameCount = 0;

            while (!ct.IsCancellationRequested)
            {
                // 1. Precision Timing Loop
                long now = Stopwatch.GetTimestamp();
                if (now < nextFrameTime)
                {
                    long remainingTicks = nextFrameTime - now;
                    // If more than 1ms roughly (assuming 10k ticks per ms is common, but safe to check Frequency)
                    long oneMsTicks = Stopwatch.Frequency / 1000;
                    
                    if (remainingTicks > oneMsTicks)
                    {
                        Thread.Yield(); 
                    }
                    else 
                    {
                        // Busy spin for sub-ms precision
                        SpinWait spin = new SpinWait();
                        while (Stopwatch.GetTimestamp() < nextFrameTime)
                        {
                            spin.SpinOnce();
                        }
                    }
                }
                
                // Update next target time (accumulating to avoid drift)
                nextFrameTime += frameIntervalTicks;
                
                // If we fell way behind, reset to avoid burst processing
                if (Stopwatch.GetTimestamp() > nextFrameTime + frameIntervalTicks)
                {
                    nextFrameTime = Stopwatch.GetTimestamp() + frameIntervalTicks;
                }

                // 2. Acquire Slot
                now = Stopwatch.GetTimestamp();
                GraphPacket packet = _arena.AcquireNextSlot(now, FrameSize);

                // 3. Draw Test Pattern (Zero Alloc)
                Span<byte> span = _arena.GetSpan(packet.BufferId);
                
                // Clear frame to black (0)
                span.Fill(0); 

                // Draw moving vertical bar
                // Bar Width: 50 pixels
                // xOffset ensures we don't go out of bounds: (1920 - 50)
                int barWidth = 50;
                int maxOffset = Width - barWidth;
                int xOffset = (frameCount * 10) % maxOffset;
                
                // Pre-calculate byte offset for X
                int byteOffsetX = xOffset * BytesPerPixel;
                int byteWidth = barWidth * BytesPerPixel; // 150 bytes

                // Fill logic
                // Iterate rows
                for (int y = 0; y < Height; y++)
                {
                    // Calculate row start
                    int rowStart = y * Stride;
                    
                    // Slice the specific segment for the bar and fill with white (255)
                    span.Slice(rowStart + byteOffsetX, byteWidth).Fill(255);
                }

                // 4. Publish
                int droppedId = _mailbox.Publish(packet.BufferId);

                // 5. Telemetry
                if (droppedId != -1)
                {
                    _droppedFrames++;
                }

                _generatedFrames++;
                frameCount++;

                if (_generatedFrames % 600 == 0)
                {
                    Console.WriteLine($"[SyntheticCaptureNode] Gen: {_generatedFrames} | Drops: {_droppedFrames} | FPS: {TargetFps:F1} (Target)");
                }
            }
        }
    }
}

