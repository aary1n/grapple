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
        
        private long _droppedFrames = 0;
        private long _generatedFrames = 0;

        public SyntheticCaptureNode(SharedMemoryArena arena, AtomicMailbox mailbox)
        {
            _arena = arena;
            _mailbox = mailbox;
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
            Console.WriteLine($"[SyntheticCaptureNode] Starting capture at {TargetFps} FPS...");
            
            long frameIntervalTicks = (long)(Stopwatch.Frequency / TargetFps);
            long oneMsTicks = Stopwatch.Frequency / 1000; // Pre-compute outside loop
            long nextFrameTime = Stopwatch.GetTimestamp();
            
            int frameCount = 0;

            while (!ct.IsCancellationRequested)
            {
                // 1. Precision Timing Loop (Hybrid Yield/Spin)
                // Loop on Yield while >1ms remains, then busy-spin for sub-ms precision
                while (true)
                {
                    long now = Stopwatch.GetTimestamp();
                    if (now >= nextFrameTime)
                        break;
                    
                    long remainingTicks = nextFrameTime - now;
                    
                    if (remainingTicks > oneMsTicks)
                    {
                        // Politely yield CPU while we have significant time remaining
                        Thread.Yield();
                    }
                    else
                    {
                        // Sub-ms precision: busy spin
                        SpinWait spin = default;
                        while (Stopwatch.GetTimestamp() < nextFrameTime)
                        {
                            spin.SpinOnce();
                        }
                        break;
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
                long timestamp = Stopwatch.GetTimestamp();
                GraphPacket packet = _arena.AcquireNextSlot(timestamp, FrameSize);

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

                // 4b. Update shared state for IPC consumers (Python)
                _arena.UpdatePublishedBuffer(packet.BufferId);

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

