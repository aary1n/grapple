using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Grapple.Core;

namespace Grapple.Nodes
{
    /// <summary>
    /// A null sink consumer that validates frame flow and measures latency.
    /// Completes M1 milestone: "Video frames flow to null sink with 0 GC."
    /// </summary>
    public class NullSinkNode : IGraphNode
    {
        private readonly SharedMemoryArena _arena;
        private readonly AtomicMailbox _mailbox;

        private long _framesProcessed = 0;
        private double _lastLatencyMs = 0.0;

        public NullSinkNode(SharedMemoryArena arena, AtomicMailbox mailbox)
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
            Console.WriteLine("[NullSink] Consumer started...");

            SpinWait spin = default; // Stack-allocated, zero-alloc

            while (!ct.IsCancellationRequested)
            {
                int bufferId = _mailbox.Consume();

                if (bufferId == -1)
                {
                    spin.SpinOnce(); // Efficient wait
                    continue;
                }

                spin.Reset(); // Got work - reset spin counter

                // 1. Reconstruct Packet
                GraphPacket packet = _arena.ReadGraphPacket(bufferId);

                // 2. Latency Calculation (High-Resolution)
                long now = Stopwatch.GetTimestamp();
                _lastLatencyMs = (now - packet.Timestamp) * 1000.0 / Stopwatch.Frequency;

                // 3. Memory Access Verification
                // Discard pattern prevents JIT optimization from eliding the read
                Span<byte> span = _arena.GetSpan(bufferId);
                _ = span[span.Length / 2];

                // 4. Telemetry
                _framesProcessed++;

                if (_framesProcessed % 600 == 0) // Every 10 seconds at 60 FPS
                {
                    Console.WriteLine($"[NullSink] Processed: {_framesProcessed} | Latency: {_lastLatencyMs:F2} ms");
                }
            }

            Console.WriteLine("[NullSink] Consumer stopped.");
        }
    }
}

