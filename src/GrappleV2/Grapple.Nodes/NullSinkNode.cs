using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Grapple.Core;

namespace Grapple.Nodes
{
    /// <summary>
    /// A null sink consumer that validates frame flow and measures latency.
    /// Uses event-based signaling for sub-millisecond latency.
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

            try
            {
                while (!ct.IsCancellationRequested)
                {
                    // 1. Block until producer signals (efficient kernel wait)
                    _mailbox.WaitForData(ct);

                    // 2. Atomic consume
                    int bufferId = _mailbox.Consume();

                    // 3. Reset signal for next frame
                    _mailbox.ResetSignal();

                    // 4. Spurious wakeup protection
                    if (bufferId == -1)
                    {
                        continue;
                    }

                    // 5. Reconstruct Packet
                    GraphPacket packet = _arena.ReadGraphPacket(bufferId);

                    // 6. Latency Calculation (High-Resolution)
                    long now = Stopwatch.GetTimestamp();
                    _lastLatencyMs = (now - packet.Timestamp) * 1000.0 / Stopwatch.Frequency;

                    // 7. Memory Access Verification
                    Span<byte> span = _arena.GetSpan(bufferId);
                    _ = span[span.Length / 2];

                    // 8. Telemetry
                    _framesProcessed++;

                    if (_framesProcessed % 600 == 0)
                    {
                        Console.WriteLine($"[NullSink] Processed: {_framesProcessed} | Latency: {_lastLatencyMs:F2} ms");
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Expected during graceful shutdown
            }

            Console.WriteLine("[NullSink] Consumer stopped.");
        }
    }
}
