using System;
using System.Threading;
using System.Threading.Tasks;
using Grapple.Core;
using Grapple.Nodes;

namespace Grapple.SmokeTests
{
    public class EndToEndTest
    {
        public static async Task RunAsync()
        {
            Console.WriteLine("\n=== Grapple End-to-End Latency Test ===");
            
            // 1. The Shared Infrastructure
            using var arena = new SharedMemoryArena();
            var mailbox = new AtomicMailbox();
            
            // 2. The Nodes
            var producer = new SyntheticCaptureNode(arena, mailbox);
            var consumer = new NullSinkNode(arena, mailbox);
            
            var cts = new CancellationTokenSource();

            // 3. Ignite the Engine
            // StartAsync returns ValueTask.CompletedTask immediately (non-blocking)
            await producer.StartAsync(cts.Token);
            await consumer.StartAsync(cts.Token);
            
            Console.WriteLine("[*] Pipeline Active. Running for 15 seconds...");
            Console.WriteLine("[*] Expectation: ~900 Frames, 0 Drops, < 0.1ms Latency");

            // 4. The Soak Phase
            try
            {
                await Task.Delay(15000, cts.Token);
            }
            catch (TaskCanceledException) { }

            // 5. Shutdown
            Console.WriteLine("[*] Shutting down...");
            cts.Cancel();
            
            // Give loops time to exit and print final stats
            await Task.Delay(500);
            
            Console.WriteLine("=== End-to-End Test Complete ===");
        }
    }
}

