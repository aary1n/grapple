using System;
using System.Threading;
using System.Threading.Tasks;
using Grapple.Core;
using Grapple.Nodes;

namespace Grapple.SmokeTests
{
    public class ProducerTest
    {
        public static async Task RunAsync()
        {
            Console.WriteLine("=== Synthetic Producer Smoke Test ===");
            
            // 1. Setup Infrastructure
            using var arena = new SharedMemoryArena();
            var mailbox = new AtomicMailbox();
            
            // 2. Setup Node
            var producer = new SyntheticCaptureNode(arena, mailbox);
            var cts = new CancellationTokenSource();

            // 3. Launch
            Console.WriteLine("[*] Starting Producer Task...");
            Task producerTask = producer.StartAsync(cts.Token);

            // 4. Observation Phase
            // The node logs every 600 frames (approx 10 seconds).
            // We run for 12 seconds to ensure we see at least one log entry.
            Console.WriteLine("[*] Running for 12 seconds (expecting ~720 frames)...");
            
            try
            {
                await Task.Delay(12000, cts.Token);
            }
            catch (TaskCanceledException) { }

            // 5. Shutdown
            Console.WriteLine("[*] Requesting Cancellation...");
            cts.Cancel();

            try
            {
                await producerTask;
                Console.WriteLine("[+] Producer Task stopped cleanly.");
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("[+] Producer Task cancelled (Expected).");
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[!] FAILURE: Producer threw unexpected exception: {ex}");
                Console.ResetColor();
            }

            Console.WriteLine("=== Test Complete ===");
        }
    }
}

