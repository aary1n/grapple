using System;
using System.Threading;
using System.Threading.Tasks;
using Grapple.Nodes;

namespace Grapple.SmokeTests
{
    /// <summary>
    /// Mouse control test - verifies hand tracking drives the cursor.
    /// Requires webcam and Python detector running in separate terminals.
    /// </summary>
    public static class MouseControlTest
    {
        public static async Task RunAsync()
        {
            Console.WriteLine("\n=== Mouse Control Test ===");
            Console.WriteLine("[*] This test moves your cursor based on hand tracking!");
            Console.WriteLine();
            Console.WriteLine("[*] Prerequisites:");
            Console.WriteLine("    1. Run 'dotnet run -- --webcam' in another terminal");
            Console.WriteLine("    2. Run 'py -3.12 GrappleDetector.py' in another terminal");
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("[!] WARNING: Your cursor will be controlled by your hand!");
            Console.WriteLine("[!] Press Ctrl+C to stop at any time.");
            Console.ResetColor();
            Console.WriteLine();

            using var controller = new MouseControllerNode(minCutoff: 1.0, beta: 0.007);
            var cts = new CancellationTokenSource();

            // Handle Ctrl+C gracefully
            Console.CancelKeyPress += (s, e) =>
            {
                e.Cancel = true;
                cts.Cancel();
            };

            Console.WriteLine("[*] Starting mouse control (30 seconds)...");
            Console.WriteLine("[*] Point with your index finger to move the cursor!");
            Console.WriteLine();

            await controller.StartAsync(cts.Token);

            try
            {
                await Task.Delay(30000, cts.Token);
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("\n[*] Cancelled by user.");
            }

            Console.WriteLine("\n=== Mouse Control Test Complete ===");
        }
    }
}

