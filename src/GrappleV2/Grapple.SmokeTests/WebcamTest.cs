using System;
using System.Threading;
using System.Threading.Tasks;
using Grapple.Core;
using Grapple.Nodes;

namespace Grapple.SmokeTests
{
    /// <summary>
    /// Webcam capture test - verifies real camera frames flow through the pipeline.
    /// Run GrappleDetector.py in another terminal to see hand detection results.
    /// </summary>
    public static class WebcamTest
    {
        public static async Task RunAsync()
        {
            Console.WriteLine("\n=== Webcam Capture Test ===");
            Console.WriteLine("[*] This test captures real webcam frames and writes them to shared memory.");
            Console.WriteLine("[*] Run 'py -3.12 GrappleDetector.py' in another terminal to see hand detection!");
            Console.WriteLine();

            // 1. Setup shared infrastructure
            using var arena = new SharedMemoryArena();
            using var mailbox = new AtomicMailbox();

            // 2. Create webcam node
            await using var webcam = new WebcamCaptureNode(arena, mailbox);
            var cts = new CancellationTokenSource();

            try
            {
                // 3. Start capture
                await webcam.StartAsync(cts.Token);

                Console.WriteLine();
                Console.WriteLine("[*] Webcam active! Wave your hand in front of the camera.");
                Console.WriteLine("[*] Running for 30 seconds... (Press Ctrl+C to stop early)");
                Console.WriteLine();

                // 4. Run for 30 seconds
                await Task.Delay(30000, cts.Token);
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("[*] Test cancelled.");
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[!] Webcam error: {ex.Message}");
                Console.ResetColor();
                
                if (ex.Message.Contains("1920x1080"))
                {
                    Console.WriteLine();
                    Console.WriteLine("[*] TIP: Your webcam may not support 1920x1080.");
                    Console.WriteLine("    Common resolutions: 1280x720, 640x480");
                    Console.WriteLine("    To fix: Update Python's WIDTH/HEIGHT constants to match your webcam.");
                }
            }
            finally
            {
                // Cleanup happens via IAsyncDisposable
                Console.WriteLine("\n=== Webcam Test Complete ===");
            }
        }
    }
}

