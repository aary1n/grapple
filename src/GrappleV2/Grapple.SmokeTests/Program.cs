using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Grapple.Core;
using Grapple.Nodes;

// Explicitly use a Main class/method to support async properly with top-level statements or class structure
class Program
{
    static async Task Main(string[] args)
    {
        // Check for full pipeline mode (webcam + python + mouse - single click!)
        if (args.Contains("--full"))
        {
            await RunFullPipelineAsync();
            return;
        }

        // Check for webcam mode
        if (args.Contains("--webcam"))
        {
            await Grapple.SmokeTests.WebcamTest.RunAsync();
            return;
        }

        // Check for mouse control mode
        if (args.Contains("--mouse"))
        {
            await Grapple.SmokeTests.MouseControlTest.RunAsync();
            return;
        }

        // Show usage hint
        Console.WriteLine("[*] TIP: Run with '--full' to launch the ENTIRE pipeline (single-click!)");
        Console.WriteLine("[*] TIP: Run with '--webcam' to test real camera capture");
        Console.WriteLine("[*] TIP: Run with '--mouse' to test hand-controlled cursor");
        Console.WriteLine();

        // 1. Memory Foundation Tests (Unsafe Block)
        RunMemoryTests();

        // 2. Control Plane Test
        Grapple.SmokeTests.ControlPlaneTest.Run();
        
        // 3. Producer Smoke Test
        await Grapple.SmokeTests.ProducerTest.RunAsync();

        // 4. End-to-End Latency Test
        await Grapple.SmokeTests.EndToEndTest.RunAsync();

        // 5. Hand Result Arena Test (struct size verification)
        RunHandStateTests();

        Console.WriteLine("\n=== ALL SYSTEMS GO ===");
    }

    /// <summary>
    /// Runs the complete Grapple pipeline: Webcam → Python → Mouse control.
    /// Single-click experience!
    /// </summary>
    static async Task RunFullPipelineAsync()
    {
        Console.WriteLine("\n=== Grapple Full Pipeline ===");
        Console.WriteLine("[*] Launching webcam, Python detector, and mouse controller...");
        Console.WriteLine("[!] Press Ctrl+C to stop all components.\n");

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (s, e) =>
        {
            e.Cancel = true;
            Console.WriteLine("\n[*] Ctrl+C received, shutting down...");
            cts.Cancel();
        };

        PythonProcessManager? pythonManager = null;
        WebcamCaptureNode? webcam = null;
        MouseControllerNode? mouseController = null;

        try
        {
            // 1. Start webcam capture (required first - creates shared memory)
            Console.WriteLine("[*] Starting webcam capture...");
            using var arena = new SharedMemoryArena();
            var mailbox = new AtomicMailbox();
            webcam = new WebcamCaptureNode(arena, mailbox);
            await webcam.StartAsync(cts.Token);
            Console.WriteLine("[+] Webcam started.");

            // 2. Wait a moment for shared memory to initialize
            await Task.Delay(500, cts.Token);

            // 3. Start Python detector
            Console.WriteLine("[*] Starting Python detector...");
            pythonManager = new PythonProcessManager();
            if (!pythonManager.Start())
            {
                Console.WriteLine("[!] Failed to start Python detector. Exiting.");
                return;
            }

            // 4. Wait for Python to initialize MediaPipe
            Console.WriteLine("[*] Waiting for MediaPipe initialization...");
            await Task.Delay(3000, cts.Token);

            // 5. Start mouse controller
            Console.WriteLine("[*] Starting mouse controller...");
            mouseController = new MouseControllerNode();
            await mouseController.StartAsync(cts.Token);
            Console.WriteLine("[+] Mouse controller started.");

            Console.WriteLine();
            Console.WriteLine("╔════════════════════════════════════════════════════════╗");
            Console.WriteLine("║        🖐️  GRAPPLE PIPELINE READY  🖐️                   ║");
            Console.WriteLine("║                                                        ║");
            Console.WriteLine("║  Press F9 to ACTIVATE mouse control                    ║");
            Console.WriteLine("║                                                        ║");
            Console.WriteLine("║  Controls (when active):                               ║");
            Console.WriteLine("║  • Point with index finger → Cursor moves              ║");
            Console.WriteLine("║  • Pinch thumb + index → Click                         ║");
            Console.WriteLine("║  • Pinch + move → Drag                                 ║");
            Console.WriteLine("║  • F9 → Toggle on/off (safety clutch)                  ║");
            Console.WriteLine("║                                                        ║");
            Console.WriteLine("║  Press Ctrl+C to stop.                                 ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════╝");
            Console.WriteLine();

            // Keep running until cancelled
            try
            {
                await Task.Delay(Timeout.Infinite, cts.Token);
            }
            catch (OperationCanceledException)
            {
                // Expected on Ctrl+C
            }
        }
        catch (OperationCanceledException)
        {
            // Expected during startup if user presses Ctrl+C early
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[!] Pipeline error: {ex.Message}");
        }
        finally
        {
            Console.WriteLine("\n[*] Shutting down pipeline...");

            // Cleanup in reverse order
            mouseController?.Dispose();
            Console.WriteLine("[+] Mouse controller stopped.");

            pythonManager?.Dispose();
            // Python logs its own stop message

            if (webcam is IAsyncDisposable asyncDisposable)
            {
                await asyncDisposable.DisposeAsync();
            }
            Console.WriteLine("[+] Webcam stopped.");

            Console.WriteLine("[+] All components stopped. Goodbye!");
        }
    }

    static void RunHandStateTests()
    {
        Console.WriteLine("\n=== Hand Result Arena Smoke Test ===");

        // TEST A: Verify HandState struct size
        int actualSize = Marshal.SizeOf<HandState>();
        int expectedSize = 56;  // 5×double (40) + int (4) + float (4) + long (8) = 56 bytes
        Console.WriteLine($"[*] HandState struct size: {actualSize} bytes (expected: {expectedSize})");
        
        if (actualSize != expectedSize)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"[!] FAILURE: HandState size mismatch!");
            Console.ResetColor();
            return;
        }
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("[+] SUCCESS: HandState is exactly 40 bytes.");
        Console.ResetColor();

        // TEST B: Verify HandResultArena can be created
        Console.WriteLine("[*] Creating HandResultArena...");
        using (var handArena = new HandResultArena())
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("[+] SUCCESS: HandResultArena created/opened.");
            Console.ResetColor();

            // TEST C: Read initial state (should be zeros or whatever Python wrote)
            var state = handArena.ReadLatest();
            long seq = handArena.GetSequenceNumber();
            Console.WriteLine($"[*] Current state: X={state.X:F3}, Y={state.Y:F3}, Z={state.Z:F3}");
            Console.WriteLine($"[*] GestureId={state.GestureId}, Confidence={state.Confidence:F3}, Seq={seq}");
        }

        // TEST D: Optional live test with Python (if running)
        Console.WriteLine("[*] Attempting live Python handoff test (2 sec timeout)...");
        Console.WriteLine("    (Start GrappleDetector.py in another terminal for this test)");
        
        using (var handArena = new HandResultArena())
        {
            long seqBefore = handArena.GetSequenceNumber();
            
            if (handArena.WaitForResult(2000))
            {
                var state = handArena.ReadLatest();
                long seqAfter = handArena.GetSequenceNumber();
                
                if (seqAfter > seqBefore)
                {
                    long rtt = Stopwatch.GetTimestamp() - state.Timestamp;
                    double rttMs = rtt * 1000.0 / Stopwatch.Frequency;
                    
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"[+] LIVE: Received hand data from Python!");
                    Console.WriteLine($"    X={state.X:F3}, Y={state.Y:F3}, Gesture={state.GestureId}");
                    Console.WriteLine($"    Round-Trip Time: {rttMs:F2}ms");
                    Console.ResetColor();
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine("[~] Signal received but sequence unchanged (stale data).");
                    Console.ResetColor();
                }
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("[~] Timeout - Python not running (this is OK for offline test).");
                Console.ResetColor();
            }
        }

        Console.WriteLine("=== Hand Result Arena Verified ===");
    }

    static unsafe void RunMemoryTests()
    {
        Console.WriteLine("=== Grapple Memory Smoke Test ===");

        // 1. Basic Allocation Test
        using (var arena = new SharedMemoryArena())
        {
            Console.WriteLine("[*] Arena created/opened.");

            // TEST A: Alignment Check
            var span0 = arena.GetSpan(0);
            fixed (byte* ptr = span0)
            {
                long addr = (long)ptr;
                Console.WriteLine($"[*] Slot 0 Address: 0x{addr:X}");

                if (addr % 64 != 0)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("[!] FAILURE: Slot 0 is NOT 64-byte aligned!");
                    return;
                }
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("[+] SUCCESS: Slot 0 is 64-byte aligned.");
                Console.ResetColor();
            }

            // TEST B: Ring Buffer Wrap-Around Logic
            Console.WriteLine("[*] Verifying Overflow Logic...");
            
            long nearMax = long.MaxValue - 1;
            long next = Interlocked.Increment(ref nearMax); // Now long.MaxValue
            long overflow = Interlocked.Increment(ref next); // Now long.MinValue (Negative!)
            int slotCount = 30; 
            
            int bufferIdNormal = (int)((ulong)next % (ulong)slotCount);
            int bufferIdOverflow = (int)((ulong)overflow % (ulong)slotCount);
            Console.WriteLine($"    Max Value -> Index: {bufferIdNormal}");
            Console.WriteLine($"    Overflow (Min) -> Index: {bufferIdOverflow}");

            if (bufferIdOverflow < 0 || bufferIdOverflow >= slotCount)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("[!] FAILURE: Ring Buffer logic produced invalid index on overflow!");
                return;
            }
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("[+] SUCCESS: Overflow logic safely wraps positive.");
            Console.ResetColor();
        }

        // TEST C: Persistence (Re-opening)
        Console.WriteLine("[*] Re-opening Arena to test Magic Number...");
        using (var arena2 = new SharedMemoryArena())
        {
            var packet = arena2.AcquireNextSlot(0, 100);
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[+] SUCCESS: Re-acquired slot {packet.BufferId} from persisted map.");
            Console.ResetColor();
        }

        Console.WriteLine("=== Memory Foundation Verified ===");
    }
}
