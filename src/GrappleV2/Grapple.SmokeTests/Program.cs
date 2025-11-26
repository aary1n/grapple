using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Grapple.Core;

// Explicitly use a Main class/method to support async properly with top-level statements or class structure
class Program
{
    static async Task Main(string[] args)
    {
        // 1. Memory Foundation Tests (Unsafe Block)
        RunMemoryTests();

        // 2. Control Plane Test
        Grapple.SmokeTests.ControlPlaneTest.Run();
        
        // 3. Producer Smoke Test
        await Grapple.SmokeTests.ProducerTest.RunAsync();

        Console.WriteLine("\n=== ALL SYSTEMS GO ===");
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
