using System;
using Grapple.Core;

namespace Grapple.SmokeTests
{
    public class ControlPlaneTest
    {
        public static void Run()
        {
            Console.WriteLine("\n=== Grapple Control Plane Smoke Test ===");

            // TEST 1: The Governor (AtomicMailbox)
            Console.WriteLine("[*] Testing AtomicMailbox LIFO Logic...");
            var mailbox = new AtomicMailbox();

            // Step A: Publish first item
            int dropped = mailbox.Publish(100);
            if (dropped != -1) Fail("Expected -1 (Empty) on first publish.");

            // Step B: Overwrite (simulate lag)
            dropped = mailbox.Publish(200);
            if (dropped != 100) Fail($"Expected 100 to be dropped, got {dropped}.");

            Console.WriteLine("    [+] LIFO Drop logic confirmed (100 was returned).");

            // Step C: Consume
            int consumed = mailbox.Consume();
            if (consumed != 200) Fail($"Expected 200 (Latest), got {consumed}.");

            // Step D: Empty Check
            if (mailbox.Consume() != -1) Fail("Expected Mailbox to be empty.");
            
            Console.WriteLine("[+] SUCCESS: Mailbox behaves correctly.");

            // TEST 2: In-Band Metadata & Memory Safety
            Console.WriteLine("[*] Testing SharedMemoryArena Metadata...");
            using (var arena = new SharedMemoryArena())
            {
                long testTimestamp = 123456789;
                int testPayload = 4096;

                // Step A: Acquire & Write Metadata
                var packet = arena.AcquireNextSlot(testTimestamp, testPayload);
                Console.WriteLine($"    [+] Acquired Slot {packet.BufferId}");

                // Step B: Verify ReadGraphPacket
                var readBack = arena.ReadGraphPacket(packet.BufferId);
                if (readBack.Timestamp != testTimestamp) Fail("Timestamp mismatch!");
                if (readBack.PayloadSize != testPayload) Fail("PayloadSize mismatch!");
                
                Console.WriteLine("    [+] Metadata Read/Write confirmed.");

                // Step C: The "Offset Trap" (CRITICAL)
                // We will write 0xFF to the very first byte of the data payload.
                // If our offset logic is wrong, this might overwrite the PayloadSize or Timestamp.
                var dataSpan = arena.GetSpan(packet.BufferId);
                dataSpan[0] = 0xFF; 

                // Check metadata again to ensure it wasn't corrupted
                var checkAgain = arena.ReadGraphPacket(packet.BufferId);
                if (checkAgain.Timestamp != testTimestamp || checkAgain.PayloadSize != testPayload)
                {
                    Fail("CRITICAL: Writing to data span corrupted the metadata! Check offset logic.");
                }

                Console.WriteLine("    [+] Memory Offset confirmed (Metadata is safe).");
            }

            Console.WriteLine("=== Control Plane Verified ===");
        }

        private static void Fail(string msg)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"[!] FAILURE: {msg}");
            Console.ResetColor();
            Environment.Exit(1);
        }
    }
}

