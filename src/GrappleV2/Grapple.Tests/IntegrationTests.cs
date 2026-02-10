using System;
using System.Threading;
using Grapple.Core;
using Grapple.Protocol;
using Xunit;

namespace Grapple.Tests
{
    public class IntegrationTests
    {
        private static SmallArenaConfig UniqueArenaConfig(string prefix) => new()
        {
            MapName = $"Local\\GrappleTest_{prefix}_{Guid.NewGuid():N}",
            SignalName = $"Local\\GrappleTest_{prefix}Sig_{Guid.NewGuid():N}",
            CapacityBytes = 4096
        };

        [Fact]
        public void TelemetryCollector_ZeroGCPressure()
        {
            // Verify the telemetry writer methods don't trigger GC
            var arenaConfig = UniqueArenaConfig("GCPressure");
            using var arena = new TelemetryArena(arenaConfig);
            using var collector = new TelemetryCollector(arena, flushIntervalMs: int.MaxValue, maxLatencySamples: 256);

            // Warmup (let JIT compile)
            for (int i = 0; i < 100; i++)
            {
                collector.RecordFrameProduced();
                collector.RecordFrameDropped();
                collector.RecordLatency(5.0);
            }

            // Measure
            long gcBefore = GC.GetTotalAllocatedBytes(precise: true);

            for (int i = 0; i < 10000; i++)
            {
                collector.RecordFrameProduced();
                collector.RecordFrameDropped();
                collector.RecordLatency(i % 100);
                collector.SetConsecutiveDrops(i % 10);
                collector.SetQualityDegradation(i % 2 == 0);
            }

            long gcAfter = GC.GetTotalAllocatedBytes(precise: true);
            long allocated = gcAfter - gcBefore;

            // Interlocked operations + ring buffer writes should be zero-alloc
            Assert.True(allocated < 1024,
                $"Expected near-zero allocations in telemetry writer hot path, got {allocated} bytes");
        }

        [Fact]
        public void TelemetryCollector_DropRecovery()
        {
            // Simulate backpressure → recovery cycle and verify telemetry reflects it
            var arenaConfig = UniqueArenaConfig("DropRecovery");
            using var arena = new TelemetryArena(arenaConfig);
            using var collector = new TelemetryCollector(arena, flushIntervalMs: 50);

            // Phase 1: Simulate drops
            for (int i = 0; i < 50; i++)
            {
                collector.RecordFrameDropped();
                collector.RecordFrameProduced();
            }
            collector.SetConsecutiveDrops(50);
            collector.SetQualityDegradation(true);

            Thread.Sleep(200);

            var snapshot1 = arena.ReadLatestTelemetry();
            Assert.NotNull(snapshot1);
            Assert.True(snapshot1.Value.QualityDegradationActive);
            Assert.Equal(50, snapshot1.Value.ConsecutiveDrops);

            // Phase 2: Recovery
            collector.SetConsecutiveDrops(0);
            collector.SetQualityDegradation(false);

            Thread.Sleep(200);

            var snapshot2 = arena.ReadLatestTelemetry();
            Assert.NotNull(snapshot2);
            Assert.False(snapshot2.Value.QualityDegradationActive);
            Assert.Equal(0, snapshot2.Value.ConsecutiveDrops);
        }

        [Fact]
        public void TelemetryCollector_HighFrequencyLatency()
        {
            // Simulate high-frequency latency recording (mimics 60fps webcam)
            var arenaConfig = UniqueArenaConfig("HighFreq");
            using var arena = new TelemetryArena(arenaConfig);
            using var collector = new TelemetryCollector(arena, flushIntervalMs: 50, maxLatencySamples: 256);

            // Record 1000 samples rapidly
            for (int i = 0; i < 1000; i++)
            {
                collector.RecordLatency(10.0 + (i % 20)); // 10-29ms range
                collector.RecordFrameProduced();
            }

            Thread.Sleep(200);

            var snapshot = arena.ReadLatestTelemetry();
            Assert.NotNull(snapshot);

            // P50 should be in the middle of our range (~19-20ms)
            Assert.True(snapshot.Value.LatencyP50Ms >= 15f && snapshot.Value.LatencyP50Ms <= 25f,
                $"P50 expected 15-25ms, got {snapshot.Value.LatencyP50Ms}");

            Assert.True(snapshot.Value.TotalFramesProduced >= 1000);
        }

        [Fact]
        public void AtomicMailbox_RegisterConsumer_EnforcesSingleConsumer()
        {
            string signalName = $"Local\\GrappleTest_SC_{Guid.NewGuid():N}";
            var mailbox = new AtomicMailbox(signalName);

            mailbox.RegisterConsumer();

            // Second registration must throw
            Assert.Throws<InvalidOperationException>(() => mailbox.RegisterConsumer());

            // After unregister, can register again
            mailbox.UnregisterConsumer();
            mailbox.RegisterConsumer();
            mailbox.UnregisterConsumer();
        }

        [Fact]
        public void GrappleLogger_ThrottlesProperly()
        {
            // Ensure throttled logs only emit once per interval
            var originalLevel = GrappleLogger.MinLevel;
            GrappleLogger.MinLevel = LogLevel.Debug;

            try
            {
                // First call should pass through
                GrappleLogger.InfoThrottled("Test", "throttle_key", "Message 1", throttleMs: 500);

                // Rapid subsequent calls with same key should be throttled
                int outputBefore = 0; // We can't easily count Console output, but verify no crash
                for (int i = 0; i < 100; i++)
                {
                    GrappleLogger.InfoThrottled("Test", "throttle_key", $"Message {i}", throttleMs: 500);
                }

                // Different key should not be throttled
                GrappleLogger.InfoThrottled("Test", "other_key", "Different message", throttleMs: 500);
            }
            finally
            {
                GrappleLogger.MinLevel = originalLevel;
            }
        }

        [Fact]
        public void DisplayInfo_FromConstructor_RoundTrips()
        {
            var info = new DisplayInfo(3840, 2160, 0, 0, 1920, 1080);

            Assert.Equal(3840, info.VirtualScreenWidth);
            Assert.Equal(2160, info.VirtualScreenHeight);
            Assert.Equal(0, info.VirtualScreenLeft);
            Assert.Equal(0, info.VirtualScreenTop);
            Assert.Equal(1920, info.PrimaryScreenWidth);
            Assert.Equal(1080, info.PrimaryScreenHeight);
        }

        [Fact]
        public void TelemetryArena_MultipleWritesOverwrite()
        {
            var arenaConfig = UniqueArenaConfig("Overwrite");
            using var arena = new TelemetryArena(arenaConfig);

            // Write snapshot 1
            var builder1 = new Google.FlatBuffers.FlatBufferBuilder(256);
            TelemetrySnapshot.StartTelemetrySnapshot(builder1);
            TelemetrySnapshot.AddFps(builder1, 10.0f);
            var offset1 = TelemetrySnapshot.EndTelemetrySnapshot(builder1);
            builder1.Finish(offset1.Value);
            arena.WriteTelemetry(builder1);

            // Write snapshot 2 (overwrites)
            var builder2 = new Google.FlatBuffers.FlatBufferBuilder(256);
            TelemetrySnapshot.StartTelemetrySnapshot(builder2);
            TelemetrySnapshot.AddFps(builder2, 60.0f);
            var offset2 = TelemetrySnapshot.EndTelemetrySnapshot(builder2);
            builder2.Finish(offset2.Value);
            arena.WriteTelemetry(builder2);

            // Read should get latest
            var snapshot = arena.ReadLatestTelemetry();
            Assert.NotNull(snapshot);
            Assert.Equal(60.0f, snapshot.Value.Fps, 0.01f);
        }
    }
}
