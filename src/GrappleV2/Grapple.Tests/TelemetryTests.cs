using System;
using System.Threading;
using Grapple.Core;
using Grapple.Protocol;
using Xunit;

namespace Grapple.Tests
{
    public class TelemetryCollectorTests
    {
        private static SmallArenaConfig TestArenaConfig() => new()
        {
            MapName = $"Local\\GrappleTest_Telemetry_{Guid.NewGuid():N}",
            SignalName = $"Local\\GrappleTest_TelSig_{Guid.NewGuid():N}",
            CapacityBytes = 4096
        };

        [Fact]
        public void RecordFrameProduced_DoesNotThrow()
        {
            using var arena = new TelemetryArena(TestArenaConfig());
            using var collector = new TelemetryCollector(arena, flushIntervalMs: int.MaxValue);

            collector.RecordFrameProduced();
            collector.RecordFrameProduced();
            collector.RecordFrameProduced();
        }

        [Fact]
        public void RecordLatency_RingBufferWraps()
        {
            using var arena = new TelemetryArena(TestArenaConfig());
            using var collector = new TelemetryCollector(arena, flushIntervalMs: int.MaxValue, maxLatencySamples: 8);

            // 16 samples into 8 slots — must not crash
            for (int i = 0; i < 16; i++)
            {
                collector.RecordLatency(i * 1.0);
            }
        }

        [Fact]
        public void FlushWritesToArena()
        {
            var arenaConfig = TestArenaConfig();
            using var arena = new TelemetryArena(arenaConfig);
            using var collector = new TelemetryCollector(arena, flushIntervalMs: 50, maxLatencySamples: 256);

            for (int i = 0; i < 10; i++)
                collector.RecordFrameProduced();
            for (int i = 0; i < 3; i++)
                collector.RecordFrameDropped();
            collector.RecordLatency(5.0);
            collector.RecordLatency(10.0);
            collector.RecordLatency(15.0);

            Thread.Sleep(200);

            var snapshot = arena.ReadLatestTelemetry();
            Assert.NotNull(snapshot);
            Assert.True(snapshot.Value.TotalFramesProduced >= 10,
                $"Expected >= 10 frames produced, got {snapshot.Value.TotalFramesProduced}");
            Assert.True(snapshot.Value.TotalFramesDropped >= 3,
                $"Expected >= 3 frames dropped, got {snapshot.Value.TotalFramesDropped}");
            Assert.True(snapshot.Value.Fps >= 0f);
            Assert.True(snapshot.Value.UptimeSeconds > 0f);
        }

        [Fact]
        public void LatencyPercentiles_Computed()
        {
            var arenaConfig = TestArenaConfig();
            using var arena = new TelemetryArena(arenaConfig);
            using var collector = new TelemetryCollector(arena, flushIntervalMs: 50, maxLatencySamples: 256);

            for (int i = 1; i <= 100; i++)
                collector.RecordLatency(i);

            Thread.Sleep(200);

            var snapshot = arena.ReadLatestTelemetry();
            Assert.NotNull(snapshot);

            Assert.True(snapshot.Value.LatencyP50Ms >= 40f && snapshot.Value.LatencyP50Ms <= 60f,
                $"P50 should be ~50ms, got {snapshot.Value.LatencyP50Ms}");
            Assert.True(snapshot.Value.LatencyP95Ms >= 85f && snapshot.Value.LatencyP95Ms <= 100f,
                $"P95 should be ~95ms, got {snapshot.Value.LatencyP95Ms}");
            Assert.True(snapshot.Value.LatencyP99Ms >= 90f && snapshot.Value.LatencyP99Ms <= 100f,
                $"P99 should be ~99ms, got {snapshot.Value.LatencyP99Ms}");
        }

        [Fact]
        public void GcCountsReported()
        {
            var arenaConfig = TestArenaConfig();
            using var arena = new TelemetryArena(arenaConfig);
            using var collector = new TelemetryCollector(arena, flushIntervalMs: 50);

            collector.RecordFrameProduced();
            Thread.Sleep(200);

            var snapshot = arena.ReadLatestTelemetry();
            Assert.NotNull(snapshot);
            Assert.True(snapshot.Value.GcGen0Collections >= 0);
            Assert.True(snapshot.Value.GcGen1Collections >= 0);
            Assert.True(snapshot.Value.GcGen2Collections >= 0);
        }

        [Fact]
        public void ConsecutiveDropsAndDegradation()
        {
            var arenaConfig = TestArenaConfig();
            using var arena = new TelemetryArena(arenaConfig);
            using var collector = new TelemetryCollector(arena, flushIntervalMs: 50);

            collector.SetConsecutiveDrops(42);
            collector.SetQualityDegradation(true);

            Thread.Sleep(200);

            var snapshot = arena.ReadLatestTelemetry();
            Assert.NotNull(snapshot);
            Assert.Equal(42, snapshot.Value.ConsecutiveDrops);
            Assert.True(snapshot.Value.QualityDegradationActive);
        }

        [Fact]
        public void DisposeStopsTimer()
        {
            var arenaConfig = TestArenaConfig();
            using var arena = new TelemetryArena(arenaConfig);
            var collector = new TelemetryCollector(arena, flushIntervalMs: 50);

            collector.RecordFrameProduced();
            Thread.Sleep(100);

            collector.Dispose();

            long seqAfterDispose = arena.GetSequenceNumber();
            Thread.Sleep(200);

            long seqLater = arena.GetSequenceNumber();
            Assert.Equal(seqAfterDispose, seqLater);
        }

        [Fact]
        public void ArenaRoundTrip_FlatBuffer()
        {
            var arenaConfig = TestArenaConfig();
            using var arena = new TelemetryArena(arenaConfig);

            var builder = new Google.FlatBuffers.FlatBufferBuilder(256);
            TelemetrySnapshot.StartTelemetrySnapshot(builder);
            TelemetrySnapshot.AddFps(builder, 30.0f);
            TelemetrySnapshot.AddLatencyMs(builder, 12.5f);
            TelemetrySnapshot.AddDroppedFrames(builder, 7);
            TelemetrySnapshot.AddLatencyP50Ms(builder, 10.0f);
            TelemetrySnapshot.AddLatencyP95Ms(builder, 20.0f);
            TelemetrySnapshot.AddLatencyP99Ms(builder, 25.0f);
            TelemetrySnapshot.AddTotalFramesProduced(builder, 1000);
            TelemetrySnapshot.AddTotalFramesDropped(builder, 50);
            TelemetrySnapshot.AddUptimeSeconds(builder, 33.3f);
            var offset = TelemetrySnapshot.EndTelemetrySnapshot(builder);
            builder.Finish(offset.Value);

            arena.WriteTelemetry(builder);

            var snapshot = arena.ReadLatestTelemetry();
            Assert.NotNull(snapshot);

            Assert.Equal(30.0f, snapshot.Value.Fps, 0.01f);
            Assert.Equal(12.5f, snapshot.Value.LatencyMs, 0.01f);
            Assert.Equal(7, snapshot.Value.DroppedFrames);
            Assert.Equal(10.0f, snapshot.Value.LatencyP50Ms, 0.01f);
            Assert.Equal(20.0f, snapshot.Value.LatencyP95Ms, 0.01f);
            Assert.Equal(25.0f, snapshot.Value.LatencyP99Ms, 0.01f);
            Assert.Equal(1000L, snapshot.Value.TotalFramesProduced);
            Assert.Equal(50L, snapshot.Value.TotalFramesDropped);
            Assert.Equal(33.3f, snapshot.Value.UptimeSeconds, 0.1f);
        }
    }
}
