using System;
using System.Diagnostics;
using System.Threading;
using Google.FlatBuffers;
using Grapple.Protocol;

namespace Grapple.Core
{
    /// <summary>
    /// Lock-free telemetry metrics accumulator.
    /// Nodes write metrics via Interlocked operations (zero-cost on hot path).
    /// A periodic 10Hz Timer flushes a FlatBuffer TelemetrySnapshot to TelemetryArena.
    /// </summary>
    public sealed class TelemetryCollector : IDisposable
    {
        // Lock-free counters (written by nodes via Interlocked)
        private long _totalFramesProduced;
        private long _totalFramesDropped;
        private long _consecutiveDrops;
        private int _qualityDegradationActive;  // 0 or 1

        // Latency ring buffer (pre-allocated, no GC pressure)
        private readonly double[] _latencySamples;
        private readonly int _maxSamples;
        private int _latencyWriteIndex;
        private int _latencySampleCount;

        // Timing
        private readonly long _startTimestamp;

        // Output
        private readonly TelemetryArena _arena;
        private readonly Timer _flushTimer;
        private readonly FlatBufferBuilder _builder;
        private bool _disposed;

        public TelemetryCollector(TelemetryArena arena, int flushIntervalMs = 100, int maxLatencySamples = 256)
        {
            _arena = arena;
            _maxSamples = maxLatencySamples;
            _latencySamples = new double[maxLatencySamples];
            _startTimestamp = Stopwatch.GetTimestamp();
            _builder = new FlatBufferBuilder(512);

            // Timer fires on ThreadPool at ~10Hz (not on hot path)
            _flushTimer = new Timer(FlushCallback, null, flushIntervalMs, flushIntervalMs);
        }

        // --- Writer methods (called by nodes on their respective threads) ---

        public void RecordFrameProduced()
        {
            Interlocked.Increment(ref _totalFramesProduced);
        }

        public void RecordFrameDropped()
        {
            Interlocked.Increment(ref _totalFramesDropped);
        }

        public void SetConsecutiveDrops(long count)
        {
            Interlocked.Exchange(ref _consecutiveDrops, count);
        }

        public void SetQualityDegradation(bool active)
        {
            Interlocked.Exchange(ref _qualityDegradationActive, active ? 1 : 0);
        }

        public void RecordLatency(double latencyMs)
        {
            int idx = Interlocked.Increment(ref _latencyWriteIndex) - 1;
            _latencySamples[idx & (_maxSamples - 1)] = latencyMs;

            // Increment sample count up to max (relaxed ordering is fine here)
            int currentCount = Volatile.Read(ref _latencySampleCount);
            if (currentCount < _maxSamples)
            {
                Interlocked.CompareExchange(ref _latencySampleCount, currentCount + 1, currentCount);
            }
        }

        // --- Flush (runs on Timer thread, ~10Hz) ---

        private void FlushCallback(object? state)
        {
            if (_disposed) return;

            try
            {
                // Read all counters (relaxed reads are fine for telemetry)
                long framesProduced = Volatile.Read(ref _totalFramesProduced);
                long framesDropped = Volatile.Read(ref _totalFramesDropped);
                long consecutiveDrops = Volatile.Read(ref _consecutiveDrops);
                bool qualityDegradation = Volatile.Read(ref _qualityDegradationActive) != 0;

                // Compute FPS from total frames / uptime
                long now = Stopwatch.GetTimestamp();
                double uptimeSec = (now - _startTimestamp) / (double)Stopwatch.Frequency;
                float fps = uptimeSec > 0.1 ? (float)(framesProduced / uptimeSec) : 0f;

                // Compute latency percentiles from ring buffer
                int sampleCount = Volatile.Read(ref _latencySampleCount);
                float p50 = 0f, p95 = 0f, p99 = 0f;
                float latencyMs = 0f;

                if (sampleCount > 0)
                {
                    // Copy samples to stack for sorting (avoid GC on shared array)
                    Span<double> samples = stackalloc double[Math.Min(sampleCount, _maxSamples)];
                    int count = samples.Length;

                    for (int i = 0; i < count; i++)
                    {
                        samples[i] = _latencySamples[i & (_maxSamples - 1)];
                    }

                    samples.Sort();

                    p50 = (float)samples[(int)(count * 0.50)];
                    p95 = (float)samples[Math.Min((int)(count * 0.95), count - 1)];
                    p99 = (float)samples[Math.Min((int)(count * 0.99), count - 1)];
                    latencyMs = p50;  // Use P50 as the primary latency metric
                }

                // Get GC counts
                int gc0 = GC.CollectionCount(0);
                int gc1 = GC.CollectionCount(1);
                int gc2 = GC.CollectionCount(2);

                // Build FlatBuffer TelemetrySnapshot
                _builder.Clear();

                TelemetrySnapshot.StartTelemetrySnapshot(_builder);
                TelemetrySnapshot.AddFps(_builder, fps);
                TelemetrySnapshot.AddLatencyMs(_builder, latencyMs);
                TelemetrySnapshot.AddDroppedFrames(_builder, (int)framesDropped);
                TelemetrySnapshot.AddGcGen0Collections(_builder, gc0);
                TelemetrySnapshot.AddGcGen1Collections(_builder, gc1);
                TelemetrySnapshot.AddGcGen2Collections(_builder, gc2);
                TelemetrySnapshot.AddConsecutiveDrops(_builder, (int)consecutiveDrops);
                TelemetrySnapshot.AddQualityDegradationActive(_builder, qualityDegradation);
                TelemetrySnapshot.AddTimestamp(_builder, now);
                TelemetrySnapshot.AddLatencyP50Ms(_builder, p50);
                TelemetrySnapshot.AddLatencyP95Ms(_builder, p95);
                TelemetrySnapshot.AddLatencyP99Ms(_builder, p99);
                TelemetrySnapshot.AddTotalFramesProduced(_builder, framesProduced);
                TelemetrySnapshot.AddTotalFramesDropped(_builder, framesDropped);
                TelemetrySnapshot.AddUptimeSeconds(_builder, (float)uptimeSec);
                var offset = TelemetrySnapshot.EndTelemetrySnapshot(_builder);

                _builder.Finish(offset.Value);

                // Write to shared memory arena
                _arena.WriteTelemetry(_builder);
            }
            catch
            {
                // Telemetry flush failures must never crash the pipeline
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
                _flushTimer.Dispose();
            }
        }
    }
}
