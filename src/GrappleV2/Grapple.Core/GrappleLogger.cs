using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;

namespace Grapple.Core
{
    public enum LogLevel
    {
        Debug = 0,
        Info = 1,
        Warning = 2,
        Error = 3,
        Silent = 4
    }

    /// <summary>
    /// Lightweight structured logger for the Grapple pipeline.
    /// Outputs JSON-lines to stdout. Per-category throttling prevents log spam from hot loops.
    /// No external dependencies (no Serilog, no ILogger).
    /// </summary>
    public static class GrappleLogger
    {
        public static LogLevel MinLevel { get; set; } = LogLevel.Info;

        // Per-category throttle tracking: key = "category:message_prefix", value = last log timestamp (ticks)
        private static readonly ConcurrentDictionary<string, long> _throttleMap = new();

        // Default throttle interval (ms) for repeated messages in the same category
        private const int DefaultThrottleMs = 1000;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Debug(string category, string message)
        {
            if (MinLevel <= LogLevel.Debug)
                WriteLog(LogLevel.Debug, category, message);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Info(string category, string message)
        {
            if (MinLevel <= LogLevel.Info)
                WriteLog(LogLevel.Info, category, message);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Warning(string category, string message)
        {
            if (MinLevel <= LogLevel.Warning)
                WriteLog(LogLevel.Warning, category, message);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Error(string category, string message)
        {
            if (MinLevel <= LogLevel.Error)
                WriteLog(LogLevel.Error, category, message);
        }

        /// <summary>
        /// Throttled log: only emits if at least <paramref name="throttleMs"/> have passed
        /// since the last log with the same category+throttleKey combo.
        /// Use in hot loops to avoid log spam while still surfacing periodic status.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void InfoThrottled(string category, string throttleKey, string message, int throttleMs = DefaultThrottleMs)
        {
            if (MinLevel > LogLevel.Info) return;
            if (ShouldThrottle(category, throttleKey, throttleMs)) return;
            WriteLog(LogLevel.Info, category, message);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void DebugThrottled(string category, string throttleKey, string message, int throttleMs = DefaultThrottleMs)
        {
            if (MinLevel > LogLevel.Debug) return;
            if (ShouldThrottle(category, throttleKey, throttleMs)) return;
            WriteLog(LogLevel.Debug, category, message);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WarningThrottled(string category, string throttleKey, string message, int throttleMs = DefaultThrottleMs)
        {
            if (MinLevel > LogLevel.Warning) return;
            if (ShouldThrottle(category, throttleKey, throttleMs)) return;
            WriteLog(LogLevel.Warning, category, message);
        }

        private static bool ShouldThrottle(string category, string throttleKey, int throttleMs)
        {
            string key = string.Concat(category, ":", throttleKey);
            long nowTicks = Stopwatch.GetTimestamp();

            if (_throttleMap.TryGetValue(key, out long lastTicks))
            {
                double elapsedMs = (nowTicks - lastTicks) / (double)Stopwatch.Frequency * 1000.0;
                if (elapsedMs < throttleMs)
                    return true;
            }

            _throttleMap[key] = nowTicks;
            return false;
        }

        private static void WriteLog(LogLevel level, string category, string message)
        {
            // Minimal JSON-lines output: {"ts":"...","lvl":"...","cat":"...","msg":"..."}
            // Use UTC ISO-8601 for timestamp
            string ts = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ");
            string lvl = level switch
            {
                LogLevel.Debug => "DBG",
                LogLevel.Info => "INF",
                LogLevel.Warning => "WRN",
                LogLevel.Error => "ERR",
                _ => "???"
            };

            // Escape quotes in message for valid JSON
            string escapedMsg = message.Replace("\\", "\\\\").Replace("\"", "\\\"");
            string escapedCat = category.Replace("\\", "\\\\").Replace("\"", "\\\"");

            Console.WriteLine($"{{\"ts\":\"{ts}\",\"lvl\":\"{lvl}\",\"cat\":\"{escapedCat}\",\"msg\":\"{escapedMsg}\"}}");
        }
    }
}
