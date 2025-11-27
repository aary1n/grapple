using System;
using System.Runtime.CompilerServices;
using System.Threading;

namespace Grapple.Core
{
    /// <summary>
    /// The Governor: A thread-safe, single-slot hand-off point.
    /// Manages flow between nodes with a LIFO / Drop-Oldest policy.
    /// Now includes event-based signaling for low-latency consumer wakeup.
    /// </summary>
    public sealed class AtomicMailbox : IDisposable
    {
        private volatile int _head = -1; // -1 indicates Empty
        private readonly ManualResetEventSlim _signal = new(false);
        private bool _disposed = false;

        /// <summary>
        /// Atomically swaps the new bufferId into _head and signals waiting consumers.
        /// Returns the previous value.
        /// If the previous value was != -1, it means we just dropped a frame.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Publish(int bufferId)
        {
            int previous = Interlocked.Exchange(ref _head, bufferId);
            _signal.Set(); // Wake up consumer immediately
            return previous;
        }

        /// <summary>
        /// Atomically swaps -1 into _head.
        /// Returns the value that was there (or -1 if empty).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Consume()
        {
            return Interlocked.Exchange(ref _head, -1);
        }

        /// <summary>
        /// Blocks the calling thread until data is available.
        /// Uses kernel-mode wait for efficient CPU usage.
        /// </summary>
        /// <param name="ct">Cancellation token for graceful shutdown.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void WaitForData(CancellationToken ct)
        {
            _signal.Wait(ct);
        }

        /// <summary>
        /// Resets the signal after consuming data.
        /// Must be called after Consume() to prepare for next frame.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ResetSignal()
        {
            _signal.Reset();
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _signal.Dispose();
                _disposed = true;
            }
        }
    }
}
