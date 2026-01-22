using System;
using System.Runtime.CompilerServices;
using System.Threading;

namespace Grapple.Core
{
    /// <summary>
    /// The Governor: A thread-safe, single-slot hand-off point.
    /// Uses named events for cross-process signaling.
    /// </summary>
    public sealed class AtomicMailbox : IDisposable
    {
        private volatile int _head = -1;
        private readonly EventWaitHandle _signal;
        private bool _disposed = false;

        public AtomicMailbox()
        {
            // AutoReset: Automatically resets after one waiter is released
            // "Local\\" namespace: Works for non-admin users within session
            _signal = new EventWaitHandle(false, EventResetMode.AutoReset, "Local\\GrappleSignal");
        }

        /// <summary>
        /// Atomically swaps the new bufferId into _head and signals waiting consumers.
        /// Returns the previous value.
        /// If the previous value was != -1, it means we just dropped a frame.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Publish(int bufferId)
        {
            int previous = Interlocked.Exchange(ref _head, bufferId);
            _signal.Set();
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
        /// Blocks until signaled. For in-process C# consumers.
        /// Uses WaitHandle.WaitAny with cancellation token's wait handle.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void WaitForData(CancellationToken ct)
        {
            int idx = WaitHandle.WaitAny(new[] { _signal, ct.WaitHandle });
            if (idx == 1)  // Index 1 = cancellation was signaled
            {
                throw new OperationCanceledException(ct);
            }
            // Index 0 = data signal (normal case)
        }

        /// <summary>
        /// Blocks until data is available or timeout expires.
        /// Returns true if signaled, false if timed out.
        /// For cross-process consumers (Python).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool WaitForData(int timeoutMs)
        {
            return _signal.WaitOne(timeoutMs);
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
