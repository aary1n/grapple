using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;

namespace Grapple.Core
{
    /// <summary>
    /// The Governor: A thread-safe, single-slot hand-off point.
    /// Uses named events for cross-process signaling.
    /// Supports exactly one consumer (enforced via RegisterConsumer).
    /// </summary>
    public sealed class AtomicMailbox : IDisposable
    {
        private volatile int _head = -1;
        private readonly EventWaitHandle _signal;
        private bool _disposed = false;

        // Single-consumer enforcement (development-time guardrail)
        private int _consumerRegistered = 0;

        public AtomicMailbox()
            : this("Local\\GrappleSignal") { }

        public AtomicMailbox(string signalName)
        {
            _signal = new EventWaitHandle(false, EventResetMode.AutoReset, signalName);
        }

        /// <summary>
        /// Registers a consumer. Throws if a consumer is already registered.
        /// Call this before Consume() or WaitForData() to enforce single-consumer invariant.
        /// </summary>
        public void RegisterConsumer()
        {
            if (Interlocked.CompareExchange(ref _consumerRegistered, 1, 0) != 0)
            {
                throw new InvalidOperationException(
                    "AtomicMailbox already has a registered consumer. " +
                    "Only one consumer is supported. Use separate mailboxes for multiple consumers.");
            }
        }

        /// <summary>
        /// Unregisters the current consumer, allowing a new one to register.
        /// </summary>
        public void UnregisterConsumer()
        {
            Interlocked.Exchange(ref _consumerRegistered, 0);
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
            Debug.Assert(_consumerRegistered == 1, "Consume called without RegisterConsumer");
            return Interlocked.Exchange(ref _head, -1);
        }

        /// <summary>
        /// Blocks until signaled. For in-process C# consumers.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void WaitForData(CancellationToken ct)
        {
            Debug.Assert(_consumerRegistered == 1, "WaitForData called without RegisterConsumer");
            int idx = WaitHandle.WaitAny(new[] { _signal, ct.WaitHandle });
            if (idx == 1)
            {
                throw new OperationCanceledException(ct);
            }
        }

        /// <summary>
        /// Blocks until data is available or timeout expires.
        /// Returns true if signaled, false if timed out.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool WaitForData(int timeoutMs)
        {
            Debug.Assert(_consumerRegistered == 1, "WaitForData called without RegisterConsumer");
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
