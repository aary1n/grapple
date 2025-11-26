using System.Runtime.CompilerServices;
using System.Threading;

namespace Grapple.Core
{
    /// <summary>
    /// The Governor: A thread-safe, single-slot hand-off point.
    /// Manages flow between nodes with a LIFO / Drop-Oldest policy.
    /// </summary>
    public sealed class AtomicMailbox
    {
        private volatile int _head = -1; // -1 indicates Empty

        /// <summary>
        /// Atomically swaps the new bufferId into _head.
        /// Returns the previous value.
        /// If the previous value was != -1, it means we just dropped a frame. 
        /// The caller must know this to recycle the old buffer immediately.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Publish(int bufferId)
        {
            return Interlocked.Exchange(ref _head, bufferId);
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
    }
}

