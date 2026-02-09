using System;
using System.Threading;
using System.Threading.Tasks;

namespace Grapple.Core
{
    /// <summary>
    /// Abstraction for a sensor backend that produces hand/eye tracking data.
    /// Each backend manages its own sidecar process or native SDK connection
    /// and writes results to the appropriate shared memory arenas.
    /// </summary>
    public interface ISensorBackend : IDisposable
    {
        /// <summary>
        /// Human-readable name for logging (e.g., "MediaPipe", "Tobii Eye Tracker 5").
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Whether this backend is currently producing data.
        /// </summary>
        bool IsRunning { get; }

        /// <summary>
        /// Starts the sensor backend. May spawn external processes or connect to hardware.
        /// Called once at pipeline startup.
        /// </summary>
        ValueTask StartAsync(CancellationToken ct);

        /// <summary>
        /// Stops the sensor backend and releases resources.
        /// </summary>
        ValueTask StopAsync();
    }
}
