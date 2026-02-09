using System;
using System.Threading;
using System.Threading.Tasks;

namespace Grapple.Core
{
    /// <summary>
    /// Stub ISensorBackend for Tobii eye tracker integration.
    /// Phase 3 placeholder: logs a message and does nothing.
    /// Future: Will use Tobii SDK to write EyeState to EyeResultArena.
    /// </summary>
    public class TobiiSensorBackend : ISensorBackend
    {
        private bool _running;

        public string Name => "Tobii Eye Tracker (Stub)";
        public bool IsRunning => _running;

        public TobiiSensorBackend(GrappleConfig config)
        {
            // Config reserved for future Tobii SDK initialization
        }

        public ValueTask StartAsync(CancellationToken ct)
        {
            Console.WriteLine("[Tobii] STUB: Tobii Eye Tracker backend is not yet implemented.");
            Console.WriteLine("[Tobii] STUB: Would write EyeState to EyeResultArena.");
            _running = true;
            return ValueTask.CompletedTask;
        }

        public ValueTask StopAsync()
        {
            _running = false;
            Console.WriteLine("[Tobii] STUB: Stopped.");
            return ValueTask.CompletedTask;
        }

        public void Dispose()
        {
            _running = false;
        }
    }
}
