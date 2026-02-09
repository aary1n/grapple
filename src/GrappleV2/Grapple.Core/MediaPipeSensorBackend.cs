using System;
using System.Threading;
using System.Threading.Tasks;

namespace Grapple.Core
{
    /// <summary>
    /// ISensorBackend implementation that wraps the Python/MediaPipe detector sidecar.
    /// Adapter pattern: delegates to PythonProcessManager for the actual work.
    /// </summary>
    public class MediaPipeSensorBackend : ISensorBackend
    {
        private readonly GrappleConfig _config;
        private PythonProcessManager? _pythonManager;

        public string Name => "MediaPipe (Python Sidecar)";
        public bool IsRunning => _pythonManager?.IsRunning ?? false;

        public MediaPipeSensorBackend(GrappleConfig config)
        {
            _config = config;
        }

        public ValueTask StartAsync(CancellationToken ct)
        {
            _pythonManager = new PythonProcessManager(
                _config.Python.PythonPath,
                _config.Python.DetectorPath);

            if (!_pythonManager.Start())
            {
                throw new InvalidOperationException("Failed to start MediaPipe Python detector.");
            }

            return ValueTask.CompletedTask;
        }

        public ValueTask StopAsync()
        {
            _pythonManager?.Stop();
            return ValueTask.CompletedTask;
        }

        public void Dispose()
        {
            _pythonManager?.Dispose();
        }
    }
}
