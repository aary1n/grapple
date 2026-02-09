using System;

namespace Grapple.Core
{
    /// <summary>
    /// Creates the appropriate ISensorBackend based on configuration.
    /// Simple switch-based factory. No reflection, no plugin loading.
    /// </summary>
    public static class SensorBackendFactory
    {
        public static ISensorBackend Create(GrappleConfig config)
        {
            return config.Sensor.Backend.ToLowerInvariant() switch
            {
                "mediapipe" => new MediaPipeSensorBackend(config),
                "tobii" => new TobiiSensorBackend(config),
                _ => throw new ArgumentException(
                    $"Unknown sensor backend: '{config.Sensor.Backend}'. " +
                    $"Valid options: mediapipe, tobii")
            };
        }
    }
}
