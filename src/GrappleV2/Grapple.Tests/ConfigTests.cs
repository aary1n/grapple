using System.Text.Json;
using Grapple.Core;

namespace Grapple.Tests
{
    public class ConfigTests
    {
        [Fact]
        public void GrappleConfig_DefaultValues_MatchLegacyConstants()
        {
            var config = new GrappleConfig();

            // Webcam defaults
            Assert.Equal(1920, config.Webcam.Width);
            Assert.Equal(1080, config.Webcam.Height);
            Assert.Equal(10, config.Webcam.BackpressureThreshold);

            // Frame arena defaults
            Assert.Equal("Local\\GrappleMap", config.Arenas.Frame.MapName);
            Assert.Equal(256, config.Arenas.Frame.CapacityMB);
            Assert.Equal(8, config.Arenas.Frame.SlotSizeMB);

            // Hand arena defaults
            Assert.Equal("Local\\GrappleHandResults", config.Arenas.Hand.MapName);
            Assert.Equal("Local\\GrappleHandSignal", config.Arenas.Hand.SignalName);
            Assert.Equal(4096, config.Arenas.Hand.CapacityBytes);

            // Sensor arena defaults
            Assert.Equal("Local\\GrappleSensorArena", config.Arenas.Sensor.MapName);
            Assert.Equal("Local\\GrappleSensorSignal", config.Arenas.Sensor.SignalName);
            Assert.Equal(8192, config.Arenas.Sensor.CapacityBytes);

            // Eye arena defaults
            Assert.Equal("Local\\GrappleEyeResults", config.Arenas.Eye.MapName);
            Assert.Equal("Local\\GrappleEyeSignal", config.Arenas.Eye.SignalName);

            // Telemetry arena defaults
            Assert.Equal("Local\\GrappleTelemetry", config.Arenas.Telemetry.MapName);

            // Frame signal defaults
            Assert.Equal("Local\\GrappleSignal", config.Arenas.FrameSignal.SignalName);

            // Cursor defaults
            Assert.Equal(120, config.Cursor.UpdateHz);
            Assert.Equal(1.3, config.Cursor.Sensitivity);
            Assert.Equal(0.5f, config.Cursor.MinConfidence);
            Assert.Equal(0.15, config.Cursor.MaxExtrapolationSec);
            Assert.Equal(0.95, config.Cursor.VelocityDecay);
            Assert.Equal(0.8, config.Cursor.Filter.MinCutoff);
            Assert.Equal(0.02, config.Cursor.Filter.Beta);
            Assert.Equal(1.0, config.Cursor.Filter.DCutoff);

            // MediaPipe defaults
            Assert.Equal(1, config.MediaPipe.MaxHands);
            Assert.Equal(0.5, config.MediaPipe.MinDetectionConfidence);
            Assert.Equal(0.4, config.MediaPipe.MinTrackingConfidence);
            Assert.Equal(0, config.MediaPipe.ModelComplexity);

            // Landmark filter defaults
            Assert.Equal(10.0, config.LandmarkFilter.MinCutoff);
            Assert.Equal(2.1, config.LandmarkFilter.Beta);
            Assert.Equal(2.5, config.LandmarkFilter.DCutoff);

            // Pinch defaults
            Assert.Equal(0.30, config.Pinch.EnterThreshold);
            Assert.Equal(0.70, config.Pinch.ExitThreshold);
            Assert.Equal(150, config.Pinch.ExitDebounceMs);
            Assert.Equal(3, config.Pinch.EnterConfirmFrames);
            Assert.Equal(5, config.Pinch.ExitConfirmFrames);
            Assert.Equal(0.5, config.Pinch.RatioAlphaDown);
            Assert.Equal(0.3, config.Pinch.RatioAlphaUp);
            Assert.Equal(5, config.Pinch.NoHandGraceFrames);
            Assert.Equal(0.001, config.Pinch.MinScaleThreshold);
            Assert.Equal(0.9, config.Pinch.MaxRatio);
            Assert.Equal(0.3, config.Pinch.VelocitySmooth);

            // Sensor backend default
            Assert.Equal("mediapipe", config.Sensor.Backend);
        }

        [Fact]
        public void GrappleConfigLoader_MissingFile_ReturnsDefaults()
        {
            var config = GrappleConfigLoader.Load("nonexistent_path_that_does_not_exist.json");
            Assert.NotNull(config);
            Assert.Equal(1920, config.Webcam.Width);
            Assert.Equal(120, config.Cursor.UpdateHz);
            Assert.Equal("mediapipe", config.Sensor.Backend);
        }

        [Fact]
        public void GrappleConfig_PartialJson_PreservesUnspecifiedDefaults()
        {
            string json = """
            {
                "webcam": { "width": 1280, "height": 720 },
                "cursor": { "sensitivity": 2.0, "updateHz": 60 }
            }
            """;

            var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
            var config = JsonSerializer.Deserialize<GrappleConfig>(json, options)!;

            // Overridden values
            Assert.Equal(1280, config.Webcam.Width);
            Assert.Equal(720, config.Webcam.Height);
            Assert.Equal(2.0, config.Cursor.Sensitivity);
            Assert.Equal(60, config.Cursor.UpdateHz);

            // Unspecified values retain defaults
            Assert.Equal(10, config.Webcam.BackpressureThreshold);
            Assert.Equal(0.5f, config.Cursor.MinConfidence);
            Assert.Equal(0.15, config.Cursor.MaxExtrapolationSec);
            Assert.Equal("Local\\GrappleMap", config.Arenas.Frame.MapName);
            Assert.Equal("mediapipe", config.Sensor.Backend);
        }

        [Fact]
        public void GrappleConfig_InvalidJson_LoaderReturnsDefaults()
        {
            // Create a temp file with invalid JSON
            string tempPath = Path.GetTempFileName();
            try
            {
                File.WriteAllText(tempPath, "{ invalid json }}}");
                var config = GrappleConfigLoader.Load(tempPath);
                Assert.NotNull(config);
                Assert.Equal(1920, config.Webcam.Width);
            }
            finally
            {
                File.Delete(tempPath);
            }
        }

        [Fact]
        public void GrappleConfig_FullJson_DeserializesCorrectly()
        {
            string json = """
            {
                "webcam": { "width": 640, "height": 480, "backpressureThreshold": 5 },
                "arenas": {
                    "frame": { "mapName": "Local\\TestMap", "capacityMB": 128, "slotSizeMB": 4 },
                    "hand": { "mapName": "Local\\TestHand", "signalName": "Local\\TestHandSignal", "capacityBytes": 2048 },
                    "frameSignal": { "signalName": "Local\\TestSignal" }
                },
                "cursor": {
                    "updateHz": 60,
                    "sensitivity": 2.5,
                    "minConfidence": 0.7,
                    "filter": { "minCutoff": 1.5, "beta": 0.05, "dCutoff": 2.0 }
                },
                "mediapipe": { "maxHands": 2, "minDetectionConfidence": 0.7 },
                "pinch": { "enterThreshold": 0.25, "exitThreshold": 0.80 },
                "sensor": { "backend": "tobii" }
            }
            """;

            var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
            var config = JsonSerializer.Deserialize<GrappleConfig>(json, options)!;

            Assert.Equal(640, config.Webcam.Width);
            Assert.Equal("Local\\TestMap", config.Arenas.Frame.MapName);
            Assert.Equal(128, config.Arenas.Frame.CapacityMB);
            Assert.Equal("Local\\TestHand", config.Arenas.Hand.MapName);
            Assert.Equal("Local\\TestSignal", config.Arenas.FrameSignal.SignalName);
            Assert.Equal(60, config.Cursor.UpdateHz);
            Assert.Equal(1.5, config.Cursor.Filter.MinCutoff);
            Assert.Equal(2, config.MediaPipe.MaxHands);
            Assert.Equal(0.25, config.Pinch.EnterThreshold);
            Assert.Equal("tobii", config.Sensor.Backend);
        }

        [Fact]
        public void SensorBackendFactory_CreatesMediaPipe()
        {
            var config = new GrappleConfig();
            using var backend = SensorBackendFactory.Create(config);
            Assert.Equal("MediaPipe (Python Sidecar)", backend.Name);
            Assert.False(backend.IsRunning);
        }

        [Fact]
        public void SensorBackendFactory_CreatesTobii()
        {
            var config = new GrappleConfig();
            config.Sensor.Backend = "tobii";
            using var backend = SensorBackendFactory.Create(config);
            Assert.Equal("Tobii Eye Tracker (Stub)", backend.Name);
        }

        [Fact]
        public void SensorBackendFactory_ThrowsOnUnknown()
        {
            var config = new GrappleConfig();
            config.Sensor.Backend = "invalid";
            Assert.Throws<ArgumentException>(() => SensorBackendFactory.Create(config));
        }
    }
}
