using System.Text.Json.Serialization;

namespace Grapple.Core
{
    /// <summary>
    /// Root configuration for the Grapple pipeline.
    /// Loaded once at startup from grapple_config.json.
    /// All properties have defaults matching current hardcoded values.
    /// </summary>
    public sealed class GrappleConfig
    {
        public WebcamConfig Webcam { get; set; } = new();
        public ArenasConfig Arenas { get; set; } = new();
        public CursorConfig Cursor { get; set; } = new();
        public MediaPipeConfig MediaPipe { get; set; } = new();
        public LandmarkFilterConfig LandmarkFilter { get; set; } = new();
        public PinchConfig Pinch { get; set; } = new();
        public PythonConfig Python { get; set; } = new();
        public SensorConfig Sensor { get; set; } = new();
    }

    public sealed class WebcamConfig
    {
        public int Width { get; set; } = 1920;
        public int Height { get; set; } = 1080;
        public int BackpressureThreshold { get; set; } = 10;
    }

    public sealed class ArenasConfig
    {
        public FrameArenaConfig Frame { get; set; } = new();
        public SmallArenaConfig Hand { get; set; } = new()
        {
            MapName = "Local\\GrappleHandResults",
            SignalName = "Local\\GrappleHandSignal",
            CapacityBytes = 4096
        };
        public SmallArenaConfig Sensor { get; set; } = new()
        {
            MapName = "Local\\GrappleSensorArena",
            SignalName = "Local\\GrappleSensorSignal",
            CapacityBytes = 8192
        };
        public SmallArenaConfig Eye { get; set; } = new()
        {
            MapName = "Local\\GrappleEyeResults",
            SignalName = "Local\\GrappleEyeSignal",
            CapacityBytes = 4096
        };
        public SmallArenaConfig Telemetry { get; set; } = new()
        {
            MapName = "Local\\GrappleTelemetry",
            SignalName = "Local\\GrappleTelemetrySignal",
            CapacityBytes = 4096
        };
        public FrameSignalConfig FrameSignal { get; set; } = new();
    }

    public sealed class FrameArenaConfig
    {
        public string MapName { get; set; } = "Local\\GrappleMap";
        public int CapacityMB { get; set; } = 256;
        public int SlotSizeMB { get; set; } = 8;
    }

    public sealed class SmallArenaConfig
    {
        public string MapName { get; set; } = "";
        public string SignalName { get; set; } = "";
        public long CapacityBytes { get; set; }
    }

    public sealed class FrameSignalConfig
    {
        public string SignalName { get; set; } = "Local\\GrappleSignal";
    }

    public sealed class CursorConfig
    {
        public int UpdateHz { get; set; } = 120;
        public double Sensitivity { get; set; } = 1.3;
        public float MinConfidence { get; set; } = 0.5f;
        public double MaxExtrapolationSec { get; set; } = 0.15;
        public double VelocityDecay { get; set; } = 0.95;
        public OneEuroFilterConfig Filter { get; set; } = new()
        {
            MinCutoff = 0.8,
            Beta = 0.02,
            DCutoff = 1.0
        };
    }

    public sealed class OneEuroFilterConfig
    {
        public double MinCutoff { get; set; } = 1.0;
        public double Beta { get; set; } = 0.0;
        public double DCutoff { get; set; } = 1.0;
    }

    public sealed class MediaPipeConfig
    {
        public int MaxHands { get; set; } = 1;
        public double MinDetectionConfidence { get; set; } = 0.5;
        public double MinTrackingConfidence { get; set; } = 0.4;
        public int ModelComplexity { get; set; } = 0;
    }

    public sealed class LandmarkFilterConfig
    {
        public double MinCutoff { get; set; } = 10.0;
        public double Beta { get; set; } = 2.1;
        public double DCutoff { get; set; } = 2.5;
    }

    public sealed class PinchConfig
    {
        public double EnterThreshold { get; set; } = 0.30;
        public double ExitThreshold { get; set; } = 0.70;
        public int ExitDebounceMs { get; set; } = 150;
        public int EnterConfirmFrames { get; set; } = 3;
        public int ExitConfirmFrames { get; set; } = 5;
        public double RatioAlphaDown { get; set; } = 0.5;
        public double RatioAlphaUp { get; set; } = 0.3;
        public int NoHandGraceFrames { get; set; } = 5;
        public double MinScaleThreshold { get; set; } = 0.001;
        public double MaxRatio { get; set; } = 0.9;
        public double VelocitySmooth { get; set; } = 0.3;
    }

    public sealed class PythonConfig
    {
        public string? PythonPath { get; set; }
        public string? DetectorPath { get; set; }
    }

    public sealed class SensorConfig
    {
        public string Backend { get; set; } = "mediapipe";
    }
}
