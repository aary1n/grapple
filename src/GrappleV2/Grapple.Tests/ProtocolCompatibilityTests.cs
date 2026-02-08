using System.Runtime.InteropServices;
using Google.FlatBuffers;
using Grapple.Core;
using Grapple.Protocol;

namespace Grapple.Tests;

/// <summary>
/// Protocol compatibility tests for Phase 2 FlatBuffers migration.
/// Validates serialization roundtrips, legacy struct sizes, and schema versioning.
/// </summary>
public class ProtocolCompatibilityTests
{
    // === Legacy Struct Size Verification ===

    [Fact]
    public void LegacyHandState_Is56Bytes()
    {
        int size = Marshal.SizeOf<Grapple.Core.HandState>();
        Assert.Equal(56, size);
    }

    // === FlatBuffer Roundtrip: HandState ===

    [Fact]
    public void HandState_FlatBuffer_Roundtrip()
    {
        var builder = new FlatBufferBuilder(256);

        var handOffset = Grapple.Protocol.HandState.CreateHandState(builder,
            x: 0.5,
            y: 0.75,
            z: 0.1,
            velocity_x: 1.2,
            velocity_y: -0.8,
            gesture: GestureType.Pinch,
            confidence: 0.95f,
            timestamp: 123456789L);

        builder.Finish(handOffset.Value);
        byte[] buf = builder.SizedByteArray();

        var bb = new ByteBuffer(buf);
        var hand = Grapple.Protocol.HandState.GetRootAsHandState(bb);

        Assert.Equal(0.5, hand.X);
        Assert.Equal(0.75, hand.Y);
        Assert.Equal(0.1, hand.Z);
        Assert.Equal(1.2, hand.VelocityX);
        Assert.Equal(-0.8, hand.VelocityY);
        Assert.Equal(GestureType.Pinch, hand.Gesture);
        Assert.Equal(0.95f, hand.Confidence);
        Assert.Equal(123456789L, hand.Timestamp);
    }

    // === FlatBuffer Roundtrip: SensorFrame with HandState ===

    [Fact]
    public void SensorFrame_WithHandState_Roundtrip()
    {
        var builder = new FlatBufferBuilder(512);

        var handOffset = Grapple.Protocol.HandState.CreateHandState(builder,
            x: 0.3,
            y: 0.6,
            z: 0.05,
            velocity_x: 0.0,
            velocity_y: 0.0,
            gesture: GestureType.Point,
            confidence: 0.85f,
            timestamp: 999L);

        var frameOffset = SensorFrame.CreateSensorFrame(builder,
            sequence: 42,
            handOffset: handOffset,
            protocol_version: 2);

        SensorFrame.FinishSensorFrameBuffer(builder, frameOffset);
        byte[] buf = builder.SizedByteArray();

        var bb = new ByteBuffer(buf);

        Assert.True(SensorFrame.SensorFrameBufferHasIdentifier(bb));

        var frame = SensorFrame.GetRootAsSensorFrame(bb);

        Assert.Equal(42L, frame.Sequence);
        Assert.Equal(2, frame.ProtocolVersion);

        Assert.NotNull(frame.Hand);
        var hand = frame.Hand!.Value;
        Assert.Equal(0.3, hand.X);
        Assert.Equal(0.6, hand.Y);
        Assert.Equal(GestureType.Point, hand.Gesture);
        Assert.Equal(0.85f, hand.Confidence);
    }

    // === SensorFrame with Optional Fields (null eye/telemetry) ===

    [Fact]
    public void SensorFrame_NullOptionalFields_ReturnsNull()
    {
        var builder = new FlatBufferBuilder(256);

        var handOffset = Grapple.Protocol.HandState.CreateHandState(builder,
            x: 0.5, y: 0.5);

        // Only set hand, no eye or telemetry
        var frameOffset = SensorFrame.CreateSensorFrame(builder,
            sequence: 1,
            handOffset: handOffset);

        SensorFrame.FinishSensorFrameBuffer(builder, frameOffset);
        byte[] buf = builder.SizedByteArray();

        var frame = SensorFrame.GetRootAsSensorFrame(new ByteBuffer(buf));

        Assert.NotNull(frame.Hand);
        Assert.Null(frame.Eye);
        Assert.Null(frame.Telemetry);
    }

    // === FlatBuffer Roundtrip: EyeState ===

    [Fact]
    public void EyeState_FlatBuffer_Roundtrip()
    {
        var builder = new FlatBufferBuilder(256);

        var eyeOffset = Grapple.Protocol.EyeState.CreateEyeState(builder,
            gaze_x: 0.5,
            gaze_y: 0.5,
            pupil_diameter_left: 3.2f,
            pupil_diameter_right: 3.1f,
            confidence: 0.9f,
            timestamp: 555L,
            fixation_duration_ms: 200,
            saccade_velocity: 150.0f);

        builder.Finish(eyeOffset.Value);
        byte[] buf = builder.SizedByteArray();

        var eye = Grapple.Protocol.EyeState.GetRootAsEyeState(new ByteBuffer(buf));

        Assert.Equal(0.5, eye.GazeX);
        Assert.Equal(0.5, eye.GazeY);
        Assert.Equal(3.2f, eye.PupilDiameterLeft);
        Assert.Equal(3.1f, eye.PupilDiameterRight);
        Assert.Equal(0.9f, eye.Confidence);
        Assert.Equal(555L, eye.Timestamp);
        Assert.Equal(200, eye.FixationDurationMs);
        Assert.Equal(150.0f, eye.SaccadeVelocity);
    }

    // === FlatBuffer Roundtrip: TelemetrySnapshot ===

    [Fact]
    public void TelemetrySnapshot_FlatBuffer_Roundtrip()
    {
        var builder = new FlatBufferBuilder(256);

        var telemetryOffset = TelemetrySnapshot.CreateTelemetrySnapshot(builder,
            fps: 15.0f,
            latency_ms: 18.5f,
            dropped_frames: 3,
            gc_gen0_collections: 0,
            gc_gen1_collections: 0,
            gc_gen2_collections: 0,
            consecutive_drops: 2,
            quality_degradation_active: false,
            timestamp: 777L);

        builder.Finish(telemetryOffset.Value);
        byte[] buf = builder.SizedByteArray();

        var telemetry = TelemetrySnapshot.GetRootAsTelemetrySnapshot(new ByteBuffer(buf));

        Assert.Equal(15.0f, telemetry.Fps);
        Assert.Equal(18.5f, telemetry.LatencyMs);
        Assert.Equal(3, telemetry.DroppedFrames);
        Assert.Equal(0, telemetry.GcGen0Collections);
        Assert.Equal(2, telemetry.ConsecutiveDrops);
        Assert.False(telemetry.QualityDegradationActive);
        Assert.Equal(777L, telemetry.Timestamp);
    }

    // === Protocol Version Default ===

    [Fact]
    public void SensorFrame_DefaultProtocolVersion_Is2()
    {
        var builder = new FlatBufferBuilder(256);

        // Don't explicitly set protocol_version (should default to 2)
        SensorFrame.StartSensorFrame(builder);
        SensorFrame.AddSequence(builder, 1);
        var offset = SensorFrame.EndSensorFrame(builder);

        SensorFrame.FinishSensorFrameBuffer(builder, offset);
        byte[] buf = builder.SizedByteArray();

        var frame = SensorFrame.GetRootAsSensorFrame(new ByteBuffer(buf));

        Assert.Equal(2, frame.ProtocolVersion);
    }

    // === GRPL File Identifier ===

    [Fact]
    public void SensorFrame_HasGRPLIdentifier()
    {
        var builder = new FlatBufferBuilder(256);

        SensorFrame.StartSensorFrame(builder);
        var offset = SensorFrame.EndSensorFrame(builder);
        SensorFrame.FinishSensorFrameBuffer(builder, offset);

        byte[] buf = builder.SizedByteArray();
        var bb = new ByteBuffer(buf);

        Assert.True(SensorFrame.SensorFrameBufferHasIdentifier(bb));
    }

    // === SensorFrame with All Sensors ===

    [Fact]
    public void SensorFrame_AllSensors_Roundtrip()
    {
        var builder = new FlatBufferBuilder(1024);

        var handOffset = Grapple.Protocol.HandState.CreateHandState(builder,
            x: 0.4, y: 0.6, gesture: GestureType.Pinch, confidence: 0.9f);

        var eyeOffset = Grapple.Protocol.EyeState.CreateEyeState(builder,
            gaze_x: 0.5, gaze_y: 0.5, confidence: 0.8f);

        var telemetryOffset = TelemetrySnapshot.CreateTelemetrySnapshot(builder,
            fps: 15.0f, latency_ms: 20.0f, dropped_frames: 0);

        var frameOffset = SensorFrame.CreateSensorFrame(builder,
            sequence: 100,
            handOffset: handOffset,
            eyeOffset: eyeOffset,
            telemetryOffset: telemetryOffset,
            protocol_version: 2);

        SensorFrame.FinishSensorFrameBuffer(builder, frameOffset);
        byte[] buf = builder.SizedByteArray();

        var frame = SensorFrame.GetRootAsSensorFrame(new ByteBuffer(buf));

        Assert.Equal(100L, frame.Sequence);
        Assert.NotNull(frame.Hand);
        Assert.NotNull(frame.Eye);
        Assert.NotNull(frame.Telemetry);

        Assert.Equal(GestureType.Pinch, frame.Hand!.Value.Gesture);
        Assert.Equal(0.5, frame.Eye!.Value.GazeX);
        Assert.Equal(15.0f, frame.Telemetry!.Value.Fps);
    }

    // === GestureType Enum Mapping ===

    [Fact]
    public void GestureType_MatchesLegacyIds()
    {
        // Verify FlatBuffer GestureType enum values match legacy int gesture IDs
        Assert.Equal(0, (int)GestureType.None);
        Assert.Equal(1, (int)GestureType.Point);
        Assert.Equal(2, (int)GestureType.Pinch);
    }

    // === Schema Evolution: New Fields Have Defaults ===

    [Fact]
    public void HandState_MissingOptionalFields_HaveDefaults()
    {
        var builder = new FlatBufferBuilder(256);

        // Only set required fields, skip handedness and landmarks
        var handOffset = Grapple.Protocol.HandState.CreateHandState(builder,
            x: 0.5, y: 0.5);

        builder.Finish(handOffset.Value);
        byte[] buf = builder.SizedByteArray();

        var hand = Grapple.Protocol.HandState.GetRootAsHandState(new ByteBuffer(buf));

        // Optional fields should return defaults
        Assert.Null(hand.Handedness);
        Assert.Equal(0, hand.LandmarksLength);
        Assert.Equal(GestureType.None, hand.Gesture);
        Assert.Equal(0.0f, hand.Confidence);
        Assert.Equal(0L, hand.Timestamp);
    }
}
