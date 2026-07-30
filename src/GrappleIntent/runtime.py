"""GrappleIntent sidecar runtime — the sensor-producer main loop.

Drop-in replacement for GrappleDetector.py as the Grapple sensor producer:

    VideoFrameReader (SharedMemoryArena)
        → LandmarkExtractor (MediaPipe)
        → ReflexiveEngine (ONNX CPU, watchdog → passthrough)
        → DegradationManager (L0–L3) + PotentialFieldBlender (α)
        → SensorArenaWriter (FlatBufferSensorArena)

Protocol note: the current C# contract is position-based — MouseControllerNode
consumes (x, y, velocity, timestamp) and does its own extrapolation, so we
write the detected wrist position and let the model drive GESTURE
classification. The model's cursor_delta head and the potential-field blend
are computed every frame (wired, logged) and become authoritative when the
delta-based protocol lands. The semantic path is not yet integrated, so the
blend α is 0 and the DegradationManager correctly reports L1.
"""

from __future__ import annotations

import argparse
import logging
import time

from .configs import GrappleIntentConfig, load_config
from .inference.blending import PotentialFieldBlender
from .inference.degradation import DegradationLevel, DegradationManager
from .inference.reflexive_engine import ReflexiveEngine
from .integration.arena_bridge import (
    _CFG as GRAPPLE_CFG,
    GESTURE_NONE,
    SensorArenaWriter,
    VideoFrameReader,
)

logger = logging.getLogger(__name__)


class GrappleIntentRuntime:
    """Owns the sidecar loop and all IPC/model resources."""

    def __init__(
        self,
        config: GrappleIntentConfig,
        onnx_path: str | None = None,
        wait_seconds: float = 15.0,
        stats_interval: int = 120,
    ) -> None:
        self._config = config
        self._onnx_path = onnx_path or config.reflexive.inference.onnx_path
        self._wait_seconds = wait_seconds
        self._stats_interval = stats_interval
        self._running = False

        self._reader = VideoFrameReader()
        self._writer = SensorArenaWriter()
        self._extractor = None
        self._engine: ReflexiveEngine | None = None
        self._degradation = DegradationManager(
            semantic_ramp_seconds=config.degradation.semantic_recovery_ramp_seconds
        )
        self._blender = PotentialFieldBlender(
            gain_k=config.blending.gain_k,
            offset_c0=config.blending.offset_c0,
            decay_tau=config.blending.decay_tau,
        )

    # ── Setup ────────────────────────────────────────────────────────────────

    def _open_video_arena(self) -> bool:
        """Open the frame arena, retrying while the C# producer starts up."""
        deadline = time.monotonic() + self._wait_seconds
        while True:
            if self._reader.open():
                return True
            if time.monotonic() >= deadline:
                logger.error(
                    "Video frame arena unavailable after %.0fs — is the C# "
                    "pipeline running?", self._wait_seconds,
                )
                return False
            time.sleep(0.5)

    def _setup(self) -> bool:
        if not self._open_video_arena():
            return False

        if not self._writer.open():
            logger.error("Could not open sensor arena for writing")
            return False

        from .data.landmarks import LandmarkExtractor

        mp_cfg = GRAPPLE_CFG.get("mediapipe", {})
        self._extractor = LandmarkExtractor(
            max_hands=mp_cfg.get("maxHands", 1),
            min_detection_confidence=mp_cfg.get("minDetectionConfidence", 0.5),
            min_tracking_confidence=mp_cfg.get("minTrackingConfidence", 0.4),
            model_complexity=mp_cfg.get("modelComplexity", 0),
        )

        ic = self._config.reflexive.inference
        self._engine = ReflexiveEngine(
            onnx_path=self._onnx_path,
            watchdog_threshold_ms=ic.watchdog_threshold_ms,
            watchdog_recovery_frames=ic.watchdog_recovery_frames,
        )
        self._engine.load()
        return True

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> int:
        if not self._setup():
            return 1

        logger.info(
            "GrappleIntent sidecar running (degradation=%s, onnx=%s)",
            self._degradation.level.name, self._onnx_path,
        )

        qpc_freq = self._reader.qpc_frequency
        frame_count = 0
        no_hand_count = 0
        latency_accum_ms = 0.0
        last_x = last_y = last_z = 0.0
        stats_t0 = time.perf_counter()
        self._running = True

        try:
            while self._running:
                if not self._reader.wait_for_frame(timeout_ms=100):
                    # Timed out (or events unavailable) — poll anyway
                    pass

                frame, timestamp = self._reader.read_latest_frame()
                if frame is None:
                    if self._reader._event_handle is None:
                        time.sleep(0.005)  # polling fallback pace
                    continue

                time_s = (timestamp / qpc_freq) if qpc_freq else time.perf_counter()

                lm = self._extractor.extract(frame, time_s)
                if lm is None:
                    no_hand_count += 1
                    # Publish a detection-lost frame: last known position,
                    # zero velocity, no gesture — C# sees confidence 0.
                    self._writer.write_sensor_frame(
                        x=last_x, y=last_y, z=last_z,
                        velocity_x=0.0, velocity_y=0.0,
                        gesture_id=GESTURE_NONE, confidence=0.0,
                        timestamp=timestamp,
                    )
                    continue

                result = self._engine.infer(lm.features)

                # Degradation bookkeeping — L2 while the watchdog holds the
                # engine in passthrough, back to L1 once it recovers.
                if result.is_passthrough and self._engine._loaded:
                    self._degradation.on_reflexive_over_budget()
                elif not result.is_passthrough:
                    self._degradation.on_reflexive_recovered()

                # Potential field blend (α = 0 until the semantic path lands;
                # DegradationManager.alpha_multiplier will scale α on L1→L0
                # recovery when it does)
                blended = self._blender.blend(result.cursor_dx, result.cursor_dy)

                self._writer.write_sensor_frame(
                    x=lm.wrist_x, y=lm.wrist_y, z=lm.wrist_z,
                    velocity_x=lm.velocity_x, velocity_y=lm.velocity_y,
                    gesture_id=result.gesture_id,
                    confidence=result.gesture_confidence,
                    timestamp=timestamp,
                )
                last_x, last_y, last_z = lm.wrist_x, lm.wrist_y, lm.wrist_z

                frame_count += 1
                latency_accum_ms += result.latency_ms

                if frame_count % self._stats_interval == 0:
                    elapsed = time.perf_counter() - stats_t0
                    fps = self._stats_interval / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "frames=%d fps=%.1f infer_mean=%.2fms level=%s α=%.3f no_hand=%d",
                        frame_count, fps,
                        latency_accum_ms / self._stats_interval,
                        self._degradation.level.name, blended.alpha, no_hand_count,
                    )
                    latency_accum_ms = 0.0
                    no_hand_count = 0
                    stats_t0 = time.perf_counter()

        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
        finally:
            self._teardown()

        return 0

    def stop(self) -> None:
        self._running = False

    def _teardown(self) -> None:
        if self._extractor is not None:
            self._extractor.close()
        self._reader.close()
        self._writer.close()
        logger.info("GrappleIntent sidecar stopped")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GrappleIntent VLA sidecar")
    parser.add_argument("--config", default=None, help="GrappleIntent YAML config path")
    parser.add_argument("--onnx", default=None,
                        help="Reflexive ONNX model (default: from config; "
                             "missing model -> landmark passthrough)")
    parser.add_argument("--wait-seconds", type=float, default=15.0,
                        help="How long to wait for the C# frame arena")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else getattr(
            logging, config.system.log_level, logging.INFO
        ),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    runtime = GrappleIntentRuntime(
        config, onnx_path=args.onnx, wait_seconds=args.wait_seconds
    )
    return runtime.run()
