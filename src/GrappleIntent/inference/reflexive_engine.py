"""Reflexive inference engine — 120Hz cursor control on CPU.

From vla-architecture.md §1 and §8:
    - Runtime: ONNX Runtime CPU with AVX-512. ALWAYS CPU — never GPU.
    - Latency: ≤10ms hard limit, ≤5ms target
    - Watchdog: >8ms → raw landmark passthrough
    - Recovery: resume when latency below threshold for 10 consecutive frames
    - Quantization: INT4-AWQ

From vla-architecture.md §9 (latency budget):
    Frame read:     <0.5ms
    Preprocessing:  ≤1ms
    Inference:      ≤5ms  [WATCHDOG: >8ms total → passthrough]
    Prototype:      <0.5ms
    Postprocessing: ≤0.5ms
    Blend:          <0.1ms
    FlatBuffer:     <0.5ms
    Headroom:       ~1.9ms
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReflexiveResult:
    """Result of a single reflexive inference step."""

    cursor_dx: float
    cursor_dy: float
    gesture_id: int  # 0=None, 1=Point, 2=Pinch, 3=Grab, 4=Swipe
    gesture_confidence: float
    embedding: np.ndarray | None  # (embed_dim,) — for prototype lookup
    latency_ms: float
    is_passthrough: bool  # True if watchdog triggered passthrough


class ReflexiveWatchdog:
    """Monitors reflexive path latency and triggers passthrough fallback.

    From §1: If reflexive inference latency exceeds 8ms (80% of 10ms budget):
        1. Log CRITICAL warning
        2. Fall back to raw landmark passthrough
        3. Resume when latency returns below threshold for 10 consecutive frames
    """

    def __init__(self, threshold_ms: float = 8.0, recovery_frames: int = 10):
        self.threshold_ms = threshold_ms
        self.recovery_frames = recovery_frames
        self._consecutive_ok: int = 0
        self._in_passthrough: bool = False

    @property
    def is_passthrough(self) -> bool:
        return self._in_passthrough

    def check(self, latency_ms: float) -> bool:
        """Check latency and update passthrough state.

        Returns:
            True if the system should use passthrough mode.
        """
        if latency_ms > self.threshold_ms:
            if not self._in_passthrough:
                logger.critical(
                    "Reflexive watchdog TRIGGERED: %.2fms > %.2fms threshold. "
                    "Falling back to landmark passthrough.",
                    latency_ms,
                    self.threshold_ms,
                )
            self._in_passthrough = True
            self._consecutive_ok = 0
            return True

        if self._in_passthrough:
            self._consecutive_ok += 1
            if self._consecutive_ok >= self.recovery_frames:
                logger.info(
                    "Reflexive watchdog RECOVERED: %d consecutive frames below threshold.",
                    self._consecutive_ok,
                )
                self._in_passthrough = False
                self._consecutive_ok = 0
                return False
            # Still in recovery period
            return True

        return False


class LandmarkPassthrough:
    """Fallback: forward raw landmarks as cursor delta without model inference.

    When the watchdog triggers, we bypass the neural network entirely and convert
    raw landmark positions to cursor deltas using simple differencing.
    """

    def __init__(self) -> None:
        self._prev_x: float | None = None
        self._prev_y: float | None = None

    def process(self, landmarks: np.ndarray) -> ReflexiveResult:
        """Convert raw landmarks to cursor delta via first-order differencing.

        Args:
            landmarks: (66,) array — 21×3 landmarks + 3D velocity.
                       We use the wrist position (landmarks[0:2]) as cursor proxy.
        """
        # Wrist position (index 0) in normalized [0,1] space
        x, y = float(landmarks[0]), float(landmarks[1])

        if self._prev_x is None:
            dx, dy = 0.0, 0.0
        else:
            dx = x - self._prev_x
            dy = y - self._prev_y

        self._prev_x = x
        self._prev_y = y

        return ReflexiveResult(
            cursor_dx=dx,
            cursor_dy=dy,
            gesture_id=0,  # Can't classify without model
            gesture_confidence=0.0,
            embedding=None,
            latency_ms=0.0,
            is_passthrough=True,
        )


class ReflexiveEngine:
    """ONNX Runtime inference engine for the reflexive path.

    Runs on CPU only (per architecture rules). Includes watchdog fallback
    and optional prototypical network lookup for personalized gesture classification.
    """

    def __init__(
        self,
        onnx_path: str | Path | None = None,
        watchdog_threshold_ms: float = 8.0,
        watchdog_recovery_frames: int = 10,
    ):
        self._onnx_path = Path(onnx_path) if onnx_path else None
        self._session = None
        self._watchdog = ReflexiveWatchdog(watchdog_threshold_ms, watchdog_recovery_frames)
        self._passthrough = LandmarkPassthrough()
        self._loaded = False

    def load(self) -> None:
        """Load the ONNX model. Call once at startup."""
        if self._onnx_path is None or not self._onnx_path.exists():
            logger.warning(
                "No ONNX model at %s — engine will use passthrough mode.", self._onnx_path
            )
            return

        try:
            import onnxruntime as ort

            # CPU only — per architecture rules §8
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 2  # Don't monopolize CPU
            sess_options.inter_op_num_threads = 1

            self._session = ort.InferenceSession(
                str(self._onnx_path),
                sess_options,
                providers=["CPUExecutionProvider"],  # ALWAYS CPU
            )
            self._loaded = True
            logger.info("Reflexive ONNX model loaded from %s", self._onnx_path)

        except Exception:
            logger.exception("Failed to load reflexive ONNX model")
            self._session = None

    def infer(self, landmarks: np.ndarray) -> ReflexiveResult:
        """Run single-frame reflexive inference.

        Args:
            landmarks: (66,) float32 array — 21×3 flattened landmarks + 3D velocity

        Returns:
            ReflexiveResult with cursor delta, gesture, confidence, timing
        """
        # If no model loaded, always passthrough
        if not self._loaded or self._session is None:
            return self._passthrough.process(landmarks)

        # If watchdog is in passthrough mode, skip inference
        if self._watchdog.is_passthrough:
            result = self._passthrough.process(landmarks)
            # Still time a speculative inference to check if we should recover
            t0 = time.perf_counter()
            try:
                self._run_onnx(landmarks)
            except Exception:
                pass
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._watchdog.check(elapsed_ms)
            return result

        # Normal inference path
        t0 = time.perf_counter()
        try:
            cursor_delta, gesture_id, gesture_conf, embedding = self._run_onnx(landmarks)
        except Exception:
            logger.exception("ONNX inference failed — falling back to passthrough")
            return self._passthrough.process(landmarks)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Check watchdog
        if self._watchdog.check(elapsed_ms):
            # Triggered — discard this result, use passthrough
            return self._passthrough.process(landmarks)

        return ReflexiveResult(
            cursor_dx=float(cursor_delta[0]),
            cursor_dy=float(cursor_delta[1]),
            gesture_id=int(gesture_id),
            gesture_confidence=float(gesture_conf),
            embedding=embedding,
            latency_ms=elapsed_ms,
            is_passthrough=False,
        )

    def _run_onnx(
        self, landmarks: np.ndarray
    ) -> tuple[np.ndarray, int, float, np.ndarray]:
        """Execute ONNX inference. Returns (cursor_delta, gesture_id, confidence, embedding)."""
        assert self._session is not None

        # Prepare input — add batch dimension
        input_data = landmarks.astype(np.float32).reshape(1, -1)

        outputs = self._session.run(
            None,
            {"landmarks": input_data},
        )

        # Expected outputs: [cursor_delta, gesture_logits, embedding]
        cursor_delta = outputs[0][0]  # (2,)
        gesture_logits = outputs[1][0]  # (num_classes,)
        embedding = outputs[2][0]  # (embed_dim,)

        # Softmax for confidence
        exp_logits = np.exp(gesture_logits - gesture_logits.max())
        probs = exp_logits / exp_logits.sum()
        gesture_id = int(probs.argmax())
        gesture_conf = float(probs[gesture_id])

        return cursor_delta, gesture_id, gesture_conf, embedding
