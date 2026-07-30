"""MediaPipe hand landmark extraction — the frame → feature-vector stage.

GrappleIntent replaces GrappleDetector.py as the sensor producer, so it runs
MediaPipe itself. The MediaPipe configuration mirrors GrappleDetector.py /
python-vision.md (lite model, single hand, low thresholds for speed).

Output contract (consumed by ReflexiveEngine and the training pipeline):
    (66,) float32 = 21 landmarks × 3 normalized coords + 3D wrist velocity
    Layout: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20, vx, vy, vz]
    Velocity is in normalized units/sec via first-order differencing of the
    wrist landmark between consecutive detections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

NUM_LANDMARKS = 21
LANDMARK_DIMS = NUM_LANDMARKS * 3  # 63
FEATURE_DIMS = LANDMARK_DIMS + 3  # 66 — landmarks + velocity
WRIST_INDEX = 0


@dataclass
class LandmarkFrame:
    """One frame of extracted hand features.

    `features` is a view into a buffer reused across frames — copy it if you
    need to hold onto it past the next extract() call.
    """

    features: np.ndarray  # (66,) float32
    wrist_x: float
    wrist_y: float
    wrist_z: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    handedness: str  # "Left" or "Right"
    detection_confidence: float


class LandmarkExtractor:
    """Runs MediaPipe Hands on RGB frames and emits the (66,) feature vector."""

    def __init__(
        self,
        max_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.4,
        model_complexity: int = 0,
    ) -> None:
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )
        self._features = np.zeros(FEATURE_DIMS, dtype=np.float32)
        self._prev_wrist: np.ndarray | None = None
        self._prev_time_s: float | None = None

    def extract(self, frame_rgb: np.ndarray, time_s: float) -> LandmarkFrame | None:
        """Extract hand features from an RGB frame.

        Args:
            frame_rgb: (H, W, 3) uint8 RGB frame.
            time_s: Capture time in seconds (monotonic — e.g. QPC ticks / freq).

        Returns:
            LandmarkFrame, or None if no hand was detected.
        """
        results = self._hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            # Velocity differencing must not bridge a detection gap
            self._prev_wrist = None
            self._prev_time_s = None
            return None

        hand = results.multi_hand_landmarks[0]

        handedness = "Right"
        confidence = 1.0
        if results.multi_handedness:
            classification = results.multi_handedness[0].classification[0]
            handedness = classification.label
            confidence = classification.score

        out = self._features
        for i, lm in enumerate(hand.landmark):
            base = i * 3
            out[base] = lm.x
            out[base + 1] = lm.y
            out[base + 2] = lm.z

        wrist = out[WRIST_INDEX * 3 : WRIST_INDEX * 3 + 3].copy()

        if self._prev_wrist is not None and self._prev_time_s is not None:
            dt = time_s - self._prev_time_s
            if dt > 1e-6:
                out[LANDMARK_DIMS : LANDMARK_DIMS + 3] = (wrist - self._prev_wrist) / dt
            else:
                out[LANDMARK_DIMS : LANDMARK_DIMS + 3] = 0.0
        else:
            out[LANDMARK_DIMS : LANDMARK_DIMS + 3] = 0.0

        self._prev_wrist = wrist
        self._prev_time_s = time_s

        return LandmarkFrame(
            features=out,
            wrist_x=float(wrist[0]),
            wrist_y=float(wrist[1]),
            wrist_z=float(wrist[2]),
            velocity_x=float(out[LANDMARK_DIMS]),
            velocity_y=float(out[LANDMARK_DIMS + 1]),
            velocity_z=float(out[LANDMARK_DIMS + 2]),
            handedness=handedness,
            detection_confidence=confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> LandmarkExtractor:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
