"""Synthetic training data for the reflexive path.

Generates deterministic (landmarks, cursor_target, gesture_label) samples from
procedural hand trajectories, per ml-research.md §6: versioned by generation
parameters, deterministic given the same seed, schema-validated before training.

This is the bootstrap dataset — it validates the training pipeline end-to-end
and gives the model a non-trivial signal (gesture poses are geometrically
distinct; cursor targets follow real motion profiles). Real recorded data
replaces or augments it later without changing the DataLoader contract.

Sample layout (matches data/landmarks.py):
    features:      (66,) float32 — 21×3 normalized landmarks + 3D wrist velocity
    cursor_target: (2,)  float32 — next-frame wrist delta × CURSOR_DELTA_SCALE,
                                   clipped to [-1, 1] (model's Tanh output range)
    gesture_label: ()    int64   — GestureType enum (0=None..4=Swipe)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUM_LANDMARKS = 21
LANDMARK_DIMS = NUM_LANDMARKS * 3
FEATURE_DIMS = LANDMARK_DIMS + 3

GESTURE_NONE = 0
GESTURE_POINT = 1
GESTURE_PINCH = 2
GESTURE_GRAB = 3
GESTURE_SWIPE = 4
NUM_GESTURES = 5

# Per-frame wrist deltas at 60fps are ~±0.05 normalized units; this gain maps
# them into the Tanh-bounded [-1, 1] regression range. The inference side must
# divide by the same constant to recover normalized-space deltas.
CURSOR_DELTA_SCALE = 20.0

# Relaxed right-hand landmark template: (x, y) offsets from the wrist in
# normalized image space (y grows downward, so extended fingers are -y).
# Index order follows the MediaPipe hand model (0=wrist, 4=thumb tip, ...).
_TEMPLATE_XY = np.array(
    [
        [0.000, 0.000],   # 0  wrist
        [-0.030, -0.020], # 1  thumb cmc
        [-0.060, -0.040], # 2  thumb mcp
        [-0.080, -0.060], # 3  thumb ip
        [-0.100, -0.080], # 4  thumb tip
        [-0.020, -0.100], # 5  index mcp
        [-0.022, -0.140], # 6  index pip
        [-0.023, -0.170], # 7  index dip
        [-0.024, -0.200], # 8  index tip
        [0.000, -0.110],  # 9  middle mcp
        [0.000, -0.155],  # 10 middle pip
        [0.000, -0.190],  # 11 middle dip
        [0.000, -0.220],  # 12 middle tip
        [0.020, -0.100],  # 13 ring mcp
        [0.021, -0.140],  # 14 ring pip
        [0.022, -0.170],  # 15 ring dip
        [0.023, -0.200],  # 16 ring tip
        [0.045, -0.090],  # 17 pinky mcp
        [0.048, -0.120],  # 18 pinky pip
        [0.050, -0.140],  # 19 pinky dip
        [0.052, -0.160],  # 20 pinky tip
    ],
    dtype=np.float64,
)

_PALM_CENTER = np.array([0.0, -0.06], dtype=np.float64)

# Finger joint indices (mcp → tip) for curling
_FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


def _curl_finger(pose: np.ndarray, joints: list[int], amount: float) -> None:
    """Pull a finger's joints toward the palm center. amount ∈ [0, 1]."""
    for depth, j in enumerate(joints):
        # Curl increases along the finger: tips move most
        t = amount * (0.4 + 0.6 * depth / (len(joints) - 1))
        pose[j] = pose[j] * (1 - t) + _PALM_CENTER * t


def _gesture_pose(gesture: int) -> np.ndarray:
    """Return the (21, 2) landmark offsets for a gesture class."""
    pose = _TEMPLATE_XY.copy()

    if gesture == GESTURE_POINT:
        for name in ("middle", "ring", "pinky"):
            _curl_finger(pose, _FINGERS[name], 0.85)
        _curl_finger(pose, _FINGERS["thumb"], 0.5)
    elif gesture == GESTURE_PINCH:
        # Thumb tip and index tip meet at their midpoint
        midpoint = (pose[4] + pose[8]) / 2
        for j, w in ((3, 0.5), (4, 1.0), (7, 0.5), (8, 1.0)):
            pose[j] = pose[j] * (1 - w) + midpoint * w
        for name in ("middle", "ring", "pinky"):
            _curl_finger(pose, _FINGERS[name], 0.4)
    elif gesture == GESTURE_GRAB:
        for joints in _FINGERS.values():
            _curl_finger(pose, joints, 0.9)
    elif gesture == GESTURE_SWIPE:
        # Flat extended hand, fingers together: shrink x spread slightly
        pose[:, 0] *= 0.7

    return pose


def _min_jerk(t: np.ndarray) -> np.ndarray:
    """Minimum-jerk position profile, t ∈ [0, 1] → s ∈ [0, 1]."""
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _wrist_trajectory(
    rng: np.random.Generator, gesture: int, num_frames: int
) -> np.ndarray:
    """Generate a (num_frames, 2) wrist path in normalized screen space."""
    start = rng.uniform(0.2, 0.8, size=2)

    if gesture == GESTURE_SWIPE:
        # One fast lateral stroke
        direction = rng.choice([-1.0, 1.0])
        end = start + np.array([direction * rng.uniform(0.25, 0.45),
                                rng.uniform(-0.05, 0.05)])
        s = _min_jerk(np.linspace(0, 1, num_frames))
        return start + s[:, None] * (end - start)

    if gesture in (GESTURE_PINCH, GESTURE_GRAB):
        # Targeting behavior: slow drift toward a nearby point, then dwell
        end = start + rng.uniform(-0.08, 0.08, size=2)
        split = int(num_frames * 0.6)
        s = _min_jerk(np.linspace(0, 1, split))
        approach = start + s[:, None] * (end - start)
        dwell = np.tile(end, (num_frames - split, 1))
        return np.concatenate([approach, dwell])

    # None / Point: casual multi-waypoint motion
    num_segments = rng.integers(2, 4)
    frames_per_seg = np.full(num_segments, num_frames // num_segments)
    frames_per_seg[-1] += num_frames - frames_per_seg.sum()
    points = [start]
    for _ in range(num_segments):
        points.append(np.clip(points[-1] + rng.uniform(-0.2, 0.2, size=2), 0.1, 0.9))
    chunks = []
    for i in range(num_segments):
        s = _min_jerk(np.linspace(0, 1, frames_per_seg[i]))
        chunks.append(points[i] + s[:, None] * (points[i + 1] - points[i]))
    return np.concatenate(chunks)


@dataclass(frozen=True)
class SyntheticConfig:
    """Generation parameters — log these to W&B with every run."""

    num_sequences: int = 250
    frames_per_sequence: int = 48
    fps: float = 60.0
    landmark_noise_std: float = 0.004
    hand_scale_range: tuple[float, float] = (0.8, 1.2)
    rotation_range_deg: float = 25.0
    mirror_probability: float = 0.5  # left-hand mirroring
    seed: int = 42


def generate_reflexive_dataset(
    cfg: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the full synthetic dataset.

    Returns:
        features:       (N, 66) float32
        cursor_targets: (N, 2) float32 in [-1, 1]
        gesture_labels: (N,) int64
    """
    rng = np.random.default_rng(cfg.seed)
    dt = 1.0 / cfg.fps

    all_features: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for seq_idx in range(cfg.num_sequences):
        gesture = seq_idx % NUM_GESTURES  # balanced classes
        n = cfg.frames_per_sequence

        wrist = _wrist_trajectory(rng, gesture, n)  # (n, 2)

        # Per-sequence hand variation
        scale = rng.uniform(*cfg.hand_scale_range)
        angle = np.deg2rad(rng.uniform(-cfg.rotation_range_deg, cfg.rotation_range_deg))
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        mirror = rng.random() < cfg.mirror_probability

        pose = _gesture_pose(gesture) * scale
        if mirror:
            pose = pose * np.array([-1.0, 1.0])
        pose = pose @ rot.T  # (21, 2)

        # Absolute landmark positions per frame + noise
        landmarks_xy = (
            wrist[:, None, :]
            + pose[None, :, :]
            + rng.normal(0, cfg.landmark_noise_std, size=(n, NUM_LANDMARKS, 2))
        )  # (n, 21, 2)
        landmarks_z = rng.normal(0, 0.01, size=(n, NUM_LANDMARKS, 1))
        landmarks = np.concatenate([landmarks_xy, landmarks_z], axis=2)  # (n, 21, 3)

        # Wrist velocity (normalized units/sec), zero at sequence start —
        # matching LandmarkExtractor's differencing behavior
        wrist3 = landmarks[:, 0, :]  # noisy wrist, like a real detector sees
        velocity = np.zeros((n, 3))
        velocity[1:] = (wrist3[1:] - wrist3[:-1]) / dt

        features = np.concatenate(
            [landmarks.reshape(n, LANDMARK_DIMS), velocity], axis=1
        ).astype(np.float32)

        # Cursor target: next-frame CLEAN wrist delta (the model should learn
        # to denoise), scaled into the Tanh range. Last frame has no target.
        deltas = np.zeros((n, 2))
        deltas[:-1] = wrist[1:] - wrist[:-1]
        targets = np.clip(deltas * CURSOR_DELTA_SCALE, -1.0, 1.0).astype(np.float32)

        all_features.append(features[:-1])
        all_targets.append(targets[:-1])
        all_labels.append(np.full(n - 1, gesture, dtype=np.int64))

    features = np.concatenate(all_features)
    targets = np.concatenate(all_targets)
    labels = np.concatenate(all_labels)

    _validate_schema(features, targets, labels)
    return features, targets, labels


def _validate_schema(
    features: np.ndarray, targets: np.ndarray, labels: np.ndarray
) -> None:
    """Schema validation before training, per ml-research.md §6."""
    assert features.ndim == 2 and features.shape[1] == FEATURE_DIMS, features.shape
    assert features.dtype == np.float32, features.dtype
    assert targets.ndim == 2 and targets.shape[1] == 2, targets.shape
    assert targets.dtype == np.float32, targets.dtype
    assert np.abs(targets).max() <= 1.0, "cursor targets outside [-1, 1]"
    assert labels.ndim == 1 and labels.dtype == np.int64, (labels.shape, labels.dtype)
    assert labels.min() >= 0 and labels.max() < NUM_GESTURES
    assert features.shape[0] == targets.shape[0] == labels.shape[0]
    assert np.isfinite(features).all() and np.isfinite(targets).all()
