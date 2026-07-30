"""Guided real-data recording for the reflexive path.

Captures LandmarkExtractor features from the live C# pipeline (VideoFrameReader)
while prompting the user through each gesture class, then persists a training
dataset in the same layout as data/synthetic.py:

    features:      (N, 66) float32 — landmarks + wrist velocity
    cursor_target: (N, 2)  float32 — next-frame wrist delta × CURSOR_DELTA_SCALE
    gesture_label: (N,)    int64   — the prompted GestureType

Per ml-research.md §6: recordings are persisted as .npz + YAML metadata
(content-hashed, schema-validated). No PII — landmarks only, frames are never
stored.

Usage (C# pipeline must be running so the frame arena exists):

    .venv/Scripts/python -m GrappleIntent.data.recorder --user-id you \
        --seconds-per-gesture 10 --passes 2
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .synthetic import (
    CURSOR_DELTA_SCALE,
    FEATURE_DIMS,
    NUM_GESTURES,
    _validate_schema,
)

logger = logging.getLogger(__name__)

RECORDING_SCHEMA_VERSION = 1

GESTURE_NAMES = {
    0: "NONE (relaxed hand, move around casually)",
    1: "POINT (index finger extended, move to targets)",
    2: "PINCH (thumb + index together, hold and move)",
    3: "GRAB (closed fist, hold and move)",
    4: "SWIPE (flat hand, lateral strokes)",
}


@dataclass(frozen=True)
class RecordingConfig:
    """Protocol parameters — persisted into the YAML sidecar."""

    seconds_per_gesture: float = 10.0
    rest_seconds: float = 3.0
    gestures: tuple[int, ...] = (0, 1, 2, 3, 4)
    passes: int = 2
    # Cursor targets are only formed between consecutive detections closer
    # than this — a detection gap must not fabricate a huge delta.
    max_frame_gap_s: float = 0.15
    # Segments with fewer usable samples than this are dropped entirely.
    min_segment_frames: int = 5


@dataclass
class GestureSegment:
    """Frames captured during one prompted gesture window."""

    gesture: int
    features: list[np.ndarray] = field(default_factory=list)  # (66,) copies
    wrists: list[np.ndarray] = field(default_factory=list)  # (2,) copies
    times: list[float] = field(default_factory=list)  # seconds (QPC-derived)

    def __len__(self) -> int:
        return len(self.features)


def segment_to_samples(
    segment: GestureSegment, max_frame_gap_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn one segment into (features, cursor_targets, labels) arrays.

    The cursor target for frame i is the wrist delta to frame i+1 scaled by
    CURSOR_DELTA_SCALE (matching data/synthetic.py); frames followed by a
    detection gap larger than max_frame_gap_s are dropped, as is the last
    frame (no next-frame target exists).
    """
    n = len(segment)
    if n < 2:
        return (
            np.zeros((0, FEATURE_DIMS), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    keep: list[int] = []
    targets: list[np.ndarray] = []
    for i in range(n - 1):
        gap = segment.times[i + 1] - segment.times[i]
        if gap <= 0 or gap > max_frame_gap_s:
            continue
        delta = segment.wrists[i + 1] - segment.wrists[i]
        targets.append(
            np.clip(delta * CURSOR_DELTA_SCALE, -1.0, 1.0).astype(np.float32)
        )
        keep.append(i)

    if not keep:
        return (
            np.zeros((0, FEATURE_DIMS), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    features = np.stack([segment.features[i] for i in keep]).astype(np.float32)
    cursor_targets = np.stack(targets)
    labels = np.full(len(keep), segment.gesture, dtype=np.int64)
    return features, cursor_targets, labels


def segments_to_arrays(
    segments: list[GestureSegment], config: RecordingConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble all segments into schema-validated training arrays."""
    feats, targs, labs = [], [], []
    for seg in segments:
        f, t, l = segment_to_samples(seg, config.max_frame_gap_s)
        if f.shape[0] < config.min_segment_frames:
            if len(seg) > 0:
                logger.warning(
                    "Dropping %s segment: only %d usable samples "
                    "(hand lost or too many gaps?)",
                    GESTURE_NAMES.get(seg.gesture, seg.gesture), f.shape[0],
                )
            continue
        feats.append(f)
        targs.append(t)
        labs.append(l)

    if not feats:
        raise RuntimeError(
            "No usable samples recorded — was a hand visible to the camera?"
        )

    features = np.concatenate(feats)
    targets = np.concatenate(targs)
    labels = np.concatenate(labs)
    _validate_schema(features, targets, labels)
    return features, targets, labels


class GestureRecorder:
    """Runs the guided protocol against an injected frame source + extractor.

    frame_source: callable() -> (frame_rgb | None, time_s) — the live wrapper
        polls VideoFrameReader and converts QPC ticks to seconds.
    extractor: LandmarkExtractor-compatible (extract(frame, time_s) -> LandmarkFrame | None)
    """

    def __init__(
        self,
        frame_source,
        extractor,
        config: RecordingConfig,
        clock=time.monotonic,
        prompt=print,
        idle_sleep_s: float = 0.002,
    ) -> None:
        self._source = frame_source
        self._extractor = extractor
        self._config = config
        self._clock = clock
        self._prompt = prompt
        self._idle_sleep_s = idle_sleep_s

    def record_segment(self, gesture: int, seconds: float) -> GestureSegment:
        """Capture one gesture window. Features are copied out of the
        extractor's reused buffer."""
        segment = GestureSegment(gesture=gesture)
        deadline = self._clock() + seconds

        while self._clock() < deadline:
            frame, time_s = self._source()
            if frame is None:
                if self._idle_sleep_s:
                    time.sleep(self._idle_sleep_s)
                continue
            lm = self._extractor.extract(frame, time_s)
            if lm is None:
                continue
            segment.features.append(lm.features.copy())
            segment.wrists.append(
                np.array([lm.wrist_x, lm.wrist_y], dtype=np.float64)
            )
            segment.times.append(time_s)

        return segment

    def run_protocol(self) -> list[GestureSegment]:
        cfg = self._config
        segments: list[GestureSegment] = []
        total = len(cfg.gestures) * cfg.passes

        step = 0
        for pass_idx in range(cfg.passes):
            for gesture in cfg.gestures:
                step += 1
                self._prompt(
                    f"\n[{step}/{total}] Get ready: "
                    f"{GESTURE_NAMES.get(gesture, gesture)}"
                )
                self._rest(cfg.rest_seconds)
                self._prompt(
                    f"  RECORDING {cfg.seconds_per_gesture:.0f}s — perform the gesture now"
                )
                seg = self.record_segment(gesture, cfg.seconds_per_gesture)
                self._prompt(f"  captured {len(seg)} detections")
                segments.append(seg)

        return segments

    def _rest(self, seconds: float) -> None:
        # Countdown so the user can reposition; also drain stale frames.
        deadline = self._clock() + seconds
        while self._clock() < deadline:
            self._source()  # keep consuming so recording starts fresh
            if self._idle_sleep_s:
                time.sleep(self._idle_sleep_s)


# ─── Persistence (.npz + YAML sidecar with content hash) ─────────────────────


def save_recording(
    output_path: str | Path,
    features: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    metadata: dict,
) -> tuple[Path, Path]:
    """Save arrays as .npz and a YAML sidecar containing its sha256.

    Returns (npz_path, yaml_path).
    """
    import yaml

    _validate_schema(features, targets, labels)

    npz_path = Path(output_path).with_suffix(".npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        features=features,
        cursor_targets=targets,
        gesture_labels=labels,
        schema_version=np.int64(RECORDING_SCHEMA_VERSION),
    )

    sha256 = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    class_counts = {
        int(g): int((labels == g).sum()) for g in np.unique(labels)
    }
    sidecar = {
        "schema_version": RECORDING_SCHEMA_VERSION,
        "content_sha256": sha256,
        "num_samples": int(features.shape[0]),
        "class_counts": class_counts,
        **metadata,
    }
    yaml_path = npz_path.with_suffix(".yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(sidecar, f, sort_keys=False)

    logger.info(
        "Saved %d samples to %s (sha256=%s)", features.shape[0], npz_path, sha256[:12]
    )
    return npz_path, yaml_path


def load_recording(
    npz_path: str | Path, verify_hash: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load a recorded .npz, verifying schema and (optionally) content hash.

    Returns (features, cursor_targets, gesture_labels, sidecar_metadata).
    """
    import yaml

    npz_path = Path(npz_path)
    yaml_path = npz_path.with_suffix(".yaml")

    meta: dict = {}
    if yaml_path.exists():
        with open(yaml_path) as f:
            meta = yaml.safe_load(f) or {}
        if verify_hash:
            actual = hashlib.sha256(npz_path.read_bytes()).hexdigest()
            expected = meta.get("content_sha256")
            if expected and actual != expected:
                raise ValueError(
                    f"Content hash mismatch for {npz_path}: "
                    f"expected {expected[:12]}..., got {actual[:12]}... "
                    "(file modified after recording?)"
                )
    elif verify_hash:
        logger.warning("No YAML sidecar for %s — cannot verify hash", npz_path)

    with np.load(npz_path) as data:
        features = data["features"]
        targets = data["cursor_targets"]
        labels = data["gesture_labels"]

    _validate_schema(features, targets, labels)
    return features, targets, labels, meta


# ─── Live capture wiring ─────────────────────────────────────────────────────


def _live_frame_source(reader):
    """Wrap VideoFrameReader into the recorder's frame_source contract."""
    qpc_freq = reader.qpc_frequency

    def source():
        reader.wait_for_frame(timeout_ms=50)
        frame, timestamp = reader.read_latest_frame()
        if frame is None:
            return None, 0.0
        time_s = (timestamp / qpc_freq) if qpc_freq else time.perf_counter()
        return frame, time_s

    return source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Guided gesture recording from the live Grapple pipeline"
    )
    parser.add_argument("--user-id", default="user0", help="Tag for the output filename")
    parser.add_argument("--output-dir", default="data/recordings")
    parser.add_argument("--seconds-per-gesture", type=float, default=10.0)
    parser.add_argument("--rest-seconds", type=float, default=3.0)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--wait-seconds", type=float, default=15.0,
                        help="How long to wait for the C# frame arena")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    from ..integration.arena_bridge import _CFG as GRAPPLE_CFG
    from ..integration.arena_bridge import VideoFrameReader
    from .landmarks import LandmarkExtractor

    reader = VideoFrameReader()
    deadline = time.monotonic() + args.wait_seconds
    while not reader.open():
        if time.monotonic() >= deadline:
            logger.error(
                "Video frame arena unavailable after %.0fs — start the C# "
                "pipeline first (dotnet run --project src/GrappleV2/Grapple.Service)",
                args.wait_seconds,
            )
            return 1
        time.sleep(0.5)

    mp_cfg = GRAPPLE_CFG.get("mediapipe", {})
    extractor = LandmarkExtractor(
        max_hands=mp_cfg.get("maxHands", 1),
        min_detection_confidence=mp_cfg.get("minDetectionConfidence", 0.5),
        min_tracking_confidence=mp_cfg.get("minTrackingConfidence", 0.4),
        model_complexity=mp_cfg.get("modelComplexity", 0),
    )

    config = RecordingConfig(
        seconds_per_gesture=args.seconds_per_gesture,
        rest_seconds=args.rest_seconds,
        passes=args.passes,
    )

    try:
        recorder = GestureRecorder(_live_frame_source(reader), extractor, config)
        print(
            f"\nRecording protocol: {len(config.gestures)} gestures x "
            f"{config.passes} passes x {config.seconds_per_gesture:.0f}s each. "
            "Keep your hand in view of the camera."
        )
        segments = recorder.run_protocol()
        features, targets, labels = segments_to_arrays(segments, config)
    finally:
        extractor.close()
        reader.close()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir) / f"recorded_{args.user_id}_{stamp}"
    metadata = {
        "source": "recorded",
        "user_id": args.user_id,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cursor_delta_scale": CURSOR_DELTA_SCALE,
        "protocol": {
            "seconds_per_gesture": config.seconds_per_gesture,
            "rest_seconds": config.rest_seconds,
            "gestures": list(config.gestures),
            "passes": config.passes,
            "max_frame_gap_s": config.max_frame_gap_s,
            "min_segment_frames": config.min_segment_frames,
        },
        "mediapipe": mp_cfg,
    }
    npz_path, yaml_path = save_recording(out, features, targets, labels, metadata)

    counts = {GESTURE_NAMES[g].split(" ")[0]: int((labels == g).sum())
              for g in range(NUM_GESTURES) if (labels == g).any()}
    print(f"\nDone: {features.shape[0]} samples -> {npz_path}")
    print(f"Per-class counts: {counts}")
    print(f"Train with: python -m GrappleIntent.training.train_reflexive "
          f"--recorded-data {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
