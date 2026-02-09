"""
Grapple Detector - MediaPipe Hands Inference Pipeline
Zero-copy consumption of frames from C# producer.
Writes hand tracking results back to C# via shared memory.

Dual-write: Outputs both legacy struct format (backward compat)
and FlatBuffer SensorFrame (Phase 2 protocol).

Dependencies: pip install mediapipe numpy flatbuffers
"""

import os
import sys
import argparse

# Silence TensorFlow/MediaPipe warnings BEFORE importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Kill TF warnings (0=all, 1=info, 2=warn, 3=error only)
os.environ['GLOG_minloglevel'] = '3'      # Kill glog warnings

import mmap
import struct
import ctypes
from ctypes import wintypes
import numpy as np
import mediapipe as mp

# === Windows API ===
kernel32 = ctypes.windll.kernel32

SYNCHRONIZE = 0x00100000
EVENT_MODIFY_STATE = 0x0002
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102

QueryPerformanceCounter = kernel32.QueryPerformanceCounter
QueryPerformanceCounter.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
QueryPerformanceCounter.restype = wintypes.BOOL

# For creating/setting events (hand results signaling)
CreateEventW = kernel32.CreateEventW
CreateEventW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
CreateEventW.restype = wintypes.HANDLE

SetEvent = kernel32.SetEvent
SetEvent.argtypes = [wintypes.HANDLE]
SetEvent.restype = wintypes.BOOL


import json


def get_qpc() -> int:
    """Get QueryPerformanceCounter value (matches C# Stopwatch.GetTimestamp())"""
    counter = ctypes.c_longlong()
    QueryPerformanceCounter(ctypes.byref(counter))
    return counter.value


# === CONFIGURATION LOADING ===
# Loads grapple_config.json (shared with C#). Falls back to defaults if missing.

def _load_config() -> dict:
    """Load grapple_config.json, searching upward from script directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_dir = script_dir
    for _ in range(8):
        candidate = os.path.join(search_dir, "grapple_config.json")
        if os.path.exists(candidate):
            with open(candidate, 'r') as f:
                print(f"[Config] Loaded from {candidate}")
                return json.load(f)
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break
        search_dir = parent
    print("[Config] No grapple_config.json found. Using defaults.")
    return {}

_CFG = _load_config()

# === Frame Arena Constants (MUST MATCH C#) ===
_frame_cfg = _CFG.get("arenas", {}).get("frame", {})
_frame_signal_cfg = _CFG.get("arenas", {}).get("frameSignal", {})
MAP_NAME = _frame_cfg.get("mapName", "Local\\GrappleMap")
EVENT_NAME = _frame_signal_cfg.get("signalName", "Local\\GrappleSignal")
_cap_mb = _frame_cfg.get("capacityMB", 256)
MAP_CAPACITY = _cap_mb * 1024 * 1024
FIRST_SLOT_OFFSET = 1024  # Protocol constant
METADATA_SIZE = 64  # Protocol constant

_webcam_cfg = _CFG.get("webcam", {})
WIDTH = _webcam_cfg.get("width", 1920)
HEIGHT = _webcam_cfg.get("height", 1080)
CHANNELS = 3  # Protocol constant (RGB24)
FRAME_SIZE = WIDTH * HEIGHT * CHANNELS

HEADER_FORMAT = '<Qiiqiiq'
HEADER_STRUCT_SIZE = struct.calcsize(HEADER_FORMAT)
METADATA_FORMAT = '<qi'

# === Hand Result Arena Constants (MUST MATCH C#) ===
_hand_cfg = _CFG.get("arenas", {}).get("hand", {})
HAND_MAP_NAME = _hand_cfg.get("mapName", "Local\\GrappleHandResults")
HAND_EVENT_NAME = _hand_cfg.get("signalName", "Local\\GrappleHandSignal")
HAND_MAP_SIZE = _hand_cfg.get("capacityBytes", 4096)
HAND_DATA_OFFSET = 64  # Protocol constant
HAND_MAGIC = 0x48414E4447525043  # Protocol constant

# Protocol version (CV-2 fix: MUST MATCH C# CurrentProtocolVersion)
PROTOCOL_VERSION = 1  # Protocol constant

# Updated format with velocity: 5×double (x,y,z,vx,vy), int, float, long = 56 bytes
HAND_STATE_FORMAT = '<dddddifq'
HAND_STATE_SIZE = struct.calcsize(HAND_STATE_FORMAT)

# === FlatBuffer Sensor Arena Constants (Phase 2 Protocol) ===
_sensor_cfg = _CFG.get("arenas", {}).get("sensor", {})
SENSOR_MAP_NAME = _sensor_cfg.get("mapName", "Local\\GrappleSensorArena")
SENSOR_EVENT_NAME = _sensor_cfg.get("signalName", "Local\\GrappleSensorSignal")
SENSOR_MAP_SIZE = _sensor_cfg.get("capacityBytes", 8192)
SENSOR_DATA_OFFSET = 64  # Protocol constant
SENSOR_MAGIC = 0x4C505247  # Protocol constant
SENSOR_PROTOCOL_VERSION = 2  # Protocol constant

# FlatBufferArenaHeader format: magic(Q) + sequence(q) + version(i) + bufferSize(i) + freq(q)
SENSOR_HEADER_FORMAT = '<QqiiQ'
SENSOR_HEADER_SIZE = struct.calcsize(SENSOR_HEADER_FORMAT)  # 32 bytes

# Add generated FlatBuffers modules to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GENERATED_DIR = os.path.join(_SCRIPT_DIR, "generated")
if _GENERATED_DIR not in sys.path:
    sys.path.insert(0, _GENERATED_DIR)

import flatbuffers
from Grapple.Protocol import HandState as FBHandState
from Grapple.Protocol import SensorFrame as FBSensorFrame
from Grapple.Protocol import GestureType as FBGestureType

# === ONE-EURO FILTER FOR LANDMARK SMOOTHING ===
# Smooths raw landmark coordinates BEFORE computing pinch distance.
# This is critical: smooth input → stable derived metrics.

import math

class LowPass:
    """Simple exponential low-pass filter."""
    def __init__(self, alpha, init=None):
        self.alpha = alpha
        self.s = init
    
    def __call__(self, x, alpha=None):
        x = float(x)
        if alpha is None:
            alpha = self.alpha
        a = float(alpha)
        self.s = x if self.s is None else (a * x + (1 - a) * float(self.s))
        return float(self.s)


def _alpha(dt, cutoff):
    """Compute smoothing factor alpha from cutoff frequency and time delta."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / max(1e-6, dt))


class OneEuroFilter:
    """
    1€ Filter (Casiez et al., CHI 2012) - adaptive low-pass filter.
    Smooth when slow, responsive when fast.
    """
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_f = LowPass(_alpha(1/60.0, d_cutoff))
        self.x_f = LowPass(_alpha(1/60.0, min_cutoff))
    
    def __call__(self, x, dt):
        if self.x_prev is None:
            self.x_prev = x
        dx = (x - self.x_prev) / max(1e-6, dt)
        edx = self.dx_f(dx, _alpha(dt, self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        x_hat = self.x_f(x, _alpha(dt, cutoff))
        self.x_prev = x_hat
        return x_hat


# === PER-LANDMARK FILTER BANK ===
# 21 landmarks × 3 coordinates (x, y, z) = 63 filters per hand

# Filter tuning (from main_aaryan.py - proven to work)
_lm_cfg = _CFG.get("landmarkFilter", {})
LM_MIN_CUTOFF = _lm_cfg.get("minCutoff", 10.0)
LM_BETA = _lm_cfg.get("beta", 2.1)
LM_D_CUTOFF = _lm_cfg.get("dCutoff", 2.5)

# Global filter storage (keyed by handedness)
_lm_filters = {}

def _get_filters_for_hand(handedness: str):
    """Get or create 63 OneEuroFilters for a hand (21 landmarks × 3 coords)."""
    if handedness not in _lm_filters:
        _lm_filters[handedness] = [
            [OneEuroFilter(LM_MIN_CUTOFF, LM_BETA, LM_D_CUTOFF) for _ in range(3)]
            for _ in range(21)
        ]
    return _lm_filters[handedness]


def smooth_landmarks(landmarks, handedness: str, dt: float):
    """
    Apply OneEuroFilter to each landmark coordinate.
    Returns list of (x, y, z) tuples with smoothed values.
    """
    filters = _get_filters_for_hand(handedness)
    smoothed = []
    for i, lm in enumerate(landmarks):
        fx, fy, fz = filters[i]
        sx = fx(lm.x, dt)
        sy = fy(lm.y, dt)
        sz = fz(lm.z, dt)
        smoothed.append((sx, sy, sz))
    return smoothed


# === INDEX-FINGER NORMALIZED PINCH DETECTION ===
# Uses index finger length (MCP→tip) as scale reference.
# This is MORE STABLE than thumb segment because:
# 1. Index finger is always extended (pointing finger)
# 2. Doesn't collapse under pinching like thumb does
# 3. Roughly constant length regardless of hand pose

def index_finger_length_sq(smoothed) -> float:
    """Squared 3D length of index finger (MCP 5 → tip 8) - stable reference."""
    mcp = smoothed[5]
    tip = smoothed[8]
    dx = tip[0] - mcp[0]
    dy = tip[1] - mcp[1]
    dz = tip[2] - mcp[2]
    return dx*dx + dy*dy + dz*dz


def pinch_distance_sq(smoothed) -> float:
    """Squared 3D distance between thumb tip (4) and index tip (8)."""
    thumb = smoothed[4]
    index = smoothed[8]
    dx = thumb[0] - index[0]
    dy = thumb[1] - index[1]
    dz = thumb[2] - index[2]
    return dx*dx + dy*dy + dz*dz


# === TUNING PARAMETERS (3D Mode) ===
# Config-backed pinch detection parameters (from grapple_config.json)
_pinch_cfg = _CFG.get("pinch", {})
PINCH_THRESHOLD = _pinch_cfg.get("enterThreshold", 0.30)
PINCH_OPEN_THRESHOLD = _pinch_cfg.get("exitThreshold", 0.70)
PINCH_EXIT_MS = _pinch_cfg.get("exitDebounceMs", 150)
RATIO_ALPHA_DOWN = _pinch_cfg.get("ratioAlphaDown", 0.5)
RATIO_ALPHA_UP = _pinch_cfg.get("ratioAlphaUp", 0.3)
NO_HAND_GRACE = _pinch_cfg.get("noHandGraceFrames", 5)
MIN_SCALE_THRESHOLD = _pinch_cfg.get("minScaleThreshold", 0.001)
MAX_RATIO = _pinch_cfg.get("maxRatio", 0.9)
VELOCITY_SMOOTH = _pinch_cfg.get("velocitySmooth", 0.3)


# === PINCH STATE MACHINE (Phase 2) ===
# Replaces simple Schmitt trigger with 4-state FSM for noise rejection
class PinchStateMachine:
    """
    4-State Finite State Machine for pinch detection with frame-based confirmation.

    States:
    - OPEN: Fingers apart, not pinching
    - APPROACHING: Fingers getting close, awaiting confirmation
    - PINCHED: Confirmed pinch, mouse button down
    - RELEASING: Fingers opening, awaiting confirmation

    This eliminates spurious clicks from hand jitter by requiring sustained
    threshold crossings (3 frames to enter, 5 frames to exit).
    """

    # State constants
    STATE_OPEN = "OPEN"
    STATE_APPROACHING = "APPROACHING"
    STATE_PINCHED = "PINCHED"
    STATE_RELEASING = "RELEASING"

    def __init__(self, enter_confirm=None, exit_confirm=None):
        self.ENTER_CONFIRM_FRAMES = enter_confirm if enter_confirm is not None else _pinch_cfg.get("enterConfirmFrames", 3)
        self.EXIT_CONFIRM_FRAMES = exit_confirm if exit_confirm is not None else _pinch_cfg.get("exitConfirmFrames", 5)
        self.state = self.STATE_OPEN
        self.entry_frames = 0
        self.exit_frames = 0
        self.state_changed = False

    def update(self, ratio: float, enter_threshold: float, exit_threshold: float) -> bool:
        """
        Update state machine with current ratio.

        Args:
            ratio: Current smoothed pinch ratio (pinch_dist² / index_finger_length²)
            enter_threshold: Ratio below which to enter pinch (e.g., 0.22)
            exit_threshold: Ratio above which to exit pinch (e.g., 0.45)

        Returns:
            bool: True if pinching (PINCHED or RELEASING states), False otherwise
        """
        self.state_changed = False

        if self.state == self.STATE_OPEN:
            if ratio < enter_threshold:
                self.entry_frames += 1
                if self.entry_frames >= self.ENTER_CONFIRM_FRAMES:
                    self.state = self.STATE_PINCHED
                    self.entry_frames = 0
                    self.state_changed = True
                else:
                    self.state = self.STATE_APPROACHING
            else:
                self.entry_frames = 0

        elif self.state == self.STATE_APPROACHING:
            if ratio < enter_threshold:
                self.entry_frames += 1
                if self.entry_frames >= self.ENTER_CONFIRM_FRAMES:
                    self.state = self.STATE_PINCHED
                    self.entry_frames = 0
                    self.state_changed = True
            else:
                # Ratio went back above threshold - abort entry
                self.state = self.STATE_OPEN
                self.entry_frames = 0

        elif self.state == self.STATE_PINCHED:
            if ratio > exit_threshold:
                self.exit_frames += 1
                if self.exit_frames >= self.EXIT_CONFIRM_FRAMES:
                    self.state = self.STATE_OPEN
                    self.exit_frames = 0
                    self.state_changed = True
                else:
                    self.state = self.STATE_RELEASING
            else:
                self.exit_frames = 0

        elif self.state == self.STATE_RELEASING:
            if ratio > exit_threshold:
                self.exit_frames += 1
                if self.exit_frames >= self.EXIT_CONFIRM_FRAMES:
                    self.state = self.STATE_OPEN
                    self.exit_frames = 0
                    self.state_changed = True
            else:
                # Ratio dipped back below threshold - abort exit
                self.state = self.STATE_PINCHED
                self.exit_frames = 0

        # Return True if in pinching state (button should be down)
        return self.state in (self.STATE_PINCHED, self.STATE_RELEASING)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Grapple Hand Detector')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug logging')
    args = parser.parse_args()

    # Startup banner (always show)
    print("=== Grapple Detector (MediaPipe Hands) ===")
    print("[*] MODE: 3D Pinch Detection with State Machine")
    print(f"[*] Thresholds: ENTER<{PINCH_THRESHOLD:.2f}, EXIT>{PINCH_OPEN_THRESHOLD:.2f}")
    print("[!] Watch [Calibration] logs to tune thresholds")

    if args.debug:
        print(f"[*] Time-based exit debounce: {PINCH_EXIT_MS}ms, Ratio smoothing: down={RATIO_ALPHA_DOWN}/up={RATIO_ALPHA_UP}")
        print(f"[*] Per-landmark OneEuroFilter: cutoff={LM_MIN_CUTOFF}, beta={LM_BETA}")
        print(f"[*] HandState format: {HAND_STATE_SIZE} bytes (with velocity)")

    # === 1. Open Frame Signal Event ===
    if args.debug:
        print(f"[*] Opening event: {EVENT_NAME}")
    event_handle = kernel32.OpenEventW(
        SYNCHRONIZE | EVENT_MODIFY_STATE,
        False,
        EVENT_NAME
    )
    if not event_handle:
        print(f"[!] ERROR: Failed to open event. Is C# producer running?")
        print(f"    Error code: {kernel32.GetLastError()}")
        return

    # === 2. Map Frame Shared Memory ===
    try:
        shm = mmap.mmap(-1, MAP_CAPACITY, tagname=MAP_NAME, access=mmap.ACCESS_READ)
    except Exception as e:
        print(f"[!] ERROR: Failed to map memory: {e}")
        kernel32.CloseHandle(event_handle)
        return

    # === 3. Read Frame Arena Header ===
    header_bytes = shm[:HEADER_STRUCT_SIZE]
    magic, slot_count, slot_size, write_head, published_id, _pad, freq = struct.unpack(
        HEADER_FORMAT, header_bytes
    )

    expected_magic = 0x31454C5050415247
    if magic != expected_magic:
        print(f"[!] ERROR: Invalid magic number!")
        shm.close()
        kernel32.CloseHandle(event_handle)
        return

    if args.debug:
        print(f"[+] Arena: {slot_count} slots × {slot_size // (1024*1024)}MB")
        print(f"[*] QPC Frequency: {freq:,} ticks/sec")

    # === 4. Setup Hand Result Arena (Write) ===
    try:
        hand_shm = mmap.mmap(-1, HAND_MAP_SIZE, tagname=HAND_MAP_NAME, access=mmap.ACCESS_WRITE)
    except Exception as e:
        print(f"[!] ERROR: Failed to create hand result memory: {e}")
        shm.close()
        kernel32.CloseHandle(event_handle)
        return
    
    # Create hand signal event (AutoReset = False for bManualReset, initially non-signaled)
    hand_event_handle = CreateEventW(None, False, False, HAND_EVENT_NAME)
    if not hand_event_handle:
        print(f"[!] ERROR: Failed to create hand signal event. Error: {kernel32.GetLastError()}")
        hand_shm.close()
        shm.close()
        kernel32.CloseHandle(event_handle)
        return

    # Initialize hand result header if needed (CV-2 fix: added protocol version)
    hand_shm.seek(0)
    existing_magic = struct.unpack('<Q', hand_shm.read(8))[0]
    if existing_magic != HAND_MAGIC:
        hand_shm.seek(0)
        hand_shm.write(struct.pack('<Q', HAND_MAGIC))    # Magic at offset 0
        hand_shm.write(struct.pack('<q', 0))             # Sequence at offset 8
        hand_shm.write(struct.pack('<i', PROTOCOL_VERSION))  # ProtocolVersion at offset 16
        hand_shm.write(struct.pack('<i', 0))             # Padding at offset 20

    # === 4b. Setup FlatBuffer Sensor Arena (Phase 2 dual-write) ===
    try:
        sensor_shm = mmap.mmap(-1, SENSOR_MAP_SIZE, tagname=SENSOR_MAP_NAME, access=mmap.ACCESS_WRITE)
    except Exception as e:
        print(f"[!] ERROR: Failed to create sensor arena memory: {e}")
        kernel32.CloseHandle(hand_event_handle)
        hand_shm.close()
        shm.close()
        kernel32.CloseHandle(event_handle)
        return

    sensor_event_handle = CreateEventW(None, False, False, SENSOR_EVENT_NAME)
    if not sensor_event_handle:
        print(f"[!] ERROR: Failed to create sensor signal event. Error: {kernel32.GetLastError()}")
        sensor_shm.close()
        kernel32.CloseHandle(hand_event_handle)
        hand_shm.close()
        shm.close()
        kernel32.CloseHandle(event_handle)
        return

    # Initialize sensor arena header
    sensor_shm.seek(0)
    existing_sensor_magic = struct.unpack('<Q', sensor_shm.read(8))[0]
    if existing_sensor_magic != SENSOR_MAGIC:
        sensor_shm.seek(0)
        sensor_shm.write(struct.pack(
            SENSOR_HEADER_FORMAT,
            SENSOR_MAGIC,           # magic
            0,                      # sequence
            SENSOR_PROTOCOL_VERSION,# version
            0,                      # bufferSize (updated per frame)
            freq                    # QPC frequency
        ))

    # Pre-allocate FlatBufferBuilder (reused across frames, avoids per-frame allocation)
    fb_builder = flatbuffers.Builder(512)

    print("[+] FlatBuffer Sensor Arena initialized (dual-write active)")

    # === 5. Initialize MediaPipe (config-backed) ===
    _mp_cfg = _CFG.get("mediapipe", {})
    mp_hands = mp.solutions.hands
    try:
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=_mp_cfg.get("maxHands", 1),
            min_detection_confidence=_mp_cfg.get("minDetectionConfidence", 0.5),
            min_tracking_confidence=_mp_cfg.get("minTrackingConfidence", 0.4),
            model_complexity=_mp_cfg.get("modelComplexity", 0)
        )
    except Exception as e:
        print(f"[!] ERROR: Failed to initialize MediaPipe: {e}")
        kernel32.CloseHandle(sensor_event_handle)
        sensor_shm.close()
        kernel32.CloseHandle(hand_event_handle)
        hand_shm.close()
        shm.close()
        kernel32.CloseHandle(event_handle)
        return

    print("[+] Detector Ready")
    if args.debug:
        print(f"    Model: Lite (complexity=0), max_hands=1")

    # === 6. Inference Loop ===
    frame_count = 0
    last_buffer_id = -1
    skipped_frames = 0
    total_inference_ms = 0.0
    total_latency_ms = 0.0
    sequence = 0
    frame = None
    
    # === PINCH STATE (Phase 2: State Machine) ===
    pinch_fsm = PinchStateMachine()
    last_frame_qpc = get_qpc()
    smoothed_ratio = 1.0  # Start open
    no_hand_streak = 0     # Consecutive frames without hand detection

    # === VELOCITY TRACKING ===
    prev_x, prev_y = None, None
    prev_time_sec = None
    velocity_x, velocity_y = 0.0, 0.0
    
    print("[*] Entering inference loop... Press Ctrl+C to quit.")
    
    try:
        while True:
            # 6a. Wait for frame signal
            result = kernel32.WaitForSingleObject(event_handle, 1000)
            
            if result == WAIT_TIMEOUT:
                continue
            elif result != WAIT_OBJECT_0:
                print(f"[!] WaitForSingleObject failed: {result}")
                break
            
            # 6b. Read PublishedBufferId from header
            shm.seek(24)
            published_id = struct.unpack('<i', shm.read(4))[0]
            
            if published_id == -1:
                continue
            
            if published_id == last_buffer_id:
                continue
            
            # 6c. Detect skipped frames
            if last_buffer_id != -1:
                expected_next = (last_buffer_id + 1) % slot_count
                if published_id != expected_next:
                    skipped_frames += 1
            
            last_buffer_id = published_id
            
            # 6d. Calculate slot offset
            slot_offset = FIRST_SLOT_OFFSET + (published_id * slot_size)
            
            # 6e. Read timestamp from slot metadata
            shm.seek(slot_offset)
            meta_bytes = shm.read(12)
            frame_timestamp, payload_size = struct.unpack(METADATA_FORMAT, meta_bytes)
            
            # 6f. Zero-copy frame access
            frame = np.frombuffer(
                shm,
                dtype=np.uint8,
                count=FRAME_SIZE,
                offset=slot_offset + METADATA_SIZE
            ).reshape((HEIGHT, WIDTH, CHANNELS))
            
            # 6g. Run inference with timing
            start_qpc = get_qpc()
            results = hands.process(frame)
            end_qpc = get_qpc()
            
            # 6h. Calculate metrics
            inference_ms = (end_qpc - start_qpc) * 1000.0 / freq
            system_latency_ms = (end_qpc - frame_timestamp) * 1000.0 / freq
            
            # 6i. Extract hand data
            sequence += 1
            
            # Calculate dt for filters using actual wall-clock time
            current_qpc = get_qpc()
            current_dt = (current_qpc - last_frame_qpc) / freq
            current_dt = max(0.001, min(current_dt, 1.0))  # Clamp to sane range
            last_frame_qpc = current_qpc
            current_time_sec = current_qpc / freq
            
            if results.multi_hand_landmarks:
                no_hand_streak = 0  # Hand visible - reset grace counter
                hand = results.multi_hand_landmarks[0]

                # Get handedness for filter persistence
                handedness = "Right"  # Default
                if results.multi_handedness:
                    handedness = results.multi_handedness[0].classification[0].label
                confidence = results.multi_handedness[0].classification[0].score if results.multi_handedness else 0.5
                num_hands = 1
                
                # === SMOOTH LANDMARKS FIRST (critical for stable detection) ===
                smoothed = smooth_landmarks(hand.landmark, handedness, current_dt)
                
                # Use smoothed index tip for cursor position
                x, y, z = smoothed[8]
                
                # === CALCULATE VELOCITY (for C# extrapolation) ===
                if prev_x is not None and prev_time_sec is not None:
                    dt_sec = current_time_sec - prev_time_sec
                    if dt_sec > 0.001:  # Avoid division by near-zero
                        # Instantaneous velocity (units per second in normalized space)
                        inst_vx = (x - prev_x) / dt_sec
                        inst_vy = (y - prev_y) / dt_sec
                        # EMA smooth the velocity
                        velocity_x = VELOCITY_SMOOTH * inst_vx + (1 - VELOCITY_SMOOTH) * velocity_x
                        velocity_y = VELOCITY_SMOOTH * inst_vy + (1 - VELOCITY_SMOOTH) * velocity_y
                
                prev_x, prev_y = x, y
                prev_time_sec = current_time_sec
                
                # === INDEX-FINGER NORMALIZED PINCH DETECTION ===
                pinch_sq = pinch_distance_sq(smoothed)
                ref_sq = index_finger_length_sq(smoothed)

                # INPUT SANITIZATION: Prevent "Ratio Explosion" bug
                # Rule 1: Minimum scale threshold (prevent division by near-zero at extreme angles)
                if ref_sq < MIN_SCALE_THRESHOLD:
                    # Hand scale collapsed (likely tracking glitch) - hold previous ratio
                    raw_ratio = smoothed_ratio if smoothed_ratio > 0 else 1.0
                    # Silently handle - this is normal when hand is edge-on to camera
                else:
                    # Calculate ratio normally
                    raw_ratio = pinch_sq / ref_sq

                    # Rule 2: Clamp ratio to physical realism (silently clamp high values)
                    # Must be above PINCH_OPEN_THRESHOLD (0.70) so exit is reachable
                    if raw_ratio > MAX_RATIO:
                        raw_ratio = MAX_RATIO

                # Asymmetric EMA: fast when dropping (pinching), slow when rising (noise)
                if raw_ratio < smoothed_ratio:
                    alpha = RATIO_ALPHA_DOWN   # 0.5 - track pinch quickly
                else:
                    alpha = RATIO_ALPHA_UP     # 0.15 - resist upward spikes
                smoothed_ratio = alpha * raw_ratio + (1.0 - alpha) * smoothed_ratio

                # Phase 2: State Machine with Frame-Based Confirmation
                is_pinching = pinch_fsm.update(smoothed_ratio, PINCH_THRESHOLD, PINCH_OPEN_THRESHOLD)

                # Log state transitions (Phase 2 requirement)
                if pinch_fsm.state_changed:
                    if pinch_fsm.state == PinchStateMachine.STATE_PINCHED:
                        print(f"[Pinch] ENTER (ratio={smoothed_ratio:.3f}, raw={raw_ratio:.3f})", flush=True)
                        print(f"[Click] TRIGGER (ref_sq={ref_sq:.6f}, pinch_sq={pinch_sq:.6f})", flush=True)
                    elif pinch_fsm.state == PinchStateMachine.STATE_OPEN:
                        print(f"[Pinch] EXIT (ratio={smoothed_ratio:.3f})", flush=True)

                gesture_id = 2 if is_pinching else 1

                # === DIAGNOSTIC LOGGING ===
                # Tab-separated format for easy analysis
                raw_pinch_dist = pinch_sq ** 0.5
                state_transition = "YES" if pinch_fsm.state_changed else "NO"
                print(f"PY\t{frame_count}\t{raw_pinch_dist:.6f}\t{smoothed_ratio:.6f}\t{pinch_fsm.state}\t{pinch_fsm.entry_frames}\t{pinch_fsm.exit_frames}\t{state_transition}\t{gesture_id}", flush=True)
                
            else:
                # No hand detected - grace period before full reset
                no_hand_streak += 1
                x, y, z = 0.0, 0.0, 0.0
                velocity_x, velocity_y = 0.0, 0.0  # Reset velocity
                num_hands = 0

                # Only reset pinch state after sustained hand loss (grace period)
                if no_hand_streak >= NO_HAND_GRACE:
                    if pinch_fsm.state != PinchStateMachine.STATE_OPEN:
                        pinch_fsm.state = PinchStateMachine.STATE_OPEN
                        pinch_fsm.entry_frames = 0
                        pinch_fsm.exit_frames = 0
                        prev_x, prev_y = None, None
                        print(f"[Pinch] RESET (hand lost for {no_hand_streak} frames)", flush=True)
                    smoothed_ratio = 1.0

                is_pinching = False
                gesture_id = 0
                confidence = 0.0

                # === DIAGNOSTIC LOGGING (No Hand) ===
                print(f"PY\t{frame_count}\t0.000000\t{smoothed_ratio:.6f}\tNO_HAND({no_hand_streak})\t0\t0\tNO\t0", flush=True)
            
            # Pack and write HandState (now with velocity)
            hand_state_bytes = struct.pack(
                HAND_STATE_FORMAT,
                x, y, z,
                velocity_x, velocity_y,
                gesture_id,
                confidence,
                frame_timestamp
            )
            
            hand_shm.seek(HAND_DATA_OFFSET)
            hand_shm.write(hand_state_bytes)
            
            # Update sequence number
            hand_shm.seek(8)
            hand_shm.write(struct.pack('<q', sequence))
            
            # Signal C# reader (legacy)
            SetEvent(hand_event_handle)

            # === FlatBuffer Dual-Write (Phase 2 Protocol) ===
            # Build SensorFrame with HandState using FlatBuffers
            fb_builder.Clear()

            # Build HandState table
            FBHandState.HandStateStart(fb_builder)
            FBHandState.HandStateAddX(fb_builder, x)
            FBHandState.HandStateAddY(fb_builder, y)
            FBHandState.HandStateAddZ(fb_builder, z)
            FBHandState.HandStateAddVelocityX(fb_builder, velocity_x)
            FBHandState.HandStateAddVelocityY(fb_builder, velocity_y)
            FBHandState.HandStateAddGesture(fb_builder, gesture_id)
            FBHandState.HandStateAddConfidence(fb_builder, confidence)
            FBHandState.HandStateAddTimestamp(fb_builder, frame_timestamp)
            hand_offset = FBHandState.HandStateEnd(fb_builder)

            # Build SensorFrame wrapper
            FBSensorFrame.SensorFrameStart(fb_builder)
            FBSensorFrame.SensorFrameAddSequence(fb_builder, sequence)
            FBSensorFrame.SensorFrameAddHand(fb_builder, hand_offset)
            FBSensorFrame.SensorFrameAddProtocolVersion(fb_builder, SENSOR_PROTOCOL_VERSION)
            sensor_frame_offset = FBSensorFrame.SensorFrameEnd(fb_builder)

            # Finish buffer with file identifier "GRPL"
            fb_builder.Finish(sensor_frame_offset)
            fb_buf = fb_builder.Output()  # Returns bytearray (builder internal, no extra alloc)
            fb_size = len(fb_buf)

            # Write FlatBuffer to sensor arena
            sensor_shm.seek(SENSOR_DATA_OFFSET)
            sensor_shm.write(bytes(fb_buf))

            # Update sensor arena header (sequence + buffer size)
            sensor_shm.seek(8)   # offset of sequence in header
            sensor_shm.write(struct.pack('<q', sequence))
            sensor_shm.seek(20)  # offset of bufferSize in header
            sensor_shm.write(struct.pack('<i', fb_size))

            # Signal C# FlatBuffer reader
            SetEvent(sensor_event_handle)

            # 6j. Accumulate stats
            frame_count += 1
            total_inference_ms += inference_ms
            total_latency_ms += system_latency_ms

            # CALIBRATION MODE: Log ratio every 15 frames to find optimal thresholds
            if frame_count % 15 == 0:
                state_name = "PINCH" if is_pinching else "OPEN"
                print(f"[Calibration] Ratio: {smoothed_ratio:.3f} | State: {state_name}", flush=True)

            # 6k. Log every 60 frames (debug only)
            if args.debug and frame_count % 60 == 0:
                avg_inf = total_inference_ms / 60
                avg_lat = total_latency_ms / 60
                gesture_name = {0: "None", 1: "Point", 2: "Pinch"}.get(gesture_id, "?")
                print(f"[Debug] Frame: {frame_count} | Inf: {avg_inf:.2f}ms | "
                      f"Latency: {avg_lat:.2f}ms | Hands: {num_hands} | "
                      f"Gesture: {gesture_name} | Ratio: {smoothed_ratio:.3f} | V: ({velocity_x:.2f}, {velocity_y:.2f})", flush=True)
                total_inference_ms = 0.0
                total_latency_ms = 0.0
                
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
    finally:
        print("[*] Cleaning up...")
        hands.close()
        del frame
        kernel32.CloseHandle(sensor_event_handle)
        sensor_shm.close()
        kernel32.CloseHandle(hand_event_handle)
        hand_shm.close()
        shm.close()
        kernel32.CloseHandle(event_handle)
        print(f"[+] Processed {frame_count} frames, {skipped_frames} skipped")
        print(f"[+] Published {sequence} sensor frames (legacy + FlatBuffer)")
        print("[+] Done.")


if __name__ == "__main__":
    main()
