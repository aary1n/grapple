"""
Grapple Detector - MediaPipe Hands Inference Pipeline
Zero-copy consumption of frames from C# producer.
Writes hand tracking results back to C# via shared memory.

Dependencies: pip install mediapipe numpy
"""

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


def get_qpc() -> int:
    """Get QueryPerformanceCounter value (matches C# Stopwatch.GetTimestamp())"""
    counter = ctypes.c_longlong()
    QueryPerformanceCounter(ctypes.byref(counter))
    return counter.value


# === Frame Arena Constants (MUST MATCH C#) ===
MAP_NAME = "Local\\GrappleMap"
EVENT_NAME = "Local\\GrappleSignal"
MAP_CAPACITY = 256 * 1024 * 1024
FIRST_SLOT_OFFSET = 1024
METADATA_SIZE = 64

WIDTH = 1920
HEIGHT = 1080
CHANNELS = 3
FRAME_SIZE = WIDTH * HEIGHT * CHANNELS

HEADER_FORMAT = '<Qiiqiiq'
HEADER_STRUCT_SIZE = struct.calcsize(HEADER_FORMAT)
METADATA_FORMAT = '<qi'

# === Hand Result Arena Constants (MUST MATCH C#) ===
HAND_MAP_NAME = "Local\\GrappleHandResults"
HAND_EVENT_NAME = "Local\\GrappleHandSignal"
HAND_MAP_SIZE = 4096
HAND_DATA_OFFSET = 64
HAND_MAGIC = 0x48414E4447525043  # "HANDGRPC" in little-endian

# Updated format with velocity: 5×double (x,y,z,vx,vy), int, float, long = 56 bytes
HAND_STATE_FORMAT = '<dddddifq'
HAND_STATE_SIZE = struct.calcsize(HAND_STATE_FORMAT)

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
LM_MIN_CUTOFF = 10    # High cutoff = responsive
LM_BETA = 2.1         # Speed-adaptive smoothing
LM_D_CUTOFF = 2.5     # Derivative cutoff

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
    """Squared length of index finger (MCP 5 → tip 8) - stable reference."""
    mcp = smoothed[5]
    tip = smoothed[8]
    dx = tip[0] - mcp[0]
    dy = tip[1] - mcp[1]
    return dx * dx + dy * dy


def pinch_distance_sq(smoothed) -> float:
    """Squared distance between thumb tip (4) and index tip (8)."""
    thumb = smoothed[4]
    index = smoothed[8]
    dx = thumb[0] - index[0]
    dy = thumb[1] - index[1]
    return dx * dx + dy * dy


# === TUNING PARAMETERS ===
# Threshold multiplier: ratio = pinch_dist² / index_finger²
# Index finger is ~3x longer than thumb segment, so thresholds are smaller
PINCH_THRESHOLD = 0.20      # Enter pinch when ratio < 0.20 (fingers very close)
PINCH_OPEN_THRESHOLD = 0.50 # Exit when ratio > 0.50 (fingers ~70% of index length apart)

# Time-based exit debounce (prevents accidental release during drag)
PINCH_EXIT_MS = 150         # Back to responsive now that reference is stable

# Ratio smoothing for detection stability
RATIO_SMOOTH_ALPHA = 0.5    # EMA smoothing on pinch ratio


def main():
    print("=== Grapple Detector (MediaPipe Hands) ===")
    print(f"[*] Pinch thresholds (index-finger ref): ENTER<{PINCH_THRESHOLD:.2f}, EXIT>{PINCH_OPEN_THRESHOLD:.2f}")
    print(f"[*] Time-based exit debounce: {PINCH_EXIT_MS}ms, Ratio smoothing: {RATIO_SMOOTH_ALPHA}")
    print(f"[*] Per-landmark OneEuroFilter: cutoff={LM_MIN_CUTOFF}, beta={LM_BETA}")
    print(f"[*] HandState format: {HAND_STATE_SIZE} bytes (with velocity)")
    
    # === 1. Open Frame Signal Event ===
    print(f"[*] Opening event: {EVENT_NAME}")
    event_handle = kernel32.OpenEventW(
        SYNCHRONIZE | EVENT_MODIFY_STATE,
        False,
        EVENT_NAME
    )
    if not event_handle:
        print(f"[!] Failed to open event. Is C# producer running?")
        print(f"    Error code: {kernel32.GetLastError()}")
        return
    print(f"[+] Event opened")
    
    # === 2. Map Frame Shared Memory ===
    print(f"[*] Mapping shared memory: {MAP_NAME}")
    try:
        shm = mmap.mmap(-1, MAP_CAPACITY, tagname=MAP_NAME, access=mmap.ACCESS_READ)
    except Exception as e:
        print(f"[!] Failed to map memory: {e}")
        kernel32.CloseHandle(event_handle)
        return
    print(f"[+] Memory mapped: {MAP_CAPACITY // (1024*1024)} MB")
    
    # === 3. Read Frame Arena Header ===
    header_bytes = shm[:HEADER_STRUCT_SIZE]
    magic, slot_count, slot_size, write_head, published_id, _pad, freq = struct.unpack(
        HEADER_FORMAT, header_bytes
    )
    
    expected_magic = 0x31454C5050415247
    if magic != expected_magic:
        print(f"[!] Invalid magic number!")
        shm.close()
        kernel32.CloseHandle(event_handle)
        return
    
    print(f"[+] Header valid: Slots={slot_count}, SlotSize={slot_size // (1024*1024)}MB")
    print(f"[*] QPC Frequency: {freq:,} ticks/sec")
    
    # === 4. Setup Hand Result Arena (Write) ===
    print(f"[*] Creating hand result arena: {HAND_MAP_NAME}")
    try:
        hand_shm = mmap.mmap(-1, HAND_MAP_SIZE, tagname=HAND_MAP_NAME, access=mmap.ACCESS_WRITE)
    except Exception as e:
        print(f"[!] Failed to create hand result memory: {e}")
        shm.close()
        kernel32.CloseHandle(event_handle)
        return
    print(f"[+] Hand result memory mapped: {HAND_MAP_SIZE} bytes")
    
    # Create hand signal event (AutoReset = False for bManualReset, initially non-signaled)
    hand_event_handle = CreateEventW(None, False, False, HAND_EVENT_NAME)
    if not hand_event_handle:
        print(f"[!] Failed to create hand signal event. Error: {kernel32.GetLastError()}")
        hand_shm.close()
        shm.close()
        kernel32.CloseHandle(event_handle)
        return
    print(f"[+] Hand signal event created")
    
    # Initialize hand result header if needed
    hand_shm.seek(0)
    existing_magic = struct.unpack('<Q', hand_shm.read(8))[0]
    if existing_magic != HAND_MAGIC:
        print(f"[*] Initializing hand result header...")
        hand_shm.seek(0)
        hand_shm.write(struct.pack('<Q', HAND_MAGIC))  # Magic at offset 0
        hand_shm.write(struct.pack('<q', 0))           # Sequence at offset 8
    print(f"[+] Hand result arena ready")
    
    # === 5. Initialize MediaPipe ===
    print("[*] Initializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    try:
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,          # Only track ONE hand
            min_detection_confidence=0.5,  # Lower threshold = easier to detect
            min_tracking_confidence=0.4,   # Lower threshold = more stable tracking
            model_complexity=0        # 0=Lite (fastest)
        )
    except Exception as e:
        print(f"[!] Failed to initialize MediaPipe: {e}")
        kernel32.CloseHandle(hand_event_handle)
        hand_shm.close()
        shm.close()
        kernel32.CloseHandle(event_handle)
        return
    print("[+] MediaPipe Hands initialized (model_complexity=0, max_hands=1)")
    
    # === 6. Inference Loop ===
    frame_count = 0
    last_buffer_id = -1
    skipped_frames = 0
    total_inference_ms = 0.0
    total_latency_ms = 0.0
    sequence = 0
    frame = None
    
    # === PINCH STATE ===
    is_pinching = False
    pinch_open_since_ms = None
    last_frame_qpc = get_qpc()
    smoothed_ratio = 1.0  # Start in middle
    
    # === VELOCITY TRACKING ===
    prev_x, prev_y = None, None
    prev_time_sec = None
    velocity_x, velocity_y = 0.0, 0.0
    VELOCITY_SMOOTH = 0.3  # EMA smoothing for velocity
    
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
            current_time_ms = int((current_qpc / freq) * 1000)
            current_time_sec = current_qpc / freq
            
            if results.multi_hand_landmarks:
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
                
                # Avoid division by zero
                if ref_sq < 1e-12:
                    ref_sq = 1e-12
                
                # Calculate ratio and apply EMA smoothing
                raw_ratio = pinch_sq / ref_sq
                smoothed_ratio = RATIO_SMOOTH_ALPHA * raw_ratio + (1.0 - RATIO_SMOOTH_ALPHA) * smoothed_ratio
                
                # Schmitt trigger with TIME-BASED exit debounce
                if not is_pinching:
                    if smoothed_ratio < PINCH_THRESHOLD:
                        is_pinching = True
                        pinch_open_since_ms = None
                        print(f"[Pinch] ENTER (ratio={smoothed_ratio:.3f})", flush=True)
                else:
                    if smoothed_ratio > PINCH_OPEN_THRESHOLD:
                        if pinch_open_since_ms is None:
                            pinch_open_since_ms = current_time_ms
                        elif (current_time_ms - pinch_open_since_ms) >= PINCH_EXIT_MS:
                            is_pinching = False
                            pinch_open_since_ms = None
                            print(f"[Pinch] EXIT (ratio={smoothed_ratio:.3f}, held {PINCH_EXIT_MS}ms)", flush=True)
                    else:
                        pinch_open_since_ms = None
                
                gesture_id = 2 if is_pinching else 1
                
            else:
                # No hand detected
                x, y, z = 0.0, 0.0, 0.0
                velocity_x, velocity_y = 0.0, 0.0  # Reset velocity
                gesture_id = 2 if is_pinching else 0
                confidence = 0.6 if is_pinching else 0.0
                num_hands = 0
                
                # Time-based release when hand lost
                if is_pinching:
                    if pinch_open_since_ms is None:
                        pinch_open_since_ms = current_time_ms
                    elif (current_time_ms - pinch_open_since_ms) >= PINCH_EXIT_MS:
                        is_pinching = False
                        pinch_open_since_ms = None
                        smoothed_ratio = 1.0
                        prev_x, prev_y = None, None
                        print("[Pinch] RESET (hand lost)", flush=True)
            
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
            
            # Signal C# reader
            SetEvent(hand_event_handle)
            
            # 6j. Accumulate stats
            frame_count += 1
            total_inference_ms += inference_ms
            total_latency_ms += system_latency_ms
            
            # 6k. Log every 60 frames
            if frame_count % 60 == 0:
                avg_inf = total_inference_ms / 60
                avg_lat = total_latency_ms / 60
                gesture_name = {0: "None", 1: "Point", 2: "Pinch"}.get(gesture_id, "?")
                print(f"[Grapple] Frame: {frame_count} | Inf: {avg_inf:.2f}ms | "
                      f"Latency: {avg_lat:.2f}ms | Hands: {num_hands} | "
                      f"Gesture: {gesture_name} | V: ({velocity_x:.2f}, {velocity_y:.2f})")
                total_inference_ms = 0.0
                total_latency_ms = 0.0
                
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
    finally:
        print("[*] Cleaning up...")
        hands.close()
        del frame
        kernel32.CloseHandle(hand_event_handle)
        hand_shm.close()
        shm.close()
        kernel32.CloseHandle(event_handle)
        print(f"[+] Processed {frame_count} frames, {skipped_frames} skipped")
        print(f"[+] Published {sequence} hand states to C#")
        print("[+] Done.")


if __name__ == "__main__":
    main()
