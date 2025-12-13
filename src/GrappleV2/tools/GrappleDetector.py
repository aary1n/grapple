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
HAND_STATE_FORMAT = '<dddifq'  # 3×double, int, float, long = 40 bytes
HAND_STATE_SIZE = struct.calcsize(HAND_STATE_FORMAT)

# === TUNING PARAMETERS ===
# Pinch detection thresholds (Schmitt Trigger)
# WIDER thresholds = easier to trigger, more reliable
PINCH_THRESHOLD = 0.07       # Distance to START pinching (larger = easier to pinch)
RELEASE_THRESHOLD = 0.10     # Distance to STOP pinching

# Pinch distance smoothing (EMA) - HEAVY smoothing to reduce flicker
PINCH_SMOOTH_ALPHA = 0.3     # Lower = more smoothing (0.3 = quite smooth)

# Temporal debounce: require N consecutive frames to change pinch state
PINCH_ENTER_FRAMES = 2       # Frames of pinch before registering click
PINCH_EXIT_FRAMES = 4        # Frames of release before registering unclick


def main():
    print("=== Grapple Detector (MediaPipe Hands) ===")
    print(f"[*] Pinch thresholds: ENTER<{PINCH_THRESHOLD}, EXIT>{RELEASE_THRESHOLD}")
    print(f"[*] Debounce: ENTER={PINCH_ENTER_FRAMES}f, EXIT={PINCH_EXIT_FRAMES}f")
    
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
    
    # === PINCH STATE MACHINE ===
    is_pinching = False
    pinch_distance_smoothed = 0.15   # Start with "open" assumption
    pinch_enter_count = 0
    pinch_exit_count = 0
    
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
            
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                
                # Get landmarks
                index_tip = hand.landmark[8]   # Index finger tip
                thumb_tip = hand.landmark[4]   # Thumb tip
                
                x, y, z = index_tip.x, index_tip.y, index_tip.z
                confidence = results.multi_handedness[0].classification[0].score
                num_hands = 1
                
                # === PINCH DETECTION ===
                # Calculate 2D distance only (Z is too noisy)
                dx = thumb_tip.x - index_tip.x
                dy = thumb_tip.y - index_tip.y
                pinch_distance_raw = (dx * dx + dy * dy) ** 0.5
                
                # Apply EMA smoothing
                pinch_distance_smoothed = (PINCH_SMOOTH_ALPHA * pinch_distance_raw + 
                                          (1.0 - PINCH_SMOOTH_ALPHA) * pinch_distance_smoothed)
                
                # Schmitt Trigger with debounce
                if pinch_distance_smoothed < PINCH_THRESHOLD:
                    pinch_enter_count += 1
                    pinch_exit_count = 0
                    if pinch_enter_count >= PINCH_ENTER_FRAMES and not is_pinching:
                        is_pinching = True
                        print(f"[Pinch] ENTER (dist={pinch_distance_smoothed:.3f})")
                        
                elif pinch_distance_smoothed > RELEASE_THRESHOLD:
                    pinch_exit_count += 1
                    pinch_enter_count = 0
                    if pinch_exit_count >= PINCH_EXIT_FRAMES and is_pinching:
                        is_pinching = False
                        print(f"[Pinch] EXIT (dist={pinch_distance_smoothed:.3f})")
                else:
                    # In hysteresis band - don't reset counters, maintain state
                    pass
                
                gesture_id = 2 if is_pinching else 1
                
            else:
                # No hand detected
                x, y, z = 0.0, 0.0, 0.0
                gesture_id = 0
                confidence = 0.0
                num_hands = 0
                
                # Reset pinch state when hand lost
                if is_pinching:
                    is_pinching = False
                    print("[Pinch] RESET (hand lost)")
                pinch_enter_count = 0
                pinch_exit_count = 0
                pinch_distance_smoothed = 0.15
            
            # Pack and write HandState
            hand_state_bytes = struct.pack(
                HAND_STATE_FORMAT,
                x, y, z,
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
                pinch_str = f"Dist: {pinch_distance_smoothed:.3f}"
                print(f"[Grapple] Frame: {frame_count} | Inf: {avg_inf:.2f}ms | "
                      f"Latency: {avg_lat:.2f}ms | Hands: {num_hands} | "
                      f"Gesture: {gesture_name} | {pinch_str} | Skips: {skipped_frames}")
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
