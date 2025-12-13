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
# Pinch detection thresholds (Schmitt Trigger) - TUNED for reliability
PINCH_THRESHOLD = 0.055      # Distance to START pinching (was 0.045 - too tight)
RELEASE_THRESHOLD = 0.075    # Distance to STOP pinching (was 0.10 - too wide, narrowed for responsiveness)

# Pinch distance smoothing (EMA)
PINCH_SMOOTH_ALPHA = 0.5     # 0 = full smoothing (laggy), 1 = no smoothing (jittery)

# Temporal debounce: require N consecutive frames to change pinch state
PINCH_ENTER_FRAMES = 2       # Frames of pinch before registering click
PINCH_EXIT_FRAMES = 3        # Frames of release before registering unclick (slightly more to prevent flicker during drag)

# Session locking: prevent cursor teleport when other hands appear
SESSION_LOCK_ENABLED = True
MAX_POSITION_JUMP = 0.20     # Reject position jumps larger than 20% of frame (teleport protection)

# Hand tracking
HAND_LOST_FRAMES_THRESHOLD = 10  # Frames without hand before resetting session


def main():
    print("=== Grapple Detector (MediaPipe Hands) ===")
    print(f"[*] Pinch thresholds: ENTER={PINCH_THRESHOLD}, EXIT={RELEASE_THRESHOLD}")
    print(f"[*] Debounce: ENTER={PINCH_ENTER_FRAMES}f, EXIT={PINCH_EXIT_FRAMES}f")
    print(f"[*] Session lock: {SESSION_LOCK_ENABLED}, Max jump: {MAX_POSITION_JUMP}")
    
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
            max_num_hands=1,  # CHANGED: Only track ONE hand to prevent ID collisions
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0  # 0=Lite (fastest), 1=Full
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
    sequence = 0  # Monotonic counter for hand results
    frame = None  # Track frame reference for cleanup
    
    # === PINCH STATE MACHINE ===
    is_pinching = False              # Current output pinch state
    pinch_distance_smoothed = 0.1    # EMA-smoothed pinch distance
    pinch_enter_count = 0            # Consecutive frames in "should pinch" zone
    pinch_exit_count = 0             # Consecutive frames in "should release" zone
    
    # === SESSION LOCKING STATE ===
    session_handedness = None        # "Left" or "Right" - locked when session starts
    last_x, last_y = None, None      # Last known position for jump detection
    hand_lost_frames = 0             # Frames since we last saw our session hand
    
    print("[*] Entering inference loop... Press Ctrl+C to quit.")
    print("[*] Expected: Inf ~10-15ms, Latency ~11-16ms (IPC + Inference)")
    
    try:
        while True:
            # 6a. Wait for frame signal (1 second timeout for Ctrl+C responsiveness)
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
                continue  # No frame published yet
            
            if published_id == last_buffer_id:
                continue  # Duplicate signal (spurious wakeup)
            
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
            
            # 6i. Extract hand data and write to result arena
            sequence += 1
            
            if results.multi_hand_landmarks:
                # With max_num_hands=1, there's only ever one hand
                hand = results.multi_hand_landmarks[0]
                handedness_label = results.multi_handedness[0].classification[0].label  # "Left" or "Right"
                
                # === SESSION LOCKING ===
                if SESSION_LOCK_ENABLED:
                    if session_handedness is None:
                        # No session yet - lock to this hand
                        session_handedness = handedness_label
                        print(f"[Session] Locked to {session_handedness} hand")
                    elif session_handedness != handedness_label:
                        # Wrong hand! Skip this frame entirely
                        hand_lost_frames += 1
                        if hand_lost_frames > HAND_LOST_FRAMES_THRESHOLD:
                            # Session hand gone too long, switch to new hand
                            session_handedness = handedness_label
                            last_x, last_y = None, None
                            print(f"[Session] Switched to {session_handedness} hand (previous lost)")
                        continue
                
                # Reset lost counter - we found our hand
                hand_lost_frames = 0
                
                # Get landmarks for cursor tracking
                index_tip = hand.landmark[8]   # Index finger tip
                thumb_tip = hand.landmark[4]   # Thumb tip
                
                x, y, z = index_tip.x, index_tip.y, index_tip.z  # Already normalized 0.0-1.0
                confidence = results.multi_handedness[0].classification[0].score
                num_hands = len(results.multi_hand_landmarks)
                
                # === TELEPORT PROTECTION ===
                if last_x is not None and SESSION_LOCK_ENABLED:
                    jump_distance = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
                    if jump_distance > MAX_POSITION_JUMP:
                        # Massive position jump - likely tracking glitch, skip frame
                        # But don't skip if we're not pinching (allow fast movements when pointing)
                        if is_pinching:
                            # During drag, reject teleports
                            continue
                        # When not dragging, allow fast movements but don't update smoothed position
                
                last_x, last_y = x, y
                
                # === PINCH DETECTION WITH SMOOTHING + DEBOUNCE ===
                # Calculate 3D Euclidean distance between thumb and index tips
                dx = thumb_tip.x - index_tip.x
                dy = thumb_tip.y - index_tip.y
                dz = (thumb_tip.z - index_tip.z) * 0.3  # Z is less reliable, weight it even lower
                pinch_distance_raw = (dx * dx + dy * dy + dz * dz) ** 0.5
                
                # Apply EMA smoothing to reduce noise
                pinch_distance_smoothed = (PINCH_SMOOTH_ALPHA * pinch_distance_raw + 
                                          (1.0 - PINCH_SMOOTH_ALPHA) * pinch_distance_smoothed)
                
                # Schmitt Trigger with TEMPORAL DEBOUNCE
                if pinch_distance_smoothed < PINCH_THRESHOLD:
                    # In "should pinch" zone
                    pinch_enter_count += 1
                    pinch_exit_count = 0
                    if pinch_enter_count >= PINCH_ENTER_FRAMES and not is_pinching:
                        is_pinching = True
                        # print(f"[Pinch] ENTER (dist={pinch_distance_smoothed:.3f})")
                        
                elif pinch_distance_smoothed > RELEASE_THRESHOLD:
                    # In "should release" zone
                    pinch_exit_count += 1
                    pinch_enter_count = 0
                    if pinch_exit_count >= PINCH_EXIT_FRAMES and is_pinching:
                        is_pinching = False
                        # print(f"[Pinch] EXIT (dist={pinch_distance_smoothed:.3f})")
                else:
                    # In hysteresis band - maintain state, don't reset counters
                    pass
                
                # Set gesture ID based on pinch state
                gesture_id = 2 if is_pinching else 1  # 2=Pinch, 1=Point
                
            else:
                # No hand detected — still write so C# knows frame was processed
                x, y, z = 0.0, 0.0, 0.0
                gesture_id = 0  # None
                confidence = 0.0
                num_hands = 0
                
                # Track hand loss for session management
                hand_lost_frames += 1
                if hand_lost_frames > HAND_LOST_FRAMES_THRESHOLD:
                    # Hand gone too long, reset session
                    if session_handedness is not None:
                        print(f"[Session] Released (hand lost for {hand_lost_frames} frames)")
                    session_handedness = None
                    last_x, last_y = None, None
                    is_pinching = False
                    pinch_enter_count = 0
                    pinch_exit_count = 0
                    pinch_distance_smoothed = 0.1
            
            # Pack and write HandState (40 bytes)
            hand_state_bytes = struct.pack(
                HAND_STATE_FORMAT,
                x, y, z,
                gesture_id,
                confidence,
                frame_timestamp  # Pass original timestamp for RTT calculation
            )
            
            hand_shm.seek(HAND_DATA_OFFSET)
            hand_shm.write(hand_state_bytes)
            
            # Update sequence number in header
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
                session_info = f"Session: {session_handedness or 'None'}"
                print(f"[Grapple] Frame: {frame_count} | Inf: {avg_inf:.2f}ms | "
                      f"Latency: {avg_lat:.2f}ms | Hands: {num_hands} | Gesture: {gesture_name} | "
                      f"{session_info} | Skips: {skipped_frames}")
                total_inference_ms = 0.0
                total_latency_ms = 0.0
                
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
    finally:
        print("[*] Cleaning up...")
        hands.close()
        # Release numpy view before closing mmap to avoid BufferError
        del frame
        # Close hand result resources
        kernel32.CloseHandle(hand_event_handle)
        hand_shm.close()
        # Close frame resources
        shm.close()
        kernel32.CloseHandle(event_handle)
        print(f"[+] Processed {frame_count} frames, {skipped_frames} skipped")
        print(f"[+] Published {sequence} hand states to C#")
        print("[+] Done.")


if __name__ == "__main__":
    main()
