"""
Grapple Detector - MediaPipe Hands Inference Pipeline
Zero-copy consumption of frames from C# producer.

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


def get_qpc() -> int:
    """Get QueryPerformanceCounter value (matches C# Stopwatch.GetTimestamp())"""
    counter = ctypes.c_longlong()
    QueryPerformanceCounter(ctypes.byref(counter))
    return counter.value


# === Constants (MUST MATCH C#) ===
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


def main():
    print("=== Grapple Detector (MediaPipe Hands) ===")
    
    # === 1. Open Named Event ===
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
    
    # === 2. Map Shared Memory ===
    print(f"[*] Mapping shared memory: {MAP_NAME}")
    try:
        shm = mmap.mmap(-1, MAP_CAPACITY, tagname=MAP_NAME, access=mmap.ACCESS_READ)
    except Exception as e:
        print(f"[!] Failed to map memory: {e}")
        kernel32.CloseHandle(event_handle)
        return
    print(f"[+] Memory mapped: {MAP_CAPACITY // (1024*1024)} MB")
    
    # === 3. Read Header ===
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
    
    # === 4. Initialize MediaPipe ===
    print("[*] Initializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    try:
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0  # 0=Lite (fastest), 1=Full
        )
    except Exception as e:
        print(f"[!] Failed to initialize MediaPipe: {e}")
        shm.close()
        kernel32.CloseHandle(event_handle)
        return
    print("[+] MediaPipe Hands initialized (model_complexity=0)")
    
    # === 5. Inference Loop ===
    frame_count = 0
    last_buffer_id = -1
    skipped_frames = 0
    total_inference_ms = 0.0
    total_latency_ms = 0.0
    
    print("[*] Entering inference loop... Press Ctrl+C to quit.")
    print("[*] Expected: Inf ~10-15ms, Latency ~11-16ms (IPC + Inference)")
    
    try:
        while True:
            # 5a. Wait for signal (1 second timeout for Ctrl+C responsiveness)
            result = kernel32.WaitForSingleObject(event_handle, 1000)
            
            if result == WAIT_TIMEOUT:
                continue
            elif result != WAIT_OBJECT_0:
                print(f"[!] WaitForSingleObject failed: {result}")
                break
            
            # 5b. Read PublishedBufferId from header
            shm.seek(24)
            published_id = struct.unpack('<i', shm.read(4))[0]
            
            if published_id == -1:
                continue  # No frame published yet
            
            if published_id == last_buffer_id:
                continue  # Duplicate signal (spurious wakeup)
            
            # 5c. Detect skipped frames
            if last_buffer_id != -1:
                expected_next = (last_buffer_id + 1) % slot_count
                if published_id != expected_next:
                    skipped_frames += 1
            
            last_buffer_id = published_id
            
            # 5d. Calculate slot offset
            slot_offset = FIRST_SLOT_OFFSET + (published_id * slot_size)
            
            # 5e. Read timestamp from slot metadata
            shm.seek(slot_offset)
            meta_bytes = shm.read(12)
            frame_timestamp, payload_size = struct.unpack(METADATA_FORMAT, meta_bytes)
            
            # 5f. Zero-copy frame access
            frame = np.frombuffer(
                shm,
                dtype=np.uint8,
                count=FRAME_SIZE,
                offset=slot_offset + METADATA_SIZE
            ).reshape((HEIGHT, WIDTH, CHANNELS))
            
            # 5g. Run inference with timing
            start_qpc = get_qpc()
            results = hands.process(frame)
            end_qpc = get_qpc()
            
            # 5h. Calculate metrics
            inference_ms = (end_qpc - start_qpc) * 1000.0 / freq
            system_latency_ms = (end_qpc - frame_timestamp) * 1000.0 / freq
            
            # 5i. Count detected hands
            num_hands = 0
            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
            
            # 5j. Accumulate stats
            frame_count += 1
            total_inference_ms += inference_ms
            total_latency_ms += system_latency_ms
            
            # 5k. Log every 60 frames
            if frame_count % 60 == 0:
                avg_inf = total_inference_ms / 60
                avg_lat = total_latency_ms / 60
                print(f"[Grapple] Frame: {frame_count} | Inf: {avg_inf:.2f}ms | "
                      f"Latency: {avg_lat:.2f}ms | Hands: {num_hands} | Skips: {skipped_frames}")
                total_inference_ms = 0.0
                total_latency_ms = 0.0
                
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
    finally:
        print("[*] Cleaning up...")
        hands.close()
        shm.close()
        kernel32.CloseHandle(event_handle)
        print(f"[+] Processed {frame_count} frames, {skipped_frames} skipped")
        print("[+] Done.")


if __name__ == "__main__":
    main()

