"""
Grapple Debug Viewer - Python IPC Consumer
Displays frames from C# producer via shared memory.

Dependencies: pip install numpy opencv-python
"""

import mmap
import struct
import ctypes
from ctypes import wintypes
import numpy as np
import cv2

# === Windows API ===
kernel32 = ctypes.windll.kernel32

# Event access flags
SYNCHRONIZE = 0x00100000
EVENT_MODIFY_STATE = 0x0002

# WaitForSingleObject return values
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102

# QueryPerformanceCounter
QueryPerformanceCounter = kernel32.QueryPerformanceCounter
QueryPerformanceCounter.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
QueryPerformanceCounter.restype = wintypes.BOOL


def get_qpc() -> int:
    """Get current QueryPerformanceCounter value (matches C# Stopwatch.GetTimestamp())"""
    counter = ctypes.c_longlong()
    QueryPerformanceCounter(ctypes.byref(counter))
    return counter.value


# === Constants (MUST MATCH C#) ===
MAP_NAME = "Local\\GrappleMap"
EVENT_NAME = "Local\\GrappleSignal"
MAP_CAPACITY = 256 * 1024 * 1024  # 256 MB
HEADER_SIZE = 1024
FIRST_SLOT_OFFSET = 1024
METADATA_SIZE = 64

# Frame dimensions
WIDTH = 1920
HEIGHT = 1080
CHANNELS = 3
FRAME_SIZE = WIDTH * HEIGHT * CHANNELS

# Header struct format (little-endian, matches C# LayoutKind.Sequential)
# ulong MagicNumber     (Q) = 8 bytes
# int SlotCount         (i) = 4 bytes
# int SlotSize          (i) = 4 bytes
# long WriteHeadIndex   (q) = 8 bytes
# int PublishedBufferId (i) = 4 bytes
# int _padding          (i) = 4 bytes
# long TimestampFreq    (q) = 8 bytes
# Total: 40 bytes
HEADER_FORMAT = '<Qiiqiiq'  # Little-endian
HEADER_STRUCT_SIZE = struct.calcsize(HEADER_FORMAT)

# Slot metadata format
# long Timestamp    (q) = 8 bytes
# int PayloadSize   (i) = 4 bytes
METADATA_FORMAT = '<qi'


def main():
    print("=== Grapple Debug Viewer (Python) ===")
    
    # 1. Open the Named Event
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
    print(f"[+] Event opened: handle={event_handle}")
    
    # 2. Map the Shared Memory
    print(f"[*] Mapping shared memory: {MAP_NAME}")
    try:
        shm = mmap.mmap(-1, MAP_CAPACITY, tagname=MAP_NAME, access=mmap.ACCESS_READ)
    except Exception as e:
        print(f"[!] Failed to map memory: {e}")
        kernel32.CloseHandle(event_handle)
        return
    print(f"[+] Memory mapped: {MAP_CAPACITY // (1024*1024)} MB")
    
    # 3. Read Header
    header_bytes = shm[:HEADER_STRUCT_SIZE]
    magic, slot_count, slot_size, write_head, published_id, _pad, freq = struct.unpack(
        HEADER_FORMAT, header_bytes
    )
    
    print(f"[*] Header: Magic=0x{magic:016X}, Slots={slot_count}, SlotSize={slot_size // (1024*1024)}MB")
    print(f"[*] Frequency: {freq:,} ticks/sec")
    
    # Validate magic number (ASCII "GRAPPLE1")
    expected_magic = 0x31454C5050415247
    if magic != expected_magic:
        print(f"[!] Invalid magic number! Expected 0x{expected_magic:X}")
        shm.close()
        kernel32.CloseHandle(event_handle)
        return
    print("[+] Magic number verified: GRAPPLE1")
    
    # 4. Create display window
    cv2.namedWindow("Grapple Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grapple Debug", 960, 540)  # Half resolution for display
    
    frame_count = 0
    last_buffer_id = -1
    
    print("[*] Entering wait loop... Press 'q' to quit.")
    
    try:
        while True:
            # 5. Wait for signal (1 second timeout to check for quit)
            result = kernel32.WaitForSingleObject(event_handle, 1000)
            
            if result == WAIT_TIMEOUT:
                # Check if window was closed
                if cv2.getWindowProperty("Grapple Debug", cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            elif result != WAIT_OBJECT_0:
                print(f"[!] WaitForSingleObject failed: {result}")
                break
            
            # 6. Read current published buffer ID from header
            shm.seek(24)  # Offset of PublishedBufferId
            published_id = struct.unpack('<i', shm.read(4))[0]
            
            if published_id == -1 or published_id == last_buffer_id:
                continue  # Spurious wakeup or duplicate
            
            last_buffer_id = published_id
            
            # 7. Calculate slot offset
            slot_offset = FIRST_SLOT_OFFSET + (published_id * slot_size)
            
            # 8. Read slot metadata
            shm.seek(slot_offset)
            meta_bytes = shm.read(12)
            timestamp, payload_size = struct.unpack(METADATA_FORMAT, meta_bytes)
            
            # 9. Calculate latency using QPC
            now = get_qpc()
            latency_ms = (now - timestamp) * 1000.0 / freq
            
            # 10. Read pixel data (skip 64-byte metadata)
            shm.seek(slot_offset + METADATA_SIZE)
            pixel_data = shm.read(FRAME_SIZE)
            
            # 11. Convert to numpy array
            frame = np.frombuffer(pixel_data, dtype=np.uint8).reshape((HEIGHT, WIDTH, CHANNELS))
            
            # 12. Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 13. Overlay telemetry
            frame_count += 1
            text = f"Frame: {frame_count} | Buffer: {published_id} | Latency: {latency_ms:.2f}ms"
            cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)
            
            # 14. Display
            cv2.imshow("Grapple Debug", frame_bgr)
            
            # Log every 60 frames
            if frame_count % 60 == 0:
                print(f"[*] Frame: {frame_count} | Latency: {latency_ms:.2f}ms")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
    finally:
        print("[*] Cleaning up...")
        cv2.destroyAllWindows()
        shm.close()
        kernel32.CloseHandle(event_handle)
        print("[+] Done.")


if __name__ == "__main__":
    main()

