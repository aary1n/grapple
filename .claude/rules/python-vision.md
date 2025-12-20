# Python Vision Rules - Grapple Sensor Layer

## Role Definition

**Python is the SENSOR. C# is the ACTUATOR.**

Python's job:
1. Consume video frames from shared memory (zero-copy)
2. Run MediaPipe inference
3. Write hand tracking results to shared memory
4. **NEVER** touch the mouse or keyboard

C# owns all OS input injection. Python is a pure data provider.

---

## 1. Zero-Copy Shared Memory IPC

**Rule:** Use `struct.pack` for direct writes. NO intermediate allocations where possible.

### Pattern
```python
# Direct pack into shared memory
hand_shm.seek(HAND_DATA_OFFSET)
hand_state_bytes = struct.pack(
    HAND_STATE_FORMAT,
    x, y, z, velocity_x, velocity_y,
    gesture_id, confidence, timestamp
)
hand_shm.write(hand_state_bytes)
```

---

## 2. Protocol Alignment (CRITICAL)

**Rule:** Python struct format MUST match C# StructLayout byte-for-byte.

Python side (56 bytes):
```python
HAND_STATE_FORMAT = '<dddddifq'  # little-endian
assert struct.calcsize(HAND_STATE_FORMAT) == 56
```

---

## 3. MediaPipe Configuration

```python
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,              # Single hand only
    min_detection_confidence=0.5,  # Lower = faster
    min_tracking_confidence=0.4,   # Lower = smoother
    model_complexity=0             # Lite model
)
```

---

## 4. Landmark Smoothing (1€ Filter)

```python
LM_MIN_CUTOFF = 10    # High cutoff = responsive
LM_BETA = 2.1         # Speed-adaptive
LM_D_CUTOFF = 2.5     # Derivative cutoff
```

---

## 5. Gesture Detection (Pinch-to-Click)

```python
PINCH_THRESHOLD = 0.065   # Enter pinch state
RELEASE_THRESHOLD = 0.12  # Exit pinch (hysteresis)
PINCH_DEBOUNCE = 2        # Frames to enter
RELEASE_DEBOUNCE = 3      # Frames to exit
```

---

## 6. No OS Input Injection

**Rule:** Python NEVER calls mouse/keyboard functions.

Forbidden: pyautogui, pynput, SendInput

**Why?** C# owns input injection for DPI scaling, multi-monitor, security.

---

**Last Updated:** 2025-12-20
