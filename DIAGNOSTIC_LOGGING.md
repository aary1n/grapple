# Diagnostic Logging for Click Detection Issues

## Overview

This document describes the diagnostic logging added to identify click detection and drag reliability issues in the Grapple pipeline.

---

## Log Format

### Python Side (GrappleDetector.py)

**Format:** `PY\t{frame}\t{raw_dist}\t{ratio}\t{state}\t{entry}\t{exit}\t{transition}\t{gesture}`

**Columns:**
1. `PY` - Prefix to identify Python logs
2. `frame` - Frame counter
3. `raw_dist` - Raw pinch distance (Euclidean distance between thumb tip and index tip)
4. `ratio` - Smoothed pinch ratio (pinch_distance² / index_finger_length²)
5. `state` - Current FSM state: `OPEN`, `APPROACHING`, `PINCHED`, `RELEASING`, or `NO_HAND`
6. `entry` - Entry debounce counter (frames accumulated toward PINCH)
7. `exit` - Exit debounce counter (frames accumulated toward OPEN)
8. `transition` - `YES` if state changed this frame, `NO` otherwise
9. `gesture` - Final gesture ID sent to C# (0=None, 1=Point, 2=Pinch)

**Example:**
```
PY	142	0.085432	0.285714	APPROACHING	2	0	NO	1
PY	143	0.078291	0.240123	PINCHED	0	0	YES	2
PY	144	0.076543	0.235891	PINCHED	0	0	NO	2
```

### C# Side (MouseControllerNode.cs)

**Format:** `CS\t{frame}\t{gesture}\t{click}\t{action}\t{stale_ms}\t{seq}\t{gap}`

**Columns:**
1. `CS` - Prefix to identify C# logs
2. `frame` - Frame counter (120Hz cursor loop)
3. `gesture` - Gesture ID received from HandState (0=None, 1=Point, 2=Pinch)
4. `click` - Current internal click state (`DOWN` or `UP`)
5. `action` - Action taken this frame:
   - `MOVE` - Cursor moved only
   - `DOWN` - Mouse button pressed (transition to pinch)
   - `UP` - Mouse button released (transition from pinch)
   - `NONE` - No hand detected, no action
   - `SAFETY_UP` - Safety release due to hand loss
6. `stale_ms` - Time since last inference update (ms) - detects stale data
7. `seq` - Python sequence number (monotonic counter from inference)
8. `gap` - Sequence gap (0=no drop, >0=dropped frames between Python updates)

**Example:**
```
CS	1024	1	UP	MOVE	12.45	84	0
CS	1025	2	DOWN	DOWN	8.23	85	0
CS	1026	2	DOWN	MOVE	16.12	85	0
```

---

## Capture Procedure

### 1. Build and Run

```bash
# Build the solution
dotnet build src/GrappleV2/GrappleGraphs.sln

# Run the full pipeline with logging redirected to file
dotnet run --project src/GrappleV2/Grapple.SmokeTests -- --full > diagnostic_log.txt 2>&1
```

### 2. Test Sequence

Once the pipeline is running:

1. **Activate with F9** - Press F9 to enable mouse control
2. **Slow Pinch-and-Hold (2 seconds)**
   - Slowly bring thumb and index finger together
   - Hold the pinch for 2 full seconds without moving
   - Observe if cursor jitters or click releases prematurely
3. **Pinch-and-Drag Circle**
   - Pinch your fingers together
   - While holding the pinch, draw a circle in the air
   - Try to maintain steady pressure throughout
4. **Release**
   - Slowly open your fingers
   - Observe if click releases at the right time
5. **Stop Logging**
   - Press Ctrl+C to stop

### 3. Extract Logs

Filter the diagnostic logs from the full output:

```bash
# Extract Python logs
grep "^PY" diagnostic_log.txt > python_logs.tsv

# Extract C# logs
grep "^CS" diagnostic_log.txt > csharp_logs.tsv
```

---

## Analysis Guide

### Issue 1: Flickering Gesture During Hold

**Symptom:** Click randomly releases during what should be a steady hold

**What to check:**
1. Look at Python logs during the hold period
2. Check if `gesture` column flickers between `1` (Point) and `2` (Pinch)
3. Check if `state` bounces between `PINCHED` and `RELEASING`

**Example Problem:**
```
PY	100	0.065	0.220	PINCHED	0	0	NO	2
PY	101	0.082	0.265	RELEASING	0	1	NO	2  <- Exit counter incrementing
PY	102	0.068	0.228	PINCHED	0	0	YES	2  <- Bounced back!
```

**Root Cause:** Ratio bouncing across exit threshold due to:
- Noise in landmark tracking
- Too tight exit threshold (needs wider hysteresis)
- Insufficient smoothing

**Fix:** Increase `PINCH_OPEN_THRESHOLD` or increase `RATIO_SMOOTH_ALPHA` in [GrappleDetector.py:202-208](src/GrappleV2/tools/GrappleDetector.py#L202-L208)

---

### Issue 2: C# Doesn't Respond to Python Gesture

**Symptom:** Python sends `gesture=2` but C# never clicks

**What to check:**
1. Compare Python `gesture` with C# `gesture` at the same wall-clock time
2. Look for sequence gaps in C# logs (`gap` column > 0)
3. Check `stale_ms` - if >100ms, inference is lagging behind cursor loop

**Example Problem:**
```
PY	150	0.065	0.220	PINCHED	0	0	YES	2  <- Python sends pinch
CS	1800	1	UP	MOVE	95.23	149	0     <- C# still sees Point (seq=149)
CS	1801	1	UP	MOVE	103.45	149	0    <- Stale data! Still seq=149
CS	1802	2	DOWN	DOWN	8.12	150	0    <- Finally got it, 100ms late
```

**Root Cause:**
- Python inference lagging (>100ms per frame)
- Extrapolation state not updating fast enough
- IPC event signaling delayed

**Fix:**
- Check Python inference time in debug logs
- Verify `SetEvent(hand_event_handle)` is being called
- Check if C# `InferenceReaderLoop` is blocked

---

### Issue 3: Premature Click Release During Drag

**Symptom:** Drawing curved lines is choppy - click releases mid-drag

**What to check:**
1. During the circle drag, look for `action=UP` events in C# logs
2. Cross-reference with Python logs - did gesture actually change?
3. Check for frame drops (`gap` > 0)

**Example Problem:**
```
# During circle drag:
CS	2500	2	DOWN	MOVE	12.34	200	0
CS	2501	2	DOWN	MOVE	15.67	200	0  <- Same seq, extrapolating
CS	2502	2	DOWN	MOVE	18.89	200	0  <- Still same seq!
CS	2503	2	DOWN	MOVE	22.12	200	0  <- Extrapolation getting stale
CS	2504	1	UP	SAFETY_UP	145.00	201	0  <- Hand lost! (stale > 145ms)
```

**Root Cause:**
- Frame rate drop during inference (hand moving fast → tracking lost)
- Extrapolation timeout triggered safety release
- `MaxExtrapolationSec` (150ms) too short for 15fps inference

**Fix:**
- Increase `MaxExtrapolationSec` in [MouseControllerNode.cs:51](src/GrappleV2/Grapple.Nodes/MouseControllerNode.cs#L51)
- Add lag detection warning in Python (if inference > 80ms)

---

### Issue 4: Debounce Not Triggering

**Symptom:** Single-frame noise causes spurious clicks

**What to check:**
1. Look for `transition=YES` with `entry=0` or `exit=0`
2. This means debounce was bypassed (shouldn't happen with 3/5 frame requirement)

**Example Problem:**
```
PY	180	0.320	0.480	OPEN	0	0	NO	1
PY	181	0.062	0.215	PINCHED	0	0	YES	2  <- INSTANT transition! (entry=0)
```

**Root Cause:**
- Bug in state machine logic (shouldn't be possible with current code)
- Ratio spike so large it saturates the counter in one frame

**Fix:**
- Review [GrappleDetector.py:243-306](src/GrappleV2/tools/GrappleDetector.py#L243-L306) for FSM bugs
- Add ratio clamping to prevent extreme spikes

---

## Side-by-Side Comparison Script

Use this Python script to correlate logs by timestamp:

```python
import pandas as pd

py_df = pd.read_csv('python_logs.tsv', sep='\t', names=[
    'prefix', 'py_frame', 'raw_dist', 'ratio', 'state', 'entry', 'exit', 'transition', 'py_gesture'
])

cs_df = pd.read_csv('csharp_logs.tsv', sep='\t', names=[
    'prefix', 'cs_frame', 'cs_gesture', 'click', 'action', 'stale_ms', 'seq', 'gap'
])

# Merge on Python sequence number (links inference to cursor frames)
merged = cs_df.merge(py_df, left_on='seq', right_on='py_frame', how='left')

# Find mismatches where C# gesture != Python gesture
mismatches = merged[merged['cs_gesture'] != merged['py_gesture']]
print(f"Found {len(mismatches)} gesture mismatches")
print(mismatches[['cs_frame', 'seq', 'cs_gesture', 'py_gesture', 'stale_ms', 'gap']])

# Find premature releases (action=UP while py_gesture=2)
premature = merged[(merged['action'] == 'UP') & (merged['py_gesture'] == 2)]
print(f"\nFound {len(premature)} premature releases")
print(premature[['cs_frame', 'seq', 'py_gesture', 'state', 'stale_ms']])

# Check debounce effectiveness (state transitions)
transitions = py_df[py_df['transition'] == 'YES']
print(f"\nState transitions: {len(transitions)}")
print(transitions[['py_frame', 'state', 'entry', 'exit', 'ratio']])
```

---

## Expected Behavior

### Healthy Pinch-and-Hold
```
PY	100	0.320	0.480	OPEN	0	0	NO	1
PY	101	0.280	0.420	APPROACHING	1	0	NO	1
PY	102	0.240	0.360	APPROACHING	2	0	NO	1
PY	103	0.065	0.220	PINCHED	0	0	YES	2  <- Clean transition after 3 frames
PY	104	0.063	0.218	PINCHED	0	0	NO	2
...
PY	134	0.062	0.217	PINCHED	0	0	NO	2  <- Rock solid for 30 frames
```

### Healthy Pinch-to-Release
```
PY	200	0.065	0.220	PINCHED	0	0	NO	2
PY	201	0.120	0.380	RELEASING	0	1	NO	2  <- Exit counter starts
PY	202	0.125	0.385	RELEASING	0	2	NO	2
PY	203	0.130	0.390	RELEASING	0	3	NO	2
PY	204	0.135	0.395	RELEASING	0	4	NO	2
PY	205	0.140	0.400	OPEN	0	0	YES	1  <- Clean release after 5 frames
```

### Healthy C# Response (No Lag)
```
CS	1500	1	UP	MOVE	12.34	100	0
CS	1501	2	DOWN	DOWN	8.45	103	0  <- Responded within 3 Python frames
CS	1502	2	DOWN	MOVE	16.23	103	0
CS	1503	2	DOWN	MOVE	24.12	103	0
CS	1504	2	DOWN	MOVE	7.89	104	0  <- New inference arrived
```

---

## Turning Off Diagnostic Logging

Once you've captured the data, comment out the logging lines:

**GrappleDetector.py** - Comment lines added around [line 560]:
```python
# print(f"PY\t{frame_count}\t{raw_pinch_dist:.6f}\t{smoothed_ratio:.6f}\t{pinch_fsm.state}\t{pinch_fsm.entry_frames}\t{pinch_fsm.exit_frames}\t{state_transition}\t{gesture_id}", flush=True)
```

**MouseControllerNode.cs** - Comment lines added around [line 306]:
```csharp
// Console.WriteLine($"CS\t{_frameCount}\t{gestureId}\t{clickState}\t{action}\t{timeSinceInference * 1000:F2}\t{currentSeq}\t{seqGap}");
```

---

## Next Steps After Analysis

1. **If debounce is too aggressive:** Reduce `ENTER_CONFIRM_FRAMES` / `EXIT_CONFIRM_FRAMES` in [GrappleDetector.py:234-235](src/GrappleV2/tools/GrappleDetector.py#L234-L235)

2. **If hysteresis is too narrow:** Widen gap between `PINCH_THRESHOLD` and `PINCH_OPEN_THRESHOLD` in [GrappleDetector.py:201-202](src/GrappleV2/tools/GrappleDetector.py#L201-L202)

3. **If ratio is too noisy:** Increase `RATIO_SMOOTH_ALPHA` in [GrappleDetector.py:208](src/GrappleV2/tools/GrappleDetector.py#L208) (0.5 → 0.7 means more smoothing)

4. **If C# lags Python:** Check inference time, consider GPU acceleration (DirectML), or increase Python frame skip

5. **If extrapolation times out during drag:** Increase `MaxExtrapolationSec` in [MouseControllerNode.cs:51](src/GrappleV2/Grapple.Nodes/MouseControllerNode.cs#L51) to 200-250ms

---

**Last Updated:** 2025-01-31
