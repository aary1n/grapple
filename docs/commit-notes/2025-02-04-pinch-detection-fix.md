# Fix: Pinch Detection Ratio Poisoning

## Problem

Click detection was unreliable. After the first successful pinch, subsequent pinch attempts took 10-15 seconds to register, and drag-while-pinching would frequently drop the click state.

## Root Cause: Ratio Poisoning

The pinch detector uses a normalized ratio (`pinch_distance^2 / index_finger_length^2`) passed through an EMA to determine pinch state. Three issues compounded:

### 1. Spike Poisoning via EMA

Brief tracking glitches (hand edge-on, 1-frame dropout) produced raw ratio values of 1.0+. With a symmetric EMA (alpha=0.5), a single spike to 1.2 required 3+ frames of clean data just to mathematically recover below the 0.30 entry threshold:

```
smoothed = 1.200
  -> 0.5 * 0.05 + 0.5 * 1.200 = 0.625
  -> 0.5 * 0.05 + 0.5 * 0.625 = 0.338
  -> 0.5 * 0.05 + 0.5 * 0.338 = 0.194  (finally below 0.30)
```

In practice, another spike would arrive before recovery completed, restarting the process. This created 15+ second dead zones where the user was physically pinching but the smoothed ratio stayed above threshold.

### 2. MAX_RATIO Clamp Too Permissive

The clamp was set at 1.2 (20% above "open hand"). This allowed spikes to push the EMA far above the exit threshold, maximizing recovery time.

### 3. Hand-Loss Resets Smoothed Ratio

A single frame of `NO_HAND` (common during fast movement) reset `smoothed_ratio` to 1.0, forcing the entire convergence process to restart from scratch.

## Diagnosis Method

Added tab-separated per-frame diagnostic logging to both Python and C# sides of the pipeline:

**Python (GrappleDetector.py):** `PY\t{frame}\t{raw_dist}\t{smoothed_ratio}\t{fsm_state}\t{entry_counter}\t{exit_counter}\t{transition}\t{gesture_id}`

**C# (MouseControllerNode.cs):** `CS\t{frame}\t{gesture_id}\t{click_state}\t{action}\t{stale_ms}\t{seq}\t{seq_gap}`

Analysis of captured logs showed:
- The FSM and debounce logic were functioning correctly (3 frames to enter, 5 to exit)
- The smoothed ratio was the sole bottleneck -- stuck above 0.30 for 150+ consecutive frames due to repeated spike contamination
- C# side had zero sequence gaps and sub-50ms staleness -- no IPC issues

## Fixes Applied (GrappleDetector.py)

### Fix 1: Asymmetric EMA Smoothing

Replaced single-alpha EMA with direction-dependent smoothing:

```python
# Before
smoothed_ratio = 0.5 * raw_ratio + 0.5 * smoothed_ratio

# After
if raw_ratio < smoothed_ratio:
    alpha = 0.5    # Fast tracking downward (user pinching)
else:
    alpha = 0.3    # Slower tracking upward (resist noise spikes)
smoothed_ratio = alpha * raw_ratio + (1.0 - alpha) * smoothed_ratio
```

This makes the ratio responsive to genuine pinch gestures while dampening upward noise. A single-frame spike from 0.2 to 0.9 only moves the ratio to 0.41 (vs 0.55 with symmetric alpha).

### Fix 2: Lower MAX_RATIO Clamp (1.2 -> 0.9)

Reduced the ceiling on raw ratio values. Key constraint: must be above `PINCH_OPEN_THRESHOLD` (0.70) so the exit condition remains reachable. 0.9 limits spike damage while preserving exit behavior.

Recovery comparison (worst case, from max clamp to threshold at 0.30):
- Old (1.2, symmetric): ~3 clean frames minimum
- New (0.9, asymmetric down): ~2 clean frames

### Fix 3: Hand-Loss Grace Period

Instead of immediately resetting `smoothed_ratio` to 1.0 on any `NO_HAND` frame, track consecutive no-hand frames and only reset after 5 consecutive frames (~500ms at 10fps inference):

```python
no_hand_streak += 1
if no_hand_streak >= 5:
    smoothed_ratio = 1.0  # Sustained loss -- full reset
# Otherwise: keep previous smoothed_ratio
```

This prevents 1-frame tracking dropouts (common during fast hand movement) from destroying the accumulated ratio state.

## Supporting Changes (MouseControllerNode.cs)

Added `SequenceNumber` field to `ExtrapolationState` struct to enable frame-drop detection in diagnostic logs. The sequence number flows from Python through the IPC layer to the C# cursor loop, allowing side-by-side correlation of Python inference frames with C# cursor updates.

## Verification

Diagnostic logging confirmed:
- Pinch entry now occurs within 3-5 frames of physical pinch (~300-500ms)
- Pinch hold is stable during circle-drawing drag gestures
- Release triggers within ~9 frames of hand open (~900ms)
- Single-frame hand-loss no longer resets ratio state

## Files Changed

- `src/GrappleV2/tools/GrappleDetector.py` -- Asymmetric EMA, lower clamp, grace period, diagnostic logging
- `src/GrappleV2/Grapple.Nodes/MouseControllerNode.cs` -- Sequence tracking in ExtrapolationState, diagnostic logging

## Tuning Reference

Current values after this fix:

| Parameter | Value | Location |
|-----------|-------|----------|
| PINCH_THRESHOLD | 0.30 | GrappleDetector.py:201 |
| PINCH_OPEN_THRESHOLD | 0.70 | GrappleDetector.py:202 |
| MAX_RATIO | 0.9 | GrappleDetector.py:548 |
| RATIO_ALPHA_DOWN | 0.5 | GrappleDetector.py:209 |
| RATIO_ALPHA_UP | 0.3 | GrappleDetector.py:212 |
| ENTER_CONFIRM_FRAMES | 3 | PinchStateMachine:237 |
| EXIT_CONFIRM_FRAMES | 5 | PinchStateMachine:238 |
| NO_HAND_GRACE | 5 | GrappleDetector.py:428 |

**Date:** 2025-02-04
