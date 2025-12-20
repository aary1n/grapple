# Grapple Operations Commands

Custom commands for common Grapple development workflows.

---

## Build Commands

### `/build`
Build the entire Grapple solution.

**Implementation:**
```bash
dotnet build src/GrappleV2/GrappleGraphs.sln
```

**Output:**
- Compiles all projects (Core, Nodes, SmokeTests, Service)
- Validates struct sizes and unsafe code
- Reports build errors if any

---

## Run Commands

### `/run-full`
Run the complete Grapple pipeline (webcam → Python detector → mouse control).

**Implementation:**
```bash
dotnet run --project src/GrappleV2/Grapple.SmokeTests -- --full
```

**What it does:**
1. Initializes `SharedMemoryArena` (256MB)
2. Starts `WebcamCaptureNode` (1920×1080 @ 60fps)
3. Launches `GrappleDetector.py` subprocess
4. Starts `MouseControllerNode` (120Hz cursor updates)
5. Press **F9** to toggle cursor control
6. Press **Ctrl+C** to stop

**Expected Output:**
```
[*] Grapple Full Pipeline
[*] Launching webcam, Python detector, and mouse controller...
[*] Press F9 to toggle control ON/OFF
[Webcam] Initialized: 1920x1080 @ 60fps
[Python] Detector starting...
[Mouse] Controller ready (F9 to toggle)
[Stats] FPS=14.8, Latency=18.2ms
```

---

### `/run-webcam`
Test webcam capture only (no Python, no mouse).

**Implementation:**
```bash
dotnet run --project src/GrappleV2/Grapple.SmokeTests -- --webcam
```

**What it does:**
- Captures frames for 30 seconds
- Writes to `SharedMemoryArena`
- Publishes to `AtomicMailbox`
- Reports FPS and frame drops

---

### `/run-mouse`
Test mouse control with synthetic hand data (no webcam, no Python).

**Implementation:**
```bash
dotnet run --project src/GrappleV2/Grapple.SmokeTests -- --mouse
```

**What it does:**
- Simulates hand tracking data
- Moves cursor in a circular pattern
- Tests click detection
- Press **F9** to toggle

---

### `/run-py`
Run Python detector standalone (for debugging).

**Implementation:**
```bash
python src/GrappleV2/tools/GrappleDetector.py
```

**What it does:**
- Waits for C# to publish frames
- Runs MediaPipe inference
- Writes hand data to `HandResultArena`
- Logs latency and FPS

**Note:** Requires C# side to be running (e.g., `/run-full` or webcam test).

---

### `/run-tests`
Run all smoke tests (non-interactive validation).

**Implementation:**
```bash
dotnet run --project src/GrappleV2/Grapple.SmokeTests
```

**What it does:**
1. Control Plane Test (Mailbox LIFO behavior)
2. Producer Test (SyntheticCaptureNode zero-GC validation)
3. End-to-End Test (Producer → Consumer via Mailbox)
4. HandState struct size verification

**Expected Output:**
```
=== GRAPPLE SMOKE TESTS ===

[*] Test 1: Control Plane (Mailbox)
[+] SUCCESS: Mailbox behaves as LIFO

[*] Test 2: Producer (Zero-GC)
[+] SUCCESS: 0 Gen 0 collections

[*] Test 3: End-to-End
[+] SUCCESS: Pipeline ran for 15 seconds

[*] Test 4: HandState Struct
[+] SUCCESS: HandState is exactly 56 bytes

=== ALL SYSTEMS GO ===
```

---

## Analysis Commands

### `/log-analyze`
Analyze console output for latency spikes and performance issues.

**Implementation:**
```bash
# Run pipeline and capture logs
dotnet run --project src/GrappleV2/Grapple.SmokeTests -- --full 2>&1 | tee grapple.log

# Analyze latency spikes (>30ms)
grep "Latency:" grapple.log | awk '{print $NF}' | awk -F'ms' '{if($1 > 30) print "SPIKE: " $0}'

# Count dropped frames
grep "OVERWRITE" grapple.log | wc -l

# FPS statistics
grep "FPS=" grapple.log | tail -20
```

**What to look for:**
- Latency spikes >30ms (indicates frame drops or inference lag)
- "OVERWRITE" messages (mailbox overwrites due to backlog)
- FPS drops below 12 (Python inference struggling)

---

### `/gc-check`
Verify zero allocations in hot path.

**Implementation:**
```bash
dotnet run --project src/GrappleV2/Grapple.SmokeTests 2>&1 | grep "Gen 0"
```

**Expected Output:**
```
[+] SUCCESS: Producer ran for 15 seconds with 0 Gen 0 collections
```

**If Failed:**
- Check for `new` keywords in `Update()` loops
- Look for LINQ usage (allocates enumerators)
- Review string interpolation in hot path

---

## Debugging Commands

### `/debug-viewer`
Launch visual debug viewer (shows hand landmarks + cursor).

**Implementation:**
```bash
python src/GrappleV2/tools/debug_viewer.py
```

**What it shows:**
- Live webcam feed
- MediaPipe hand landmarks overlay
- Pinch distance visualization
- Current gesture state (Point/Pinch)

---

### `/clean`
Clean all build artifacts and temporary files.

**Implementation:**
```bash
dotnet clean src/GrappleV2/GrappleGraphs.sln
find src/GrappleV2 -type d -name "bin" -exec rm -rf {} + 2>/dev/null
find src/GrappleV2 -type d -name "obj" -exec rm -rf {} + 2>/dev/null
```

---

### `/kill-shared-memory`
Force-close all shared memory handles (if processes crashed).

**Implementation:**
```powershell
# Windows: Use Process Explorer or restart
# Or manually close via Task Manager

# Alternative: Reboot to clear orphaned memory-mapped files
```

**When to use:**
- After crashes leave shared memory locked
- Before starting clean test runs

---

## Performance Commands

### `/perf-capture`
Capture performance trace (requires dotnet-trace).

**Implementation:**
```bash
# Install dotnet-trace if not present
dotnet tool install --global dotnet-trace

# Start pipeline in background
dotnet run --project src/GrappleV2/Grapple.SmokeTests -- --full &
PID=$!

# Capture 30-second trace
dotnet-trace collect -p $PID --duration 00:00:30 --format speedscope

# Open in speedscope.app
```

**What to analyze:**
- Hot paths (should be in Arena/Mailbox methods)
- GC pauses (should be 0 during pipeline run)
- Thread contention (mailbox should have no locks)

---

### `/perf-flamegraph`
Generate CPU flamegraph (Linux/WSL only).

**Implementation:**
```bash
# Requires: perf, FlameGraph scripts
sudo perf record -F 99 -p $(pgrep -f Grapple.SmokeTests) -g -- sleep 30
sudo perf script | stackcollapse-perf.pl | flamegraph.pl > grapple-flamegraph.svg
```

---

## Quick Reference

| Command | Purpose | Duration |
|---------|---------|----------|
| `/build` | Compile solution | ~5s |
| `/run-full` | Full pipeline test | Until Ctrl+C |
| `/run-webcam` | Webcam capture test | 30s |
| `/run-mouse` | Mouse control test | Until Ctrl+C |
| `/run-py` | Python detector only | Until Ctrl+C |
| `/run-tests` | All smoke tests | ~30s |
| `/log-analyze` | Parse logs for issues | N/A |
| `/gc-check` | Verify zero-GC | ~15s |
| `/debug-viewer` | Visual debugger | Until Ctrl+C |
| `/clean` | Clean build artifacts | ~2s |

---

## Troubleshooting

### "Could not find file GrappleGraphs.sln"
**Fix:** Ensure you're in the repo root, not inside `src/GrappleV2`.

### "Python process failed to start"
**Fix:** Verify Python 3.12 is in PATH and dependencies are installed:
```bash
python --version  # Should be 3.12.x
pip list | grep mediapipe  # Should show 0.10.9
```

### "Shared memory access violation"
**Fix:** Kill orphaned processes:
```bash
# Windows
taskkill /F /IM Grapple.SmokeTests.exe
taskkill /F /IM python.exe

# Then rerun
```

### "Latency always >50ms"
**Check:**
- Webcam resolution (should be 1920×1080, not 4K)
- MediaPipe model complexity (should be 0, not 1)
- CPU usage (inference should use <30% of one core)

---

**Last Updated:** 2025-12-20
