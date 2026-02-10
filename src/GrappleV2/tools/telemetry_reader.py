"""
Grapple Telemetry Reader — LAM-readable shared memory consumer.

Reads TelemetrySnapshot FlatBuffers from the Grapple telemetry arena
and prints them as JSON for external consumption (LAMs, dashboards, etc.)

Usage:
    python telemetry_reader.py [--interval 0.5] [--json]

The telemetry arena is at: Local\\GrappleTelemetry (4096 bytes)
Arena header: 32 bytes (magic=0x4C505247, seq, version, bufferSize, freq)
Data starts at offset 64.
"""

import sys
import os
import struct
import time
import json
import argparse
import mmap

# Add generated FlatBuffer bindings to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "generated"))

from Grapple.Protocol.TelemetrySnapshot import TelemetrySnapshot

# Arena constants (must match C# TelemetryArena)
MAP_NAME = "Local\\GrappleTelemetry"
MAP_CAPACITY = 4096
HEADER_SIZE = 32  # FlatBufferArenaHeader: magic(8) + seq(8) + version(4) + bufferSize(4) + freq(8)
DATA_OFFSET = 64
MAGIC = 0x4C505247  # "GRPL"

HEADER_FORMAT = "<QqiiQ"  # magic, sequence, version, bufferSize, freq
assert struct.calcsize(HEADER_FORMAT) == HEADER_SIZE


def open_telemetry_arena():
    """Open the existing telemetry shared memory arena (read-only)."""
    try:
        shm = mmap.mmap(-1, MAP_CAPACITY, MAP_NAME, access=mmap.ACCESS_READ)
        return shm
    except Exception as e:
        print(f"ERROR: Cannot open telemetry arena '{MAP_NAME}': {e}", file=sys.stderr)
        print("Is the Grapple pipeline running? (dotnet run -- --full)", file=sys.stderr)
        sys.exit(1)


def read_header(shm):
    """Read the arena header and return (magic, sequence, version, bufferSize, freq)."""
    shm.seek(0)
    raw = shm.read(HEADER_SIZE)
    return struct.unpack(HEADER_FORMAT, raw)


def read_snapshot(shm, buffer_size):
    """Read and decode the FlatBuffer TelemetrySnapshot from the arena."""
    if buffer_size <= 0 or buffer_size > (MAP_CAPACITY - DATA_OFFSET):
        return None

    shm.seek(DATA_OFFSET)
    raw = shm.read(buffer_size)

    snapshot = TelemetrySnapshot.GetRootAs(raw, 0)
    return snapshot


def snapshot_to_dict(snap):
    """Convert a TelemetrySnapshot to a plain dict for JSON output."""
    return {
        "fps": round(snap.Fps(), 1),
        "latency_ms": round(snap.LatencyMs(), 2),
        "latency_p50_ms": round(snap.LatencyP50Ms(), 2),
        "latency_p95_ms": round(snap.LatencyP95Ms(), 2),
        "latency_p99_ms": round(snap.LatencyP99Ms(), 2),
        "dropped_frames": snap.DroppedFrames(),
        "total_frames_produced": snap.TotalFramesProduced(),
        "total_frames_dropped": snap.TotalFramesDropped(),
        "consecutive_drops": snap.ConsecutiveDrops(),
        "quality_degradation": snap.QualityDegradationActive(),
        "gc_gen0": snap.GcGen0Collections(),
        "gc_gen1": snap.GcGen1Collections(),
        "gc_gen2": snap.GcGen2Collections(),
        "uptime_s": round(snap.UptimeSeconds(), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Grapple Telemetry Reader")
    parser.add_argument("--interval", type=float, default=0.5, help="Poll interval in seconds (default: 0.5)")
    parser.add_argument("--json", action="store_true", help="Output JSON lines (one per poll)")
    parser.add_argument("--once", action="store_true", help="Read once and exit")
    args = parser.parse_args()

    shm = open_telemetry_arena()
    last_seq = -1

    print(f"Grapple Telemetry Reader — polling every {args.interval}s", file=sys.stderr)
    print(f"Arena: {MAP_NAME} ({MAP_CAPACITY} bytes)", file=sys.stderr)

    try:
        while True:
            magic, seq, version, buffer_size, freq = read_header(shm)

            if magic != MAGIC:
                print("Waiting for arena initialization...", file=sys.stderr)
                time.sleep(1)
                continue

            if seq == last_seq:
                if args.once:
                    break
                time.sleep(args.interval)
                continue

            last_seq = seq

            snapshot = read_snapshot(shm, buffer_size)
            if snapshot is None:
                if args.once:
                    break
                time.sleep(args.interval)
                continue

            data = snapshot_to_dict(snapshot)
            data["_seq"] = seq

            if args.json:
                print(json.dumps(data), flush=True)
            else:
                print(
                    f"[seq={seq:>6}] "
                    f"FPS: {data['fps']:>5.1f} | "
                    f"Latency P50/P95/P99: {data['latency_p50_ms']:>6.1f}/{data['latency_p95_ms']:>6.1f}/{data['latency_p99_ms']:>6.1f}ms | "
                    f"Produced: {data['total_frames_produced']:>8} | "
                    f"Dropped: {data['total_frames_dropped']:>6} | "
                    f"GC: {data['gc_gen0']}/{data['gc_gen1']}/{data['gc_gen2']} | "
                    f"Up: {data['uptime_s']:>6.1f}s"
                    + (" [DEGRADED]" if data['quality_degradation'] else ""),
                    flush=True,
                )

            if args.once:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    finally:
        shm.close()


if __name__ == "__main__":
    main()
