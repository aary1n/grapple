# Grapple Performance Telemetry

## Real-Time Metrics (via TelemetryCollector → TelemetryArena)

- FPS (frames processed per second)
- End-to-end latency: P50, P95, P99 (ms)
- Total frames produced / dropped
- Consecutive drops + quality degradation flag
- GC Gen 0/1/2 collection counts
- Uptime (seconds)

## Reading Telemetry

```bash
# Human-readable (polls every 0.5s)
python src/GrappleV2/tools/telemetry_reader.py

# JSON-lines (for LAM / dashboard consumption)
python src/GrappleV2/tools/telemetry_reader.py --json

# Single snapshot
python src/GrappleV2/tools/telemetry_reader.py --once --json
```

## Validation

Run unit + integration tests (55 tests):
```bash
dotnet test src/GrappleV2/Grapple.Tests/Grapple.Tests.csproj
```

Run smoke tests to verify zero-GC:
```bash
dotnet run --project src/GrappleV2/Grapple.SmokeTests
```

Expected output:
```
[+] SUCCESS: Producer ran for 15 seconds with 0 Gen 0 collections
[+] SUCCESS: HandState is exactly 56 bytes
```
