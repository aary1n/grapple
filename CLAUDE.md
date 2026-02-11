# Grapple - OS-Level Gesture Controller

## Project Manifest

**Version:** GrappleV2 (Clean-Room Implementation) + GrappleIntent (VLA Research)
**Status:** M4 Complete - Production Hardened | GrappleIntent Phase 0 (Infrastructure)
**Target:** Enterprise CAD Integration (SolidWorks First) → Hierarchical VLA-Driven HCI

---

## Architecture: "GrappleGraph"

Grapple is a **zero-GC, ultra-low-latency gesture-to-input pipeline** for professional CAD applications.

### Core Principles

1. **Zero Allocations in Hot Path**: No `new` keywords inside `Update()` loops. Gen 0 GC must remain at 0.
2. **LIFO Scheduling**: Latest frame wins. Drop backlog instantly if processing lags.
3. **Lock-Free IPC**: Interlocked operations only. No `lock`, no `Monitor`, no `SemaphoreSlim`.
4. **Type Safety**: No `object` or `dynamic`. Strict struct-based contracts.

### Latency Budget

**Target:** <30ms Motion-to-Photon | **Measured:** ~15-20ms

### Technology Stack

- **C# (.NET 9):** Self-contained desktop app. `unsafe`, `readonly struct`, `Span<byte>`, `Interlocked`. Deps: FlashCap, Microsoft.Extensions.Hosting.
- **Python (3.11.9):** Vision sensor sidecar. MediaPipe 0.10.x, numpy. Zero-copy shared memory IPC.
- **ML (GrappleIntent):** PyTorch ≥2.2, ONNX Runtime ≥1.17, transformers, timm, peft, autoawq, wandb. See `src/GrappleIntent/pyproject.toml`.

---

## Behavioral Rules

1. **Never guess library APIs** → fetch docs via context7 MCP first (see `docs/mcp-protocols.md`)
2. **Never assume code structure** → read files before suggesting changes
3. **Never suggest changes without reading existing implementation**
4. **When in doubt, fetch** → 3 tool calls and accurate beats 0 calls and wrong

---

## Reference Docs (read on-demand, not upfront)

Read these **only** when the task touches the relevant area:

| Working on...                | Read first                            |
|------------------------------|---------------------------------------|
| Pipeline / IPC / arenas      | `docs/architecture.md`                |
| Data contracts / structs     | `docs/PROTOCOL.md`                    |
| MCP tool usage               | `docs/mcp-protocols.md`               |
| Tuning / config              | `src/GrappleV2/grapple_config.json`   |
| Telemetry / observability    | `docs/telemetry.md`                   |
| VLA / ML research            | `.claude/rules/vla-architecture.md`   |
| ML experiment workflow       | `.claude/rules/ml-research.md`        |
| C# hot path code             | `.claude/rules/csharp-core.md`        |
| Python sensor code           | `.claude/rules/python-vision.md`      |
| Git branching / commits      | `.claude/rules/git-workflow.md`       |
| Build / run / test           | `.claude/commands/ops.md`             |
| Training / eval / export     | `.claude/commands/research.md`        |
| Feature planning             | `ROADMAP.md`                          |

---

## Build & Run (Quick Reference)

```bash
dotnet build src/GrappleV2/GrappleGraphs.sln                          # Build
dotnet test src/GrappleV2/Grapple.Tests/Grapple.Tests.csproj          # 55 tests
dotnet run --project src/GrappleV2/Grapple.SmokeTests -- --full       # Full pipeline
python src/GrappleV2/tools/GrappleDetector.py                          # Python detector
```

Prerequisites: .NET 9 SDK, Python 3.12, Visual C++ Redistributable

---

**Last Updated:** 2026-02-11
**Architecture Version:** GrappleGraph V2 (Phase 4 Complete) + GrappleIntent V0 (Infrastructure)
