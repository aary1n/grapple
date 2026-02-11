# Git Workflow — Grapple + GrappleIntent

## Branch Naming

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feat/` | Production features (C# pipeline, IPC, integration) | `feat/onnx-runtime-backend` |
| `fix/` | Bug fixes | `fix/flatbuffer-alignment` |
| `research/` | ML research (architecture experiments, new methods) | `research/intent-fields-v2` |
| `exp/` | Short-lived experiments (may never merge) | `exp/lora-rank-sweep` |
| `infra/` | Tooling, CI/CD, build system, config | `infra/wandb-integration` |
| `docs/` | Documentation only | `docs/vla-architecture` |

## Commit Conventions

### Format
```
<type>: <short description>

<optional body — "why" not "what">

Co-Authored-By: ...
```

### Types
| Type | When |
|------|------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `perf` | Performance improvement (latency, throughput, memory) |
| `refactor` | Code restructuring without behavior change |
| `test` | Adding or updating tests |
| `docs` | Documentation changes |
| `exp` | Experiment results or research code (may be exploratory) |
| `data` | Dataset changes, synthetic data pipeline updates |
| `model` | Model architecture changes, new checkpoints |
| `config` | Configuration changes |
| `infra` | Build, CI/CD, tooling |

### Examples
```
feat: add reflexive path MobileNetV3 architecture

exp: sweep LoRA rank [4, 8, 16] on calibration dataset
Results: r=8 gives best accuracy/latency tradeoff (93.5% @ 4ms)

perf: quantize reflexive model to INT4-AWQ
P95 latency drops from 8ms to 4ms with <1% accuracy loss

model: export semantic transformer v0.3 to ONNX
Validated against PyTorch eager: max output delta 1e-5
```

## Research vs Production

### Research branches (`research/*`, `exp/*`)
- Can be messy — exploration is expected
- Squash-merge when promoting to main
- Include experiment results in merge commit body
- OK to force-push on personal experiment branches

### Production branches (`feat/*`, `fix/*`)
- Clean commit history
- Every commit should build and pass tests
- No force-push after PR is opened

## Experiment Branch Lifecycle

1. Branch from `main`: `git checkout -b research/intent-fields`
2. Develop, train, evaluate
3. Log results to W&B
4. If successful: open PR with results summary, squash-merge to `main`
5. If unsuccessful: document findings in PR description, close without merge
6. Delete branch after close

## What NOT to Commit

- Model weights (`.pt`, `.onnx`, `.safetensors`) — use W&B Artifacts
- Datasets — use versioned cloud storage or DVC
- W&B run directories (`wandb/`)
- Notebook outputs (use `nbstripout`)
- `.env` files with API keys
