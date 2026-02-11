# ML Research Rules — GrappleIntent

## Scope
These rules govern all code under `src/GrappleIntent/`. They complement (never override) the zero-GC rules in `csharp-core.md` — those still govern the C# hot path that GrappleIntent models feed into.

---

## 1. Reproducibility is Non-Negotiable

**Rule:** Every training run must be fully reproducible from its logged config.

### Requirements
- Pin random seeds everywhere: `torch.manual_seed()`, `np.random.seed()`, `random.seed()`, CUDA deterministic mode
- Log the full config (hyperparams, data splits, model architecture, seed) to W&B at run start
- Use config files (YAML or dataclass), never hardcoded hyperparams scattered in code
- Tag every run with git commit hash

### Forbidden
```python
# ❌ BAD: Magic numbers in training loop
lr = 0.0003
batch_size = 32

# ✅ GOOD: Config-driven
@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 32
    seed: int = 42
```

---

## 2. Model Naming & Versioning

**Rule:** Use semantic naming: `{arch}_{task}_{version}_{quantization}`

### Examples
```
mobilenetv3_cursor_v1.0_fp32.pt
mobilenetv3_cursor_v1.0_int4-awq.onnx
vl_transformer_intent_v0.3_fp16.safetensors
lora_adapter_user42_v1.safetensors
prototypes_user42_v1.npz
```

### Checkpoint Hygiene
- Save config YAML alongside every checkpoint
- Never commit model weights to git (use `.claudeignore` + W&B Artifacts)
- Keep only last N checkpoints locally (configurable, default 3)

---

## 3. Training vs Inference Separation

**Rule:** Training code and inference code live in separate modules with a clear export boundary.

```
src/GrappleIntent/
├── training/          # Training loops, data loaders, loss functions
├── models/            # Architecture definitions (shared by both)
├── inference/         # Optimized inference paths
├── data/              # Data loading, augmentation, synthetic pipeline
├── evaluation/        # Metrics, benchmarks, ablation scripts
└── integration/       # Bridge to C# IPC (FlatBuffer writers)
```

### Export Contract
The **only** output of `inference/` that touches the C# pipeline is a FlatBuffer-compatible struct written to shared memory. This is the integration boundary.

---

## 4. W&B Integration

**Rule:** Every experiment must log to Weights & Biases. No silent runs.

### Required Logging
```python
import wandb

wandb.init(
    project="grapple-intent",
    config=config.__dict__,
    tags=["reflexive" | "semantic", "v0.x"],
)

# Per-step: loss, lr, grad norm
wandb.log({"loss": loss, "lr": scheduler.get_last_lr()[0]})

# Per-epoch: val metrics, latency benchmarks
wandb.log({"val_accuracy": acc, "inference_ms_p95": p95})

# End: model artifact
wandb.log_artifact(checkpoint_path, type="model")
```

### Run Naming
`{task}-{arch}-{short_description}` e.g., `cursor-mobilenetv3-lora-calibration`

---

## 5. Latency-Aware Development

**Rule:** Always benchmark inference latency alongside accuracy. A model that's accurate but slow is useless for Grapple.

### Latency Budgets
| Path | Target | Hard Limit |
|------|--------|------------|
| Reflexive (cursor) | ≤5ms | 10ms |
| Semantic (intent) | ≤50ms | 100ms |
| Semantic LoRA switch | ≤200ms | 500ms |

### Benchmark Protocol
- Measure on target hardware (not just dev machine)
- Report P50/P95/P99 (not just mean)
- Warm up for 100 iterations before measuring
- Include preprocessing + postprocessing in latency
- Log to W&B as `inference_ms_p50`, `inference_ms_p95`, `inference_ms_p99`

---

## 6. Data Pipeline Conventions

### Synthetic Data (Unity/Isaac)
- Version datasets with content hashes
- Document generation parameters (lighting, hand pose distribution, occlusion %)
- Validate schema before training: assert expected fields, shapes, dtypes

### Real Data
- Never store PII (faces, identifying info) — hands only
- Document consent/collection context
- Train/val/test splits must be deterministic and logged

---

## 7. Quantization Workflow

**Rule:** Quantization is not an afterthought. Test quantized models early and often.

### AWQ Pipeline
1. Train in FP32/FP16
2. Identify salient weights (top 1% attention heads)
3. Quantize remainder to INT4
4. Validate accuracy drop < threshold
5. Benchmark latency on target runtime
6. Export to ONNX with quantization ops

---

## 8. Notebook Discipline

**Rule:** Notebooks are for exploration only. Production code lives in `.py` files.

- Notebooks go in `src/GrappleIntent/notebooks/` (outputs gitignored)
- Once an experiment succeeds in a notebook, extract to a proper training script
- Never import from notebooks
- Clear outputs before committing

---

## 9. Documentation for Context7

**Rule:** When using unfamiliar ML library APIs, fetch docs first via context7 MCP.

Before writing code that uses:
- PyTorch quantization → `context7: PyTorch quantization`
- ONNX Runtime → `context7: ONNX Runtime inference`
- HuggingFace PEFT/LoRA → `context7: PEFT LoRA adapters`
- Prototypical Networks → `context7: metric learning prototypical networks`
- DirectML → `context7: DirectML ONNX Runtime`
- AWQ → `context7: AutoAWQ quantization`

This is inherited from the existing Grapple MCP protocol — never guess library APIs.

---

## 10. Testing ML Code

### Unit Tests
- Model forward pass produces correct output shape
- Data loader yields expected batch format
- Preprocessing/postprocessing are invertible where applicable
- Config parsing handles defaults and overrides

### Integration Tests
- End-to-end: synthetic input → model → FlatBuffer output → C# consumption
- Latency regression: P95 doesn't regress >10% between commits
- Quantized model parity: accuracy within tolerance of full-precision
