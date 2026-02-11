# GrappleIntent Research Commands

Custom commands for ML research and model development workflows.

---

## Training Commands

### `/train`
Launch a training run with config.

**Implementation:**
```bash
# Reflexive path (MobileNetV3)
python src/GrappleIntent/training/train_reflexive.py --config configs/reflexive_v1.yaml

# Semantic path (VL-Transformer)
python src/GrappleIntent/training/train_semantic.py --config configs/semantic_v1.yaml
```

**Requirements:**
- Config YAML must exist in `src/GrappleIntent/configs/`
- W&B must be logged in (`wandb login`)
- All hyperparams in config, not CLI args

---

### `/calibrate`
Run LoRA calibration for a specific user.

**Implementation:**
```bash
python src/GrappleIntent/training/calibrate_lora.py --user-id <USER_ID> --anchor-data <PATH>
```

**What it does:**
1. Loads frozen base model
2. Fine-tunes LoRA adapter on anchor gesture data (5-10 poses)
3. Validates adapter doesn't regress base performance
4. Saves to `checkpoints/lora_adapter_{user_id}_v{N}.safetensors`
5. Logs calibration metrics to W&B

---

## Evaluation Commands

### `/eval`
Run evaluation suite on a trained model.

**Implementation:**
```bash
# Evaluate reflexive path
python src/GrappleIntent/evaluation/evaluate_accuracy.py --model <PATH> --dataset <PATH>

# Evaluate with latency
python src/GrappleIntent/evaluation/evaluate_accuracy.py --model <PATH> --dataset <PATH> --benchmark
```

**Output:**
- Accuracy, precision, recall per gesture class
- Confusion matrix (logged to W&B)
- If `--benchmark`: P50/P95/P99 latency

---

### `/benchmark`
Latency and throughput benchmarking on target hardware.

**Implementation:**
```bash
python src/GrappleIntent/evaluation/benchmark_latency.py --model <PATH> --runtime <onnx-cpu|onnx-directml|tensorrt|pytorch> --iterations 1000
```

**Output:**
```
Runtime: onnx-cpu
Warmup: 100 iterations
Measured: 1000 iterations
P50: 4.2ms | P95: 5.1ms | P99: 6.3ms
Throughput: 238 inf/sec
Memory: 42MB peak
```

---

### `/ablation`
Run ablation study template.

**Implementation:**
```bash
python src/GrappleIntent/evaluation/ablation.py --config <ABLATION_CONFIG>
```

**Ablation config format (YAML):**
```yaml
base_config: configs/reflexive_v1.yaml
sweep:
  - param: model.lora_rank
    values: [4, 8, 16, 32]
  - param: training.lr
    values: [1e-4, 3e-4, 1e-3]
metrics: [accuracy, inference_ms_p95]
```

**What it does:**
1. Generates all parameter combinations
2. Trains each variant (sequentially or parallel)
3. Logs comparison table to W&B
4. Prints summary with best config

---

## Model Conversion

### `/convert-onnx`
Export PyTorch model to ONNX.

**Implementation:**
```bash
python src/GrappleIntent/inference/export.py --model <PT_PATH> --output <ONNX_PATH> --opset 17
```

**What it does:**
1. Loads PyTorch model
2. Traces with dummy input
3. Exports to ONNX with specified opset
4. Validates ONNX model (onnx.checker)
5. Runs numerical comparison (PyTorch vs ONNX, max delta < 1e-5)

---

### `/quantize`
Quantize a model (AWQ or dynamic quantization).

**Implementation:**
```bash
# AWQ quantization (INT4 with salient weight preservation)
python src/GrappleIntent/inference/quantize.py --model <PATH> --method awq --output <PATH>

# ONNX dynamic quantization (INT8)
python src/GrappleIntent/inference/quantize.py --model <ONNX_PATH> --method onnx-dynamic --output <PATH>
```

**What it does:**
1. Loads model
2. Identifies salient weights (top 1% attention heads for AWQ)
3. Quantizes remaining weights
4. Validates accuracy within tolerance
5. Benchmarks latency before/after
6. Logs comparison to W&B

---

## Data Commands

### `/dataset-stats`
Dataset summary and validation.

**Implementation:**
```bash
python src/GrappleIntent/data/validate_dataset.py --path <DATASET_PATH>
```

**Output:**
```
Dataset: hand_gestures_v2
Samples: 12,450 (train: 9,960 | val: 1,245 | test: 1,245)
Classes: {point: 4,120, pinch: 3,890, open: 4,440}
Image size: 224x224 RGB
Integrity: OK (all files loadable, no NaN labels)
```

---

## Experiment Management

### `/experiment-log`
Log experiment metadata and results.

**Implementation:**
```bash
# View recent experiments
wandb runs list --project grapple-intent --last 10

# Compare two runs
wandb runs compare <RUN_ID_1> <RUN_ID_2> --metrics accuracy,inference_ms_p95
```

---

## Integration Commands

### `/intent-smoke`
End-to-end smoke test: synthetic input → GrappleIntent → FlatBuffer → C# consumer.

**Implementation:**
```bash
# Start C# consumer in background
dotnet run --project src/GrappleV2/Grapple.SmokeTests -- --mouse &

# Run GrappleIntent with synthetic data
python src/GrappleIntent/integration/smoke_test.py --duration 10

# Verify FlatBuffer output was consumed
```

**What to check:**
- FlatBuffer sensor frames are valid
- Cursor moves in response to intent predictions
- Latency within budget (reflexive ≤10ms, semantic ≤100ms)

---

## Quick Reference

| Command | Purpose | Duration |
|---------|---------|----------|
| `/train` | Launch training run | Minutes to hours |
| `/calibrate` | LoRA user calibration | ~1 minute |
| `/eval` | Evaluate model accuracy | ~30s |
| `/benchmark` | Latency benchmarking | ~30s |
| `/ablation` | Parameter sweep | Hours |
| `/convert-onnx` | PyTorch → ONNX export | ~10s |
| `/quantize` | Model quantization | ~1 minute |
| `/dataset-stats` | Validate dataset | ~5s |
| `/experiment-log` | View W&B experiments | Instant |
| `/intent-smoke` | Integration smoke test | ~15s |
