# ADR-001: Split Calibration Strategy — Metric Learning (Reflexive) vs LoRA (Semantic)

## Status

**Accepted**

---

## Context

GrappleIntent's user calibration requires personalization from only 5-10 anchor gestures. The original design proposed LoRA fine-tuning for both inference paths. This creates a critical instability risk on the reflexive path:

**Problem:** SGD-based LoRA fine-tuning on 5-10 samples is prone to:
- **Catastrophic forgetting** — the adapter overfits to the few anchors and degrades on the general gesture distribution
- **Training instability** — loss oscillation and convergence failure with a sample count this low
- **Latency risk** — any adapter loading/switching overhead on the reflexive path eats into the 10ms hard budget

The reflexive path (120Hz, ≤10ms) needs stable, instant calibration. The semantic path (10Hz, ≤100ms) has budget for richer personalization.

---

## Decision

### Reflexive Path → Prototypical Networks (Metric Learning)

Use a pre-trained embedding network to compute frozen prototype vectors from the anchor gestures. At runtime, classify new inputs via cosine distance to the stored prototypes.

**Why this works for reflexive:**
- No weight updates → no forgetting, no training instability
- Calibration is a forward pass (seconds), not optimization (minutes)
- Nearest-neighbor lookup adds <0.5ms — fits within the 5ms target
- Embedding model is small (same MobileNetV3 backbone) and CPU-friendly
- Well-studied in few-shot learning literature (Snell et al., 2017)

**Why NOT LoRA for reflexive:**
- 5-10 samples is below the minimum viable training set for stable SGD on even a low-rank adapter
- The reflexive path's simple output (cursor delta + gesture class) doesn't require the expressiveness of fine-tuning
- Any adapter switch overhead (even 100ms) is unacceptable at 120Hz

### Semantic Path → LoRA Fine-Tuning

Keep LoRA adapters on the frozen VL-Transformer backbone. The semantic path's richer output (intent classification, gradient fields) benefits from parametric personalization.

**Why this works for semantic:**
- 100ms latency budget accommodates LoRA adapter loading (≤500ms, amortized at startup/switch)
- Intent parsing requires nuanced personalization that nearest-neighbor can't capture (e.g., user-specific gesture → intent mappings)
- Rank r=8 LoRA modifies <0.5% of parameters — forgetting risk is manageable with proper validation
- Fine-tuning takes ~1 minute — acceptable for a one-time calibration

**Why NOT metric learning for semantic:**
- The semantic output space (intent probabilities, heatmaps) is too rich for prototype matching
- Intent boundaries between users are not cleanly separable in a fixed embedding space

### Both Paths: Augmentation Requirement

The 5-10 raw anchor gestures must be augmented before any learning:
- **Temporal jittering:** Shift gesture timestamps ±50ms to simulate timing variation
- **Synthetic perturbations:** Add Gaussian noise to landmark positions, vary gesture speed
- **Result:** Expand effective training set from 5-10 to ~100-200 augmented samples

This is non-negotiable for both metric learning (prototype stability) and LoRA (convergence).

---

## Consequences

### Positive
- Reflexive calibration is **instant and deterministic** — no training loop, no convergence risk
- Hard compute isolation preserved — reflexive stays pure CPU with no adapter loading overhead
- Each path uses the personalization method matched to its latency budget and output complexity
- System degrades gracefully — if calibration fails, base models still work (uncalibrated)

### Negative
- Two calibration codepaths to maintain (metric learning + LoRA)
- Reflexive personalization is **limited in expressiveness** — can only classify gestures the embedding model already understands; novel gesture types require retraining the embedding model
- Dependency on `peft` library remains (for semantic LoRA), but is scoped to semantic path only
- Prototype storage format (`.npz`) is different from adapter format (`.safetensors`) — two artifact types per user

### Risks to Monitor
- If prototype accuracy proves insufficient on real users → consider expanding to more anchor gestures (10-20) before adding LoRA to reflexive
- If augmentation pipeline introduces distribution shift → validate augmented prototypes against held-out real data

---

## References

- Snell, J., Swersky, K., Zemel, R. (2017). "Prototypical Networks for Few-shot Learning." NeurIPS.
- Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
