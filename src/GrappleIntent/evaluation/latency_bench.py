"""Latency benchmarking for the reflexive path.

From vla-architecture.md §9:
    Rule: If any component exceeds its budget, it must be profiled and
    optimized before merging. Log per-component latency breakdowns.

Budget (120Hz, ≤10ms total):
    Frame read:     <0.5ms
    Preprocessing:  ≤1ms
    Inference:      ≤5ms
    Prototype:      <0.5ms
    Postprocessing: ≤0.5ms
    Blend:          <0.1ms
    FlatBuffer:     <0.5ms
    Headroom:       ~1.9ms
"""

from __future__ import annotations

import time
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LatencyProfile:
    """Per-component latency breakdown for a single inference."""
    preprocessing_ms: float = 0.0
    inference_ms: float = 0.0
    prototype_ms: float = 0.0
    postprocessing_ms: float = 0.0
    blend_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class BenchmarkResult:
    """Aggregate benchmark results over N iterations."""
    num_iterations: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    budget_ms: float
    within_budget: bool
    component_breakdown: dict[str, float]  # component → mean ms


class ReflexiveBenchmark:
    """Benchmark the reflexive inference path against latency budget."""

    def __init__(self, budget_ms: float = 10.0, warmup_iterations: int = 50):
        self.budget_ms = budget_ms
        self.warmup_iterations = warmup_iterations

    def run(
        self,
        infer_fn,
        num_iterations: int = 1000,
        input_dim: int = 66,
    ) -> BenchmarkResult:
        """Benchmark an inference function.

        Args:
            infer_fn: Callable(np.ndarray) → any — the function to benchmark
            num_iterations: Number of timed iterations
            input_dim: Input dimension for dummy data

        Returns:
            BenchmarkResult with latency statistics
        """
        dummy_input = np.random.randn(input_dim).astype(np.float32)

        # Warmup
        for _ in range(self.warmup_iterations):
            infer_fn(dummy_input)

        # Timed iterations
        latencies: list[float] = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            infer_fn(dummy_input)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

        latencies.sort()
        n = len(latencies)

        return BenchmarkResult(
            num_iterations=n,
            p50_ms=latencies[n // 2],
            p95_ms=latencies[int(n * 0.95)],
            p99_ms=latencies[int(n * 0.99)],
            mean_ms=statistics.mean(latencies),
            min_ms=latencies[0],
            max_ms=latencies[-1],
            budget_ms=self.budget_ms,
            within_budget=latencies[int(n * 0.99)] <= self.budget_ms,
            component_breakdown={},  # Filled by component-level profiling
        )

    def report(self, result: BenchmarkResult) -> str:
        """Generate a human-readable benchmark report."""
        # ASCII only — Windows consoles often run cp1252
        status = "PASS" if result.within_budget else "FAIL"
        lines = [
            f"Reflexive Path Benchmark {status}",
            f"  Iterations: {result.num_iterations}",
            f"  Budget:     {result.budget_ms:.1f}ms",
            f"  P50:        {result.p50_ms:.2f}ms",
            f"  P95:        {result.p95_ms:.2f}ms",
            f"  P99:        {result.p99_ms:.2f}ms",
            f"  Mean:       {result.mean_ms:.2f}ms",
            f"  Min/Max:    {result.min_ms:.2f}ms / {result.max_ms:.2f}ms",
        ]
        if result.component_breakdown:
            lines.append("  Component breakdown:")
            for comp, ms in result.component_breakdown.items():
                lines.append(f"    {comp}: {ms:.2f}ms")
        return "\n".join(lines)


# ─── CLI entry point ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import argparse
    import logging
    import sys

    from ..configs import load_config
    from ..inference.reflexive_engine import ReflexiveEngine

    for stream in (sys.stdout, sys.stderr):
        if stream and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Benchmark reflexive path latency")
    parser.add_argument(
        "--onnx",
        default="checkpoints/reflexive/mobilenetv3_cursor_v0.1_fp32.onnx",
        help="Path to exported ONNX model",
    )
    parser.add_argument("--config", default=None, help="GrappleIntent YAML config")
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    config = load_config(args.config)
    ic = config.reflexive.inference

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"[!] ONNX model not found: {onnx_path}")
        print("    Export first: python -m GrappleIntent.inference.export_onnx")
        return 1

    engine = ReflexiveEngine(
        onnx_path=onnx_path,
        watchdog_threshold_ms=ic.watchdog_threshold_ms,
        watchdog_recovery_frames=ic.watchdog_recovery_frames,
    )
    engine.load()

    bench = ReflexiveBenchmark(budget_ms=ic.latency_budget_ms)
    result = bench.run(engine.infer, num_iterations=args.iterations)
    print(bench.report(result))

    target_status = "within" if result.p95_ms <= ic.latency_target_ms else "above"
    print(f"  P95 is {target_status} the {ic.latency_target_ms:.1f}ms design target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
