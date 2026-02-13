"""Potential field blending — fuses reflexive and semantic cursor control.

From vla-architecture.md §1:
    cursor_delta = reflexive_delta + α · semantic_gradient

    α = sigmoid(k · (semantic_confidence - c₀)) × exp(-Δt / τ)

Invariants:
    - The reflexive path ALWAYS contributes — it is never suppressed
    - The semantic gradient modulates, never overrides
    - Active reflexive gestures (pinch/drag) naturally dominate via large reflexive_delta
    - When no fresh semantic prediction arrives, α decays exponentially toward 0
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class BlendedDelta:
    """Result of blending reflexive and semantic cursor contributions."""

    dx: float
    dy: float
    alpha: float  # Current blending coefficient (0 = pure reflexive, 1 = max semantic)
    reflexive_dx: float
    reflexive_dy: float
    semantic_dx: float  # Raw semantic gradient before alpha scaling
    semantic_dy: float


class PotentialFieldBlender:
    """Continuous potential field blending between reflexive and semantic paths.

    The blending coefficient α is smoothly modulated by:
        1. Semantic confidence (sigmoid activation)
        2. Temporal staleness (exponential decay)

    There are no hard thresholds or binary mode switches.
    """

    def __init__(
        self,
        gain_k: float = 5.0,
        offset_c0: float = 0.5,
        decay_tau: float = 0.3,
    ):
        """
        Args:
            gain_k: Sigmoid steepness — higher = sharper transition around c0
            offset_c0: Confidence threshold center — α ≈ 0.5 when confidence = c0
            decay_tau: Temporal decay time constant in seconds
        """
        self.gain_k = gain_k
        self.offset_c0 = offset_c0
        self.decay_tau = decay_tau

        self._last_semantic_time: float | None = None
        self._last_semantic_confidence: float = 0.0
        self._last_semantic_gradient: tuple[float, float] = (0.0, 0.0)

    def update_semantic(
        self,
        gradient_dx: float,
        gradient_dy: float,
        confidence: float,
    ) -> None:
        """Update the latest semantic path output.

        Called at ~10Hz when the semantic path produces a new prediction.
        """
        self._last_semantic_time = time.perf_counter()
        self._last_semantic_confidence = confidence
        self._last_semantic_gradient = (gradient_dx, gradient_dy)

    def blend(self, reflexive_dx: float, reflexive_dy: float) -> BlendedDelta:
        """Blend reflexive delta with semantic gradient.

        Called at ~120Hz (every reflexive frame).

        Args:
            reflexive_dx: Cursor delta from reflexive path
            reflexive_dy: Cursor delta from reflexive path

        Returns:
            BlendedDelta with final cursor delta and diagnostics
        """
        alpha = self._compute_alpha()
        sem_dx, sem_dy = self._last_semantic_gradient

        return BlendedDelta(
            dx=reflexive_dx + alpha * sem_dx,
            dy=reflexive_dy + alpha * sem_dy,
            alpha=alpha,
            reflexive_dx=reflexive_dx,
            reflexive_dy=reflexive_dy,
            semantic_dx=sem_dx,
            semantic_dy=sem_dy,
        )

    def _compute_alpha(self) -> float:
        """Compute blending coefficient α.

        α = sigmoid(k × (confidence - c₀)) × exp(-Δt / τ)
        """
        # Confidence component: sigmoid activation
        confidence_alpha = _sigmoid(
            self.gain_k * (self._last_semantic_confidence - self.offset_c0)
        )

        # Temporal decay: exponential staleness
        if self._last_semantic_time is None:
            return 0.0  # No semantic data yet

        dt = time.perf_counter() - self._last_semantic_time
        staleness = math.exp(-dt / self.decay_tau)

        return confidence_alpha * staleness

    def reset(self) -> None:
        """Reset semantic state — used when semantic path crashes or restarts."""
        self._last_semantic_time = None
        self._last_semantic_confidence = 0.0
        self._last_semantic_gradient = (0.0, 0.0)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)
