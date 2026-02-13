"""Graceful degradation state machine.

From vla-architecture.md §10:
    Principle: The cursor must always move.

    L0: Full VLA — both paths active, potential field blending (α > 0)
    L1: Semantic unavailable — pure reflexive control, α = 0
    L2: Reflexive over-budget — raw landmark passthrough, no model inference
    L3: Both paths degraded — fall back to GrappleDetector.py rule-based system

Recovery:
    L1 → L0: Background reload, α ramps from 0 over 1 second (no jerk)
    L2 → L0/L1: Resume when latency below 8ms for 10 consecutive frames
    L3 → L2/L1: Requires process restart or operator intervention
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class DegradationLevel(enum.IntEnum):
    L0_FULL_VLA = 0
    L1_SEMANTIC_UNAVAILABLE = 1
    L2_REFLEXIVE_OVER_BUDGET = 2
    L3_BOTH_DEGRADED = 3


@dataclass
class DegradationEvent:
    """Logged on every state transition."""

    previous_level: DegradationLevel
    new_level: DegradationLevel
    trigger_reason: str
    timestamp: float  # time.perf_counter()


class DegradationManager:
    """Manages the VLA degradation hierarchy and recovery transitions.

    All transitions are logged at Warning (L1) or Critical (L2, L3) level
    per architecture rules.
    """

    def __init__(self, semantic_ramp_seconds: float = 1.0):
        self._level = DegradationLevel.L1_SEMANTIC_UNAVAILABLE  # Start without semantic
        self._semantic_ramp_seconds = semantic_ramp_seconds
        self._recovery_start_time: float | None = None
        self._history: list[DegradationEvent] = []

    @property
    def level(self) -> DegradationLevel:
        return self._level

    @property
    def alpha_multiplier(self) -> float:
        """Multiplier applied to blending α during semantic recovery ramp.

        Returns 1.0 during normal operation, 0.0 when semantic is unavailable,
        and ramps linearly from 0→1 during L1→L0 recovery.
        """
        if self._level != DegradationLevel.L0_FULL_VLA:
            return 0.0

        if self._recovery_start_time is None:
            return 1.0

        elapsed = time.perf_counter() - self._recovery_start_time
        if elapsed >= self._semantic_ramp_seconds:
            self._recovery_start_time = None  # Ramp complete
            return 1.0

        return elapsed / self._semantic_ramp_seconds

    def transition(self, new_level: DegradationLevel, reason: str) -> None:
        """Transition to a new degradation level."""
        if new_level == self._level:
            return

        event = DegradationEvent(
            previous_level=self._level,
            new_level=new_level,
            trigger_reason=reason,
            timestamp=time.perf_counter(),
        )
        self._history.append(event)

        # Log at appropriate severity
        if new_level >= DegradationLevel.L2_REFLEXIVE_OVER_BUDGET:
            logger.critical(
                "Degradation %s → %s: %s", self._level.name, new_level.name, reason
            )
        elif new_level >= DegradationLevel.L1_SEMANTIC_UNAVAILABLE:
            logger.warning(
                "Degradation %s → %s: %s", self._level.name, new_level.name, reason
            )
        else:
            logger.info(
                "Recovery %s → %s: %s", self._level.name, new_level.name, reason
            )

        # Handle recovery ramp for L1 → L0
        if (
            self._level >= DegradationLevel.L1_SEMANTIC_UNAVAILABLE
            and new_level == DegradationLevel.L0_FULL_VLA
        ):
            self._recovery_start_time = time.perf_counter()

        self._level = new_level

    # ── Convenience triggers ──────────────────────────────────────────────

    def on_semantic_loaded(self) -> None:
        """Semantic model successfully loaded/reloaded."""
        if self._level == DegradationLevel.L1_SEMANTIC_UNAVAILABLE:
            self.transition(DegradationLevel.L0_FULL_VLA, "Semantic model loaded")

    def on_semantic_failed(self, reason: str = "Semantic model unavailable") -> None:
        """Semantic path crashed, OOM, GPU error, etc."""
        if self._level == DegradationLevel.L0_FULL_VLA:
            self.transition(DegradationLevel.L1_SEMANTIC_UNAVAILABLE, reason)

    def on_reflexive_over_budget(self) -> None:
        """Reflexive watchdog triggered."""
        if self._level <= DegradationLevel.L1_SEMANTIC_UNAVAILABLE:
            self.transition(
                DegradationLevel.L2_REFLEXIVE_OVER_BUDGET, "Reflexive latency > 8ms"
            )

    def on_reflexive_recovered(self) -> None:
        """Reflexive watchdog recovered."""
        if self._level == DegradationLevel.L2_REFLEXIVE_OVER_BUDGET:
            # Recover to L1 or L0 depending on semantic availability
            # Caller should also call on_semantic_loaded() if semantic is ready
            self.transition(
                DegradationLevel.L1_SEMANTIC_UNAVAILABLE, "Reflexive latency recovered"
            )

    def on_both_failed(self, reason: str = "Both paths degraded") -> None:
        """Both paths are down — need rule-based fallback."""
        self.transition(DegradationLevel.L3_BOTH_DEGRADED, reason)

    def get_history(self) -> list[DegradationEvent]:
        """Return all state transition events for logging/telemetry."""
        return list(self._history)
