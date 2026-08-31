"""
CXR Sentinel — the one place reinforcement learning has a real role here.

Not deep RL, not an agent playing a game — there's no environment or
sequential decision process anywhere else in this pipeline to justify that.
What DOES fit: your original plan's item #10 (human-AI disagreement
learning). Every time a reviewer accepts or rejects a prediction, that's a
reward signal for one specific decision — where to set the "confidence too
low, defer to a human" abstention threshold. That's a genuine contextual
bandit problem: pick a threshold (arm), observe a reward (did abstaining
or predicting match what the reviewer wanted), update.

This is intentionally small (epsilon-greedy over a discrete set of candidate
thresholds, one bandit per finding). It's real and it learns from real
feedback, it's just honestly scoped — a starting point for Phase 4's
disagreement-learning loop, not a production RLHF system.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class ThresholdBandit:
    """One bandit per finding. Arms = candidate abstention thresholds."""

    candidate_thresholds: list[float] = field(default_factory=lambda: [0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    epsilon: float = 0.15
    seed: int | None = None

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self.counts = {t: 0 for t in self.candidate_thresholds}
        self.value_estimates = {t: 0.0 for t in self.candidate_thresholds}

    def select_threshold(self) -> float:
        """Epsilon-greedy: explore a random arm with prob epsilon, else exploit the best-known arm."""
        if self._rng.random() < self.epsilon:
            return self._rng.choice(self.candidate_thresholds)
        return max(self.value_estimates, key=self.value_estimates.get)

    def update(self, threshold: float, reward: float) -> None:
        """Incremental sample-average update — standard bandit update rule."""
        self.counts[threshold] += 1
        n = self.counts[threshold]
        old_estimate = self.value_estimates[threshold]
        self.value_estimates[threshold] = old_estimate + (reward - old_estimate) / n

    def best_threshold(self) -> float:
        return max(self.value_estimates, key=self.value_estimates.get)


def compute_reward(model_prob: float, threshold: float, reviewer_accepted: bool) -> float:
    """
    Reward design: abstaining (model_prob < threshold) is "correct" if the
    reviewer would have rejected the prediction anyway (low reviewer trust in
    that call); predicting (model_prob >= threshold) is "correct" if the
    reviewer accepted it. Wrong call in either direction gets penalized.

    reviewer_accepted: ground-truth signal from your accept/reject/edit UI
    (see PROJECT_PLAN.md Phase 4 — the reviewer feedback loop this depends on).
    """
    would_predict = model_prob >= threshold
    if would_predict and reviewer_accepted:
        return 1.0
    if not would_predict and not reviewer_accepted:
        return 1.0  # correctly deferred on a case the reviewer would've rejected
    return -1.0  # either predicted-and-wrong, or abstained-when-reviewer-would-have-accepted


def simulate_feedback_round(bandit: ThresholdBandit, model_prob: float, true_reviewer_accepts: bool) -> float:
    """One bandit training step: pick a threshold, compute reward against the
    (simulated or real) reviewer decision, update the bandit. Returns the reward."""
    threshold = bandit.select_threshold()
    reward = compute_reward(model_prob, threshold, true_reviewer_accepts)
    bandit.update(threshold, reward)
    return reward
