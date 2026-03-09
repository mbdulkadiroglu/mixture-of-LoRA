"""
Cascade router — threshold on mean token log-prob.

"Generate-then-route" design: the student always generates first,
then the router decides whether to keep the student response or
escalate to the teacher. This guarantees confidence metrics for every query.
"""

from dataclasses import dataclass

from .config import CascadeConfig
from .student import GenerationResult


@dataclass
class RoutingDecision:
    target: str  # "student" or "teacher"
    confidence_value: float
    threshold: float


class CascadeRouter:
    def __init__(self, config: CascadeConfig):
        self.config = config
        self.threshold = config.router_threshold
        self.metric = config.router_metric

    def decide(self, gen_result: GenerationResult) -> RoutingDecision:
        """
        Decide whether to use the student response or escalate to teacher.

        Returns RoutingDecision with target="student" if confident,
        otherwise target="teacher".
        """
        if self.metric == "mean_logprob":
            value = gen_result.mean_log_prob
        elif self.metric == "min_logprob":
            value = gen_result.min_log_prob
        elif self.metric == "mean_entropy":
            # For entropy, higher = less confident, so we negate for threshold comparison
            # threshold should be negative (e.g., -5.0 means entropy < 5.0)
            value = -gen_result.mean_entropy
        else:
            value = gen_result.mean_log_prob

        target = "student" if value >= self.threshold else "teacher"

        return RoutingDecision(
            target=target,
            confidence_value=value,
            threshold=self.threshold,
        )

    def _get_confidence(self, gen_result: GenerationResult) -> float:
        """Extract confidence metric from a generation result."""
        if self.metric == "mean_logprob":
            return gen_result.mean_log_prob
        elif self.metric == "min_logprob":
            return gen_result.min_log_prob
        elif self.metric == "mean_entropy":
            return -gen_result.mean_entropy
        return gen_result.mean_log_prob

    def decide_batch(
        self, gen_results: list[GenerationResult], round_idx: int
    ) -> list[RoutingDecision]:
        """
        Batch routing: sort by confidence, route bottom target_rate% to teacher.

        Uses the cascade rate schedule from config. Independent of absolute
        logprob values — immune to distribution shift from training.
        """
        target_rate = self.config.get_cascade_rate(round_idx)
        n = len(gen_results)
        n_teacher = int(round(n * target_rate))

        # Get confidence for each result
        confidences = [self._get_confidence(gr) for gr in gen_results]

        # Find the cutoff: sort indices by confidence ascending
        sorted_indices = sorted(range(n), key=lambda i: confidences[i])
        teacher_set = set(sorted_indices[:n_teacher])

        # Compute the effective threshold (confidence of the boundary query)
        effective_threshold = confidences[sorted_indices[n_teacher]] if n_teacher < n else float("inf")

        decisions = []
        for i in range(n):
            target = "teacher" if i in teacher_set else "student"
            decisions.append(RoutingDecision(
                target=target,
                confidence_value=confidences[i],
                threshold=effective_threshold,
            ))
        return decisions
