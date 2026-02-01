"""
Query router for deciding between student and teacher models.
"""

from dataclasses import dataclass
from enum import Enum

from loguru import logger

from ..config import RouterConfig
from ..models.student import StudentModel
from ..models.teacher import TeacherModel
from ..utils import classify_query_domain


class RoutingTarget(Enum):
    """Target for routing decision."""

    STUDENT = "student"
    TEACHER = "teacher"


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    target: RoutingTarget
    domain: str
    confidence: float
    reason: str
    adapter_path: str | None = None


class QueryRouter:
    """
    Routes queries between student and teacher models.

    The router uses multiple strategies to decide whether the student
    model (with appropriate LoRA adapter) can handle a query, or if
    it should be forwarded to the teacher model.
    """

    def __init__(
        self,
        config: RouterConfig,
        student: StudentModel | None = None,
        teacher: TeacherModel | None = None,
    ):
        """
        Initialize the router.

        Args:
            config: Router configuration.
            student: Student model instance (optional, for perplexity routing).
            teacher: Teacher model instance (optional, for self-eval routing).
        """
        self.config = config
        self.student = student
        self.teacher = teacher

        # Domain performance tracking
        self._domain_stats: dict[str, dict] = {}

        # Adapter availability tracking
        self._available_adapters: dict[str, str] = {}

        logger.info(
            f"Router initialized with method: {config.routing_method}, "
            f"threshold: {config.confidence_threshold}"
        )

    def register_adapter(self, domain: str, adapter_path: str) -> None:
        """
        Register an available adapter for a domain.

        Args:
            domain: Domain identifier.
            adapter_path: Path to the adapter.
        """
        self._available_adapters[domain] = adapter_path
        logger.info(f"Registered adapter for domain '{domain}': {adapter_path}")

    def unregister_adapter(self, domain: str) -> None:
        """
        Unregister an adapter for a domain.

        Args:
            domain: Domain identifier.
        """
        if domain in self._available_adapters:
            del self._available_adapters[domain]
            logger.info(f"Unregistered adapter for domain: {domain}")

    def route(self, query: str, domain: str | None = None) -> RoutingDecision:
        """
        Make a routing decision for a query.

        Args:
            query: The user's query.
            domain: Optional domain override. If not provided, auto-detected.

        Returns:
            RoutingDecision with target, confidence, and reason.
        """
        # Classify domain if not provided
        if domain is None:
            domain = classify_query_domain(query)

        logger.debug(f"Routing query for domain: {domain}")

        # Check if adapter exists for this domain
        adapter_path = self._available_adapters.get(domain)

        if adapter_path is None:
            return RoutingDecision(
                target=RoutingTarget.TEACHER,
                domain=domain,
                confidence=0.0,
                reason=f"No adapter available for domain: {domain}",
                adapter_path=None,
            )

        # Route based on configured method
        if self.config.routing_method == "perplexity":
            decision = self._route_by_perplexity(query, domain, adapter_path)
        elif self.config.routing_method == "self_eval":
            decision = self._route_by_self_eval(query, domain, adapter_path)
        elif self.config.routing_method == "classifier":
            decision = self._route_by_classifier(query, domain, adapter_path)
        else:
            # Default: check domain stats
            decision = self._route_by_stats(query, domain, adapter_path)

        logger.info(
            f"Routing decision: {decision.target.value} "
            f"(confidence: {decision.confidence:.2f}, domain: {domain})"
        )

        return decision

    def _route_by_perplexity(
        self,
        query: str,
        domain: str,
        adapter_path: str,
    ) -> RoutingDecision:
        """
        Route based on student model's perplexity on the query.

        Lower perplexity suggests higher confidence.
        """
        if self.student is None or not self.student.is_loaded:
            return RoutingDecision(
                target=RoutingTarget.TEACHER,
                domain=domain,
                confidence=0.0,
                reason="Student model not available for perplexity check",
                adapter_path=adapter_path,
            )

        try:
            # Compute perplexity with the adapter loaded
            self.student.load_adapter(adapter_path)
            perplexity = self.student.compute_perplexity(query)

            # Convert perplexity to confidence (lower perplexity = higher confidence)
            # Using exponential decay: confidence = exp(-perplexity/scale)
            scale = 50.0  # Tunable parameter
            confidence = min(1.0, max(0.0, 2.0 / (1 + perplexity / scale)))

            if confidence >= self.config.confidence_threshold:
                return RoutingDecision(
                    target=RoutingTarget.STUDENT,
                    domain=domain,
                    confidence=confidence,
                    reason=f"Perplexity {perplexity:.2f} indicates high confidence",
                    adapter_path=adapter_path,
                )
            else:
                return RoutingDecision(
                    target=RoutingTarget.TEACHER,
                    domain=domain,
                    confidence=confidence,
                    reason=f"Perplexity {perplexity:.2f} below threshold",
                    adapter_path=adapter_path,
                )

        except Exception as e:
            logger.warning(f"Perplexity routing failed: {e}")
            return RoutingDecision(
                target=RoutingTarget.TEACHER,
                domain=domain,
                confidence=0.0,
                reason=f"Perplexity computation failed: {e}",
                adapter_path=adapter_path,
            )

    def _route_by_self_eval(
        self,
        query: str,
        domain: str,
        adapter_path: str,
    ) -> RoutingDecision:
        """
        Route by asking the teacher model to evaluate query difficulty.
        """
        if self.teacher is None:
            return RoutingDecision(
                target=RoutingTarget.TEACHER,
                domain=domain,
                confidence=0.0,
                reason="Teacher model not available for self-eval",
                adapter_path=adapter_path,
            )

        try:
            # Get domain stats for context
            stats = self._domain_stats.get(domain, {})
            success_rate = stats.get("success_rate", 0.5)

            eval_prompt = f"""Evaluate if a fine-tuned small language model (14B parameters) can correctly handle this query.

Domain: {domain}
Historical success rate for this domain: {success_rate:.1%}

Query: {query}

Consider:
1. Query complexity
2. Required reasoning depth
3. Domain-specific knowledge needed
4. Whether similar queries typically succeed

Respond with a single number between 0.0 and 1.0 representing the probability the small model can handle this correctly."""

            messages = [{"role": "user", "content": eval_prompt}]
            response = self.teacher.generate(messages, max_tokens=50, temperature=0.1)

            # Parse confidence score
            import re

            numbers = re.findall(r"0?\.\d+|1\.0|1|0", response)
            confidence = float(numbers[0]) if numbers else 0.5

            if confidence >= self.config.confidence_threshold:
                return RoutingDecision(
                    target=RoutingTarget.STUDENT,
                    domain=domain,
                    confidence=confidence,
                    reason=f"Teacher evaluation: {confidence:.2f}",
                    adapter_path=adapter_path,
                )
            else:
                return RoutingDecision(
                    target=RoutingTarget.TEACHER,
                    domain=domain,
                    confidence=confidence,
                    reason=f"Teacher evaluation: {confidence:.2f} (below threshold)",
                    adapter_path=adapter_path,
                )

        except Exception as e:
            logger.warning(f"Self-eval routing failed: {e}")
            return RoutingDecision(
                target=RoutingTarget.TEACHER,
                domain=domain,
                confidence=0.0,
                reason=f"Self-eval failed: {e}",
                adapter_path=adapter_path,
            )

    def _route_by_classifier(
        self,
        query: str,
        domain: str,
        adapter_path: str,
    ) -> RoutingDecision:
        """
        Route using a trained classifier model.

        This is a placeholder for a more sophisticated approach
        using a dedicated classifier trained on routing decisions.
        """
        # For now, use stats-based routing as fallback
        return self._route_by_stats(query, domain, adapter_path)

    def _route_by_stats(
        self,
        query: str,
        domain: str,
        adapter_path: str,
    ) -> RoutingDecision:
        """
        Route based on historical domain statistics.
        """
        stats = self._domain_stats.get(domain, {})

        # Use success rate as confidence
        success_rate = stats.get("success_rate", 0.0)
        total_queries = stats.get("total", 0)

        # If we have enough data and good success rate, use student
        if total_queries >= 10 and success_rate >= self.config.confidence_threshold:
            return RoutingDecision(
                target=RoutingTarget.STUDENT,
                domain=domain,
                confidence=success_rate,
                reason=f"Historical success rate: {success_rate:.1%} ({total_queries} queries)",
                adapter_path=adapter_path,
            )

        # Otherwise, use teacher (especially for cold start)
        return RoutingDecision(
            target=RoutingTarget.TEACHER,
            domain=domain,
            confidence=success_rate,
            reason=f"Insufficient confidence: {success_rate:.1%} ({total_queries} queries)",
            adapter_path=adapter_path,
        )

    def update_stats(
        self,
        domain: str,
        success: bool,
        student_used: bool,
    ) -> None:
        """
        Update domain statistics after a query is processed.

        Args:
            domain: The domain of the query.
            success: Whether the response was successful.
            student_used: Whether the student model was used.
        """
        if domain not in self._domain_stats:
            self._domain_stats[domain] = {
                "total": 0,
                "successes": 0,
                "student_total": 0,
                "student_successes": 0,
                "success_rate": 0.0,
            }

        stats = self._domain_stats[domain]
        stats["total"] += 1

        if success:
            stats["successes"] += 1

        if student_used:
            stats["student_total"] += 1
            if success:
                stats["student_successes"] += 1

        # Update success rate (for student model specifically)
        if stats["student_total"] > 0:
            stats["success_rate"] = stats["student_successes"] / stats["student_total"]

    def get_domain_stats(self, domain: str | None = None) -> dict:
        """
        Get statistics for a domain or all domains.

        Args:
            domain: Optional domain to get stats for.

        Returns:
            Statistics dictionary.
        """
        if domain:
            return self._domain_stats.get(domain, {})
        return self._domain_stats.copy()

    def should_retrain(self, domain: str) -> bool:
        """
        Determine if a domain's adapter should be retrained.

        Args:
            domain: Domain to check.

        Returns:
            True if retraining is recommended.
        """
        stats = self._domain_stats.get(domain, {})

        # Retrain if:
        # 1. Success rate dropped below threshold
        # 2. We have enough recent failures to be confident
        success_rate = stats.get("success_rate", 1.0)
        student_total = stats.get("student_total", 0)

        if student_total >= 20 and success_rate < self.config.confidence_threshold - 0.1:
            return True

        return False
