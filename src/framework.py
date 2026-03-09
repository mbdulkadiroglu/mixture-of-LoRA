"""
Main framework orchestrator for the Adaptive SLM system.
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from .adapters import AdapterManager
from .config import FrameworkConfig, load_config
from .datasets import DatasetLoader
from .evaluation import Evaluator, EvaluationResult
from .models import StudentModel, TeacherModel
from .router import QueryRouter, RoutingDecision, RoutingTarget
from .training import DataProcessor, LoRATrainer, TrainingExample
from .utils import (
    ExperienceReplayBuffer,
    set_seed,
    setup_logging,
    classify_query_domain,
)


@dataclass
class QueryResult:
    """Result of processing a query."""

    query: str
    response: str
    domain: str
    used_teacher: bool
    routing_decision: RoutingDecision
    confidence: float


class AdaptiveSLMFramework:
    """
    Main orchestrator for the Adaptive Small Language Model Framework.

    This class coordinates:
    - Query routing between student and teacher models
    - LoRA adapter management and switching
    - Online training from teacher responses
    - Performance evaluation and monitoring
    """

    def __init__(
        self,
        config: FrameworkConfig | None = None,
        config_path: str | Path | None = None,
    ):
        """
        Initialize the framework.

        Args:
            config: Framework configuration object.
            config_path: Path to configuration file (used if config is None).
        """
        if config is None:
            config = load_config(config_path)

        self.config = config

        # Setup logging
        setup_logging(config.log_level)
        set_seed(config.seed)

        # Initialize components
        self.teacher: TeacherModel | None = None
        self.student: StudentModel | None = None
        self.router: QueryRouter | None = None
        self.adapter_manager: AdapterManager | None = None
        self.trainer: LoRATrainer | None = None
        self.evaluator: Evaluator | None = None
        self.data_processor: DataProcessor | None = None
        self.replay_buffer: ExperienceReplayBuffer | None = None

        # State tracking
        self._initialized = False
        self._training_examples: list[TrainingExample] = []

        logger.info("AdaptiveSLMFramework created")

    def initialize(
        self,
        load_student: bool = True,
        load_teacher: bool = True,
    ) -> None:
        """
        Initialize all framework components.

        Args:
            load_student: Whether to load the student model.
            load_teacher: Whether to initialize teacher API client.
        """
        logger.info("Initializing framework components...")

        # Initialize teacher model
        if load_teacher:
            self.teacher = TeacherModel(self.config.teacher)
            logger.info("Teacher model initialized")

        # Initialize student model
        if load_student:
            self.student = StudentModel(
                self.config.student,
                self.config.lora,
                self.config.device_map,
            )
            self.student.load_model()
            logger.info("Student model loaded")

        # Initialize adapter manager
        self.adapter_manager = AdapterManager(self.config.adapter_manager)

        # Initialize router
        self.router = QueryRouter(
            self.config.router,
            self.student,
            self.teacher,
        )

        # Register existing adapters with router
        for domain in self.adapter_manager.list_domains():
            adapter_path = self.adapter_manager.get_adapter_path(domain)
            if adapter_path:
                self.router.register_adapter(domain, adapter_path)

        # Initialize replay buffer
        self.replay_buffer = ExperienceReplayBuffer(
            self.config.training.replay_buffer_size
        )

        # Initialize trainer
        if load_student:
            self.trainer = LoRATrainer(
                self.student,
                self.config.training,
                self.config.lora,
                Path(self.config.adapter_manager.base_path) / "training_runs",
                self.replay_buffer,
            )

            # Initialize data processor
            self.data_processor = DataProcessor(
                self.student.tokenizer,
                self.config.student.max_seq_length,
            )

        # Initialize evaluator with SQL execution support
        spider_db_dir = self.config.evaluation.spider_db_dir
        bird_db_dir = self.config.evaluation.bird_db_dir
        self.evaluator = Evaluator(spider_db_dir=spider_db_dir, bird_db_dir=bird_db_dir)

        self._initialized = True
        logger.info("Framework initialization complete")

    def process_query(
        self,
        query: str,
        domain: str | None = None,
        force_teacher: bool = False,
        collect_training_data: bool = True,
    ) -> QueryResult:
        """
        Process a user query through the adaptive system.

        Args:
            query: User's query.
            domain: Optional domain override.
            force_teacher: Force use of teacher model.
            collect_training_data: Whether to collect data for training.

        Returns:
            QueryResult with response and metadata.
        """
        if not self._initialized:
            raise RuntimeError("Framework not initialized. Call initialize() first.")

        # Classify domain if not provided
        if domain is None:
            domain = classify_query_domain(query)

        logger.debug(f"Processing query for domain: {domain}")

        # Get routing decision
        if force_teacher:
            routing = RoutingDecision(
                target=RoutingTarget.TEACHER,
                domain=domain,
                confidence=0.0,
                reason="Forced teacher mode",
            )
        else:
            routing = self.router.route(query, domain)

        # Generate response based on routing decision
        if routing.target == RoutingTarget.STUDENT and routing.adapter_path:
            response, used_teacher = self._generate_student_response(
                query, domain, routing
            )
        else:
            response, used_teacher = self._generate_teacher_response(query, domain)

        # Collect training data if teacher was used
        if collect_training_data and used_teacher:
            self._collect_training_example(query, response, domain)

        # Router stats are NOT updated here — we have no ground truth to verify
        # correctness. Stats should only be updated when SQL execution results
        # are available (wired in a later step).

        return QueryResult(
            query=query,
            response=response,
            domain=domain,
            used_teacher=used_teacher,
            routing_decision=routing,
            confidence=routing.confidence,
        )

    # Domain-specific system prompts (must match training prompts in DataProcessor)
    SYSTEM_PROMPTS = {
        "text_to_sql": """You are an expert SQL assistant. Given a database schema and a natural language question, output only the SQL query. Do not include any explanation, formatting, or markdown. Output only valid SQLite SQL.""",
        "text_to_sql_bird": """You are an expert SQL assistant. Given a database schema and a natural language question, output only the SQL query. Do not include any explanation, formatting, or markdown. Output only valid SQLite SQL.""",
        "math_reasoning": """You are a mathematics tutor. Solve problems step by step.
Show your work clearly and provide the final numerical answer.""",
        "code_generation": """You are an expert Python programmer.
Write clean, efficient, well-documented code with type hints.""",
        "general": """You are a helpful, accurate assistant.
Provide clear and well-structured responses.""",
    }

    def _generate_student_response(
        self,
        query: str,
        domain: str,
        routing: RoutingDecision,
    ) -> tuple[str, bool]:
        """
        Generate response using the student model.

        Returns tuple of (response, used_teacher).
        """
        try:
            # Load appropriate adapter
            if self.student.current_adapter != routing.adapter_path:
                self.student.load_adapter(routing.adapter_path)

            # Generate response with system prompt for domain context
            system_prompt = self.SYSTEM_PROMPTS.get(domain, self.SYSTEM_PROMPTS["general"])
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
            response = self.student.generate_chat(messages)

            # Optionally verify with teacher
            if self.config.router.max_retries > 0:
                confidence, feedback = self.teacher.evaluate_confidence(
                    query, response, domain
                )

                if confidence < self.config.router.confidence_threshold:
                    logger.info(f"Student response below threshold ({confidence:.2f}), using teacher")
                    teacher_response = self.teacher.generate_training_response(query, domain)
                    return teacher_response, True

            return response, False

        except Exception as e:
            logger.warning(f"Student generation failed: {e}, falling back to teacher")
            response = self.teacher.generate_training_response(query, domain)
            return response, True

    def _generate_teacher_response(
        self,
        query: str,
        domain: str,
    ) -> tuple[str, bool]:
        """Generate response using the teacher model."""
        response = self.teacher.generate_training_response(query, domain)
        return response, True

    def _collect_training_example(
        self,
        query: str,
        response: str,
        domain: str,
    ) -> None:
        """Collect a training example for future training."""
        example = TrainingExample(
            query=query,
            response=response,
            domain=domain,
            metadata={"source": "teacher"},
        )
        self._training_examples.append(example)

        logger.debug(f"Collected training example for domain: {domain}")

    def train_domain(
        self,
        domain: str,
        examples: list[TrainingExample] | None = None,
        num_epochs: int | None = None,
    ) -> dict:
        """
        Train a LoRA adapter for a specific domain.

        Args:
            domain: Domain to train.
            examples: Training examples (uses collected if None).
            num_epochs: Number of epochs.

        Returns:
            Training metrics.
        """
        if examples is None:
            # Use collected examples for this domain
            examples = [ex for ex in self._training_examples if ex.domain == domain]

        if not examples:
            raise ValueError(f"No training examples available for domain: {domain}")

        logger.info(f"Training domain '{domain}' with {len(examples)} examples")

        # Prepare dataset
        dataset = self.data_processor.prepare_dataset(examples, domain)

        # Setup LoRA if needed
        if self.student.peft_model is None:
            self.student.setup_lora()

        # Train
        metrics = self.trainer.train(dataset, domain, num_epochs=num_epochs)

        # Save adapter
        output_path = Path(self.config.adapter_manager.base_path) / domain / "latest"
        self.trainer.save_adapter(output_path)

        # Register with adapter manager
        adapter_info = self.adapter_manager.register_adapter(
            adapter_path=output_path,
            domain=domain,
            training_samples=len(examples),
            eval_score=0.0,  # Will be updated after evaluation
            lora_r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            base_model=self.config.student.name,
        )

        # Register with router
        self.router.register_adapter(domain, str(output_path))

        # Clear used training examples
        self._training_examples = [
            ex for ex in self._training_examples if ex.domain != domain
        ]

        logger.info(f"Training complete for domain '{domain}'")

        return {
            "training_metrics": metrics,
            "adapter_info": adapter_info,
        }

    def evaluate_domain(
        self,
        domain: str,
        test_dataset=None,
        max_samples: int | None = None,
        adapter_path: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate the student model on a domain.

        Args:
            domain: Domain to evaluate.
            test_dataset: Test dataset (loaded automatically if None).
            max_samples: Maximum samples to evaluate.
            adapter_path: Explicit adapter path to evaluate (uses best if None).

        Returns:
            EvaluationResult with metrics.
        """
        if test_dataset is None:
            loader = DatasetLoader()
            if domain == "text_to_sql":
                domain_data = loader.load_spider(max_samples=max_samples)
            elif domain == "text_to_sql_bird":
                domain_data = loader.load_bird(max_samples=max_samples)
            elif domain == "math_reasoning":
                domain_data = loader.load_gsm8k(max_samples=max_samples)
            elif domain == "code_generation":
                domain_data = loader.load_mbpp(max_samples=max_samples)
            else:
                raise ValueError(f"Unknown domain: {domain}")
            test_dataset = domain_data.test

        # Load adapter — use explicit path if provided, otherwise best
        if adapter_path is None:
            adapter_path = self.adapter_manager.get_adapter_path(domain)
        if adapter_path:
            self.student.load_adapter(adapter_path)
        else:
            raise ValueError(f"No adapter found for domain: {domain}")

        # Determine keys based on domain
        # For text_to_sql, use 'prompt' which includes schema context
        if domain == "text_to_sql":
            query_key, ref_key = "prompt", "query"
        elif domain == "text_to_sql_bird":
            query_key, ref_key = "prompt", "SQL"
        elif domain == "math_reasoning":
            query_key, ref_key = "question", "answer"
        elif domain == "code_generation":
            query_key, ref_key = "text", "code"
        else:
            query_key, ref_key = "question", "answer"

        # Run evaluation
        result = self.evaluator.evaluate_dataset(
            self.student,
            test_dataset,
            domain,
            query_key=query_key,
            reference_key=ref_key,
            max_samples=max_samples,
        )

        # Update adapter score — find adapter matching the path we evaluated
        if adapter_path:
            adapter = self.adapter_manager.get_adapter_by_path(domain, adapter_path)
            if adapter:
                self.adapter_manager.update_adapter_score(
                    domain, adapter.version, result.score
                )

        return result

    def get_statistics(self) -> dict:
        """
        Get framework statistics.

        Returns:
            Dictionary with various statistics.
        """
        return {
            "domains": self.adapter_manager.list_domains() if self.adapter_manager else [],
            "adapters": {
                d: self.adapter_manager.get_domain_stats(d)
                for d in (self.adapter_manager.list_domains() if self.adapter_manager else [])
            },
            "routing_stats": self.router.get_domain_stats() if self.router else {},
            "pending_training_examples": len(self._training_examples),
            "replay_buffer_size": len(self.replay_buffer) if self.replay_buffer else 0,
        }

    def save_state(self, path: str | Path) -> None:
        """Save framework state to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save replay buffer
        if self.replay_buffer:
            self.replay_buffer.save(path / "replay_buffer.json")

        # Save pending training examples
        import json

        examples_data = [
            {
                "query": ex.query,
                "response": ex.response,
                "domain": ex.domain,
            }
            for ex in self._training_examples
        ]
        with open(path / "pending_examples.json", "w") as f:
            json.dump(examples_data, f, indent=2)

        logger.info(f"Framework state saved to: {path}")

    def load_state(self, path: str | Path) -> None:
        """Load framework state from disk."""
        path = Path(path)

        # Load replay buffer
        if self.replay_buffer and (path / "replay_buffer.json").exists():
            self.replay_buffer.load(path / "replay_buffer.json")

        # Load pending training examples
        import json

        examples_path = path / "pending_examples.json"
        if examples_path.exists():
            with open(examples_path) as f:
                examples_data = json.load(f)
            self._training_examples = [
                TrainingExample(**ex) for ex in examples_data
            ]

        logger.info(f"Framework state loaded from: {path}")
