# Architecture Documentation

The agentic-llm-eval framework uses a modular, extensible architecture that separates concerns and enables easy customization. Everything is designed to be explicit and traceable, with clear interfaces between components.

Evaluators live in src/evaluators and orchestrate evaluation workflows while aggregating metrics. BaseEvaluator defines the abstract evaluation interface, while AgentEvaluator implements the comprehensive evaluation logic. These components execute agents on tasks, collect metrics from multiple sources, calculate overall scores, and determine success or failure.

Agents are defined in src/agents and provide the interface for agent execution along with execution tracing. BaseAgent is the abstract interface that agents must implement. AgentExecutionTrace tracks execution steps, and AgentTracer provides automatic tracing wrappers. These components define the agent execution contract, capture execution traces, and record tool usage.

Metrics in src/metrics calculate various performance metrics. AccuracyMetric measures correctness, EfficiencyMetric measures resource usage, SafetyMetric detects unsafe content, CoherenceMetric measures logical consistency, and AdaptabilityMetric measures context adaptation. Each metric is independent and implements a collect method, following a consistent design pattern.

Benchmarks in src/benchmarks provide test tasks and scenarios. BaseBenchmark defines the abstract benchmark interface, TaskBenchmark implements the standard task benchmark, and Task represents individual tasks. These components load and manage task sets, provide task metadata, and validate task results.

RL training components in src/rl fine-tune agent parameters using reinforcement learning. PolicyNetwork handles parameter adjustment, AgentPolicy wraps the policy to apply it to parameters, RLTrainer orchestrates training, and AgentParameters defines the tunable parameter set. Features include adaptive learning rates, experience replay, momentum-based updates, and multi-objective optimization.

Utilities in src/utils provide shared helpers. The config module handles centralized configuration management, validation provides input validation utilities, and logger sets up logging infrastructure.

Data flows from tasks through agents to execution traces, then through metrics to evaluation results. The RL trainer can adjust parameters based on results, creating an improved agent that feeds back into the system.

Design principles guide the architecture. Everything is explicit and traceable with no black boxes. The system is modular and extensible so each component can be extended or replaced independently. Behavior is controlled through configuration rather than code changes. Type hints throughout provide better IDE support and error detection. Clear interfaces make testing straightforward.

To add a new metric, create a class in src/metrics that implements the collect method taking agent, task, result, and trace parameters, then return a dictionary with metric values and register it in AgentEvaluator._calculate_all_metrics. To add a new benchmark, extend BaseBenchmark, implement load_tasks, create a task configuration file, and use TaskBenchmark.load to load it. To create a custom agent, extend BaseAgent, implement execute with task and trace parameters, optionally record steps in the trace, and return the task result.

Configuration flows through multiple layers with clear priority. Defaults are hardcoded in Config._load_defaults, files are loaded from configs/config.json, environment variables are loaded with prefix support, and runtime overrides can be set programmatically. Priority goes runtime over environment over file over defaults.

Error handling is comprehensive. Validation errors are caught at input boundaries, execution errors are captured in EvaluationResult.error, and configuration errors are raised during initialization.

Performance considerations include lightweight traces stored as dictionaries, lazy metric calculation, parallelizable batch evaluation, and distributable RL training.

Security measures include input validation to prevent injection, safety metrics that detect harmful content, configuration validation to prevent misconfiguration, and no arbitrary code execution.
