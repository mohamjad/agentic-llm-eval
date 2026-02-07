# API Documentation

Complete API reference for the agentic-llm-eval framework.

## Table of Contents

- [Evaluators](#evaluators)
- [Agents](#agents)
- [Metrics](#metrics)
- [Benchmarks](#benchmarks)
- [RL Training](#rl-training)
- [Configuration](#configuration)
- [Utilities](#utilities)

## Evaluators

### `AgentEvaluator`

Main evaluator class for assessing agent performance.

```python
from src.evaluators import AgentEvaluator

evaluator = AgentEvaluator(config={
    "use_tracer": True,
    "success_threshold": 0.7,
    "metric_weights": {
        "accuracy": 0.4,
        "efficiency": 0.3,
        "tool_usage": 0.2,
        "safety": 0.1
    }
})
```

#### Methods

##### `evaluate(agent, task) -> EvaluationResult`

Evaluate an agent on a single task.

**Parameters:**
- `agent` (BaseAgent): Agent instance to evaluate
- `task` (Task): Task to evaluate on

**Returns:**
- `EvaluationResult`: Evaluation result with score, metrics, and trace

**Example:**
```python
result = evaluator.evaluate(agent, task)
print(f"Score: {result.score}, Success: {result.success}")
```

##### `evaluate_batch(agent, tasks) -> List[EvaluationResult]`

Evaluate an agent on multiple tasks.

**Parameters:**
- `agent` (BaseAgent): Agent instance to evaluate
- `tasks` (List[Task]): List of tasks to evaluate on

**Returns:**
- `List[EvaluationResult]`: List of evaluation results

**Example:**
```python
results = evaluator.evaluate_batch(agent, tasks)
success_rate = sum(1 for r in results if r.success) / len(results)
```

### `EvaluationResult`

Result of an evaluation.

**Attributes:**
- `task_id` (str): Identifier of the evaluated task
- `success` (bool): Whether the task was completed successfully
- `score` (float): Overall score (0.0 to 1.0)
- `metrics` (Dict[str, Any]): Detailed metrics dictionary
- `error` (Optional[str]): Error message if evaluation failed
- `execution_trace` (Optional[List[Dict]]): Execution trace

## Agents

### `BaseAgent`

Abstract base class for agents.

```python
from src.agents.base import BaseAgent, AgentExecutionTrace

class MyAgent(BaseAgent):
    def execute(self, task, trace: AgentExecutionTrace = None):
        # Your agent logic here
        if trace:
            trace.add_step(
                action_type="reasoning",
                action_name="think",
                input_data={"task": task.input},
                output_data={"thought": "..."},
                duration=0.1
            )
        return {"result": "done"}
```

#### Methods

##### `execute(task, trace=None) -> Any`

Execute a task.

**Parameters:**
- `task`: Task to execute
- `trace` (Optional[AgentExecutionTrace]): Execution trace for recording steps

**Returns:**
- Task result (any type)

### `AgentExecutionTrace`

Tracks agent execution steps.

**Methods:**
- `add_step(action_type, action_name, input_data, output_data, duration=0.0, metadata=None) -> TraceStep`
- `get_tools_used() -> List[str]`: Get list of tools used
- `get_total_duration() -> float`: Get total execution time
- `to_dict_list() -> List[Dict]`: Convert to list of dictionaries

## Metrics

### `AccuracyMetric`

Measures correctness of agent outputs.

```python
from src.metrics.accuracy import AccuracyMetric

metric = AccuracyMetric()
metrics = metric.collect(agent, task, result, trace)
accuracy_score = metrics["accuracy"]  # 0.0 to 1.0
```

### `EfficiencyMetric`

Measures resource efficiency (steps, time).

```python
from src.metrics.efficiency import EfficiencyMetric

metric = EfficiencyMetric(max_reasonable_steps=20)
metrics = metric.collect(agent, task, result, trace)
efficiency_score = metrics["efficiency"]  # 0.0 to 1.0
```

### `SafetyMetric`

Detects unsafe or harmful content.

```python
from src.metrics.safety import SafetyMetric

metric = SafetyMetric(
    severity_threshold=0.5,
    check_patterns=True
)
metrics = metric.collect(agent, task, result, trace)
safety_score = metrics["safety_score"]  # 0.0 to 1.0
```

**Features:**
- Multi-level keyword detection (severe, moderate, mild)
- Security pattern detection
- Context-aware analysis

### `CoherenceMetric`

Measures logical consistency and flow.

```python
from src.metrics.coherence import CoherenceMetric

metric = CoherenceMetric()
metrics = metric.collect(agent, task, result, trace)
coherence_score = metrics["coherence"]  # 0.0 to 1.0
```

### `AdaptabilityMetric`

Measures adaptation to different contexts.

```python
from src.metrics.adaptability import AdaptabilityMetric

metric = AdaptabilityMetric()
metrics = metric.collect(agent, task, result, trace)
adaptability_score = metrics["adaptability"]  # 0.0 to 1.0
```

## Benchmarks

### `TaskBenchmark`

Standard task benchmark suite.

```python
from src.benchmarks import TaskBenchmark

benchmark = TaskBenchmark.load("standard_tasks")
tasks = benchmark.tasks
```

#### Methods

##### `load(name, config=None) -> TaskBenchmark`

Load a benchmark by name.

**Parameters:**
- `name` (str): Benchmark name
- `config` (Optional[Dict]): Configuration dictionary

**Returns:**
- `TaskBenchmark`: Benchmark instance

### `Task`

Represents a single task.

**Attributes:**
- `id` (str): Task identifier
- `description` (str): Task description
- `input` (Dict): Task input data
- `expected_output` (Optional[Any]): Expected output
- `metadata` (Optional[Dict]): Additional metadata

**Methods:**
- `check_success(result) -> bool`: Check if result matches expected output

## RL Training

### `RLTrainer`

Reinforcement learning trainer for fine-tuning agent parameters.

```python
from src.rl import RLTrainer, AgentParameters
from src.evaluators import AgentEvaluator
from src.benchmarks import TaskBenchmark

trainer = RLTrainer(
    agent=agent,
    evaluator=evaluator,
    benchmark=benchmark,
    initial_params=AgentParameters(),
    learning_rate=0.01
)

result = trainer.train(episodes=10, tasks_per_episode=5)
best_params = trainer.get_best_parameters()
```

#### Methods

##### `train(episodes, tasks_per_episode=None, update_policy=True) -> Dict`

Train agent over multiple episodes.

**Parameters:**
- `episodes` (int): Number of training episodes
- `tasks_per_episode` (Optional[int]): Tasks per episode (None = all)
- `update_policy` (bool): Whether to update policy weights

**Returns:**
- `Dict`: Training results with final params, metrics, and history

##### `get_best_parameters() -> AgentParameters`

Get parameters from best performing episode.

**Returns:**
- `AgentParameters`: Best parameters found

### `AgentParameters`

Tunable agent parameters.

**Attributes:**
- `context_length` (int): Context window size (100-5000)
- `temperature` (float): Sampling temperature (0.1-2.0)
- `max_steps` (int): Maximum execution steps (1-50)
- `tool_usage_threshold` (float): Tool usage threshold (0.0-1.0)
- `reasoning_depth` (int): Reasoning depth (1-10)

**Methods:**
- `to_dict() -> Dict`: Convert to dictionary
- `from_dict(params) -> AgentParameters`: Create from dictionary

### `PolicyNetwork`

Enhanced policy network with adaptive learning.

**Features:**
- Momentum-based updates
- Experience replay buffer
- Adaptive learning rate decay
- Multi-objective optimization

## Configuration

### `Config`

Centralized configuration manager.

```python
from src.utils.config import Config

config = Config()
config.load_from_file("config.json")
config.load_from_env(prefix="AGENTIC_")

evaluator_config = config.get("evaluator")
```

#### Methods

##### `load_from_file(filepath, merge=True) -> Config`

Load configuration from JSON or YAML file.

##### `load_from_env(prefix="AGENTIC_", separator="__", merge=True) -> Config`

Load configuration from environment variables.

##### `get(key, default=None, required=False) -> Any`

Get configuration value (supports dot notation).

##### `set(key, value) -> Config`

Set configuration value.

##### `validate(schema=None) -> bool`

Validate configuration.

## Utilities

### Validation

```python
from src.utils.validation import (
    validate_agent_parameters,
    validate_metrics,
    validate_task_input,
    ValidationError
)

try:
    validate_agent_parameters(params)
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Logging

```python
from src.utils.logger import setup_logger

logger = setup_logger("my_module", level="INFO")
logger.info("Evaluation started")
```

## Examples

See the `examples/` directory for complete usage examples:

- `simple_evaluation.py`: Basic evaluation workflow
- `rl_training_example.py`: RL-based fine-tuning
- `parameter_impact_example.py`: Testing parameter effects
