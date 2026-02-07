# Architecture Documentation

## Overview

The agentic-llm-eval framework is designed with a modular, extensible architecture that separates concerns and enables easy customization.

## Core Components

### 1. Evaluators (`src/evaluators/`)

**Purpose**: Orchestrate evaluation workflows and aggregate metrics.

**Key Classes:**
- `BaseEvaluator`: Abstract base class defining evaluation interface
- `AgentEvaluator`: Main evaluator implementing comprehensive evaluation logic

**Responsibilities:**
- Execute agents on tasks
- Collect metrics from multiple sources
- Calculate overall scores
- Determine success/failure

### 2. Agents (`src/agents/`)

**Purpose**: Define agent interface and execution tracing.

**Key Classes:**
- `BaseAgent`: Abstract interface for agents
- `AgentExecutionTrace`: Tracks execution steps
- `AgentTracer`: Automatic tracing wrapper

**Responsibilities:**
- Define agent execution contract
- Capture execution traces
- Record tool usage

### 3. Metrics (`src/metrics/`)

**Purpose**: Calculate various performance metrics.

**Key Classes:**
- `AccuracyMetric`: Measures correctness
- `EfficiencyMetric`: Measures resource usage
- `SafetyMetric`: Detects unsafe content
- `CoherenceMetric`: Measures logical consistency
- `AdaptabilityMetric`: Measures context adaptation

**Design Pattern**: Each metric is independent and implements a `collect()` method.

### 4. Benchmarks (`src/benchmarks/`)

**Purpose**: Provide test tasks and scenarios.

**Key Classes:**
- `BaseBenchmark`: Abstract benchmark interface
- `TaskBenchmark`: Standard task benchmark
- `Task`: Individual task representation

**Responsibilities:**
- Load and manage task sets
- Provide task metadata
- Validate task results

### 5. RL Training (`src/rl/`)

**Purpose**: Fine-tune agent parameters using reinforcement learning.

**Key Classes:**
- `PolicyNetwork`: Policy network for parameter adjustment
- `AgentPolicy`: Wrapper applying policy to parameters
- `RLTrainer`: Training orchestrator
- `AgentParameters`: Tunable parameter set

**Features:**
- Adaptive learning rates
- Experience replay
- Momentum-based updates
- Multi-objective optimization

### 6. Utilities (`src/utils/`)

**Purpose**: Shared utilities and helpers.

**Modules:**
- `config.py`: Centralized configuration management
- `validation.py`: Input validation utilities
- `logger.py`: Logging setup

## Data Flow

```
Task → Agent → Execution Trace → Metrics → Evaluation Result
                ↓
            RL Trainer → Parameter Adjustment → Improved Agent
```

## Design Principles

### 1. Explicit and Traceable

All calculations are explicit and documented. No black boxes.

### 2. Modular and Extensible

Each component can be extended or replaced independently.

### 3. Configurable

Behavior controlled through configuration, not code changes.

### 4. Type-Safe

Uses type hints throughout for better IDE support and error detection.

### 5. Testable

Clear interfaces make testing straightforward.

## Extension Points

### Adding a New Metric

1. Create a new class in `src/metrics/`
2. Implement `collect(agent, task, result, trace)` method
3. Return dictionary with metric values
4. Register in `AgentEvaluator._calculate_all_metrics()`

### Adding a New Benchmark

1. Extend `BaseBenchmark`
2. Implement `load_tasks()` method
3. Create task configuration file
4. Use `TaskBenchmark.load()` to load

### Creating a Custom Agent

1. Extend `BaseAgent`
2. Implement `execute(task, trace)` method
3. Optionally record steps in trace
4. Return task result

## Configuration System

Configuration flows through multiple layers:

1. **Defaults**: Hardcoded in `Config._load_defaults()`
2. **File**: Loaded from `configs/config.json`
3. **Environment**: Loaded from environment variables
4. **Runtime**: Can be overridden programmatically

Priority: Runtime > Environment > File > Defaults

## Error Handling

- **Validation Errors**: Caught at input boundaries
- **Execution Errors**: Captured in `EvaluationResult.error`
- **Configuration Errors**: Raised during initialization

## Performance Considerations

- Traces are lightweight (dictionaries)
- Metrics calculated lazily
- Batch evaluation parallelizable
- RL training can be distributed

## Security

- Input validation prevents injection
- Safety metric detects harmful content
- Configuration validation prevents misconfiguration
- No arbitrary code execution
