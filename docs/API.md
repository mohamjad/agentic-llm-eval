# API Documentation

This document summarizes the public package surface and the main extension
points.

## Evaluators

`AgentEvaluator` is the main entry point. It accepts an optional configuration
dictionary with fields such as `use_tracer`, `success_threshold`, and
`metric_weights`.

- `evaluate(agent, task)` returns a single `EvaluationResult`
- `evaluate_batch(agent, tasks)` returns a list of `EvaluationResult` objects

`EvaluationResult` includes:

- `task_id`
- `success`
- `score`
- `metrics`
- `error`
- `execution_trace`

## Agents and Traces

`BaseAgent` defines the agent interface. Implement `execute(task, trace=None)`
and return a result object that your metrics can consume.

`AgentExecutionTrace` captures execution steps and tool calls.

`AgentTracer` wraps an agent to collect traces automatically when you do not
want to instrument trace creation manually.

## Benchmarks

`TaskBenchmark.load(name)` loads a benchmark bundle and exposes a `tasks` list.

Each `Task` includes:

- `id`
- `input`
- `expected_output`
- `description`
- `metadata`

## Metrics

The built-in metrics are small and inspectable.

- `AccuracyMetric`
- `EfficiencyMetric`
- `SafetyMetric`
- `CoherenceMetric`
- `AdaptabilityMetric`

They are intended as practical scoring helpers, not as validated research
measures. See [`../METRICS.md`](../METRICS.md) for the implementation-level
behavior of each metric.

## Parameter Tuning

`RLTrainer` runs repeated evaluate-adjust cycles against an agent that supports
parameter updates.

`AgentParameters` defines the tunable fields:

- `context_length`
- `temperature`
- `max_steps`
- `tool_usage_threshold`
- `reasoning_depth`

The default trainer and policy classes are lightweight parameter-tuning
utilities. Optional neural and Bayesian modules are available when their
dependencies are installed, but they are not required for the base workflow.

## Utilities

`Config` loads defaults, file-based overrides, and environment variables.

Validation helpers enforce parameter ranges, metric value shape, and task-input
sanity checks.

`StatisticalAnalyzer` provides descriptive comparisons and interval helpers for
result sets. It should be treated as support tooling rather than as a complete
experimental framework.
