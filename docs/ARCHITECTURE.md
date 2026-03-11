# Architecture

The package is organized around a simple pipeline:

1. Load tasks from a benchmark
2. Run an agent against those tasks
3. Capture an execution trace
4. Score the result with independent metrics
5. Aggregate the metrics into a final evaluation result

## Core Modules

`src/evaluators`
: Orchestrates evaluation and score aggregation.

`src/agents`
: Defines the base agent interface and trace objects.

`src/benchmarks`
: Loads task sets and exposes benchmark/task abstractions.

`src/metrics`
: Implements individual scoring functions.

`src/rl`
: Contains parameter-tuning helpers and optional experimental modules.

`src/utils`
: Configuration, validation, logging, and statistics support.

## Design Goals

- Keep the baseline path readable and easy to test
- Separate task execution from scoring logic
- Allow optional dependencies without breaking the base package
- Favor explicit data structures over hidden state

## Extension Points

To add a metric, implement a metric class and register it with the evaluator.

To add a benchmark, extend the benchmark abstraction and provide task-loading
logic.

To add a custom agent, subclass `BaseAgent` and implement `execute`.

## Notes on Optional Modules

The repo includes optional semantic, Bayesian, and neural components. They sit
off the main path on purpose:

- the base evaluator works without them
- imports degrade gracefully when optional packages are missing
- tests cover the lightweight path first and treat optional modules as additive

That structure is deliberate. The project's main value is the evaluation
harness and traceable scoring flow, not the claim that every optional module is
a production-ready research system.
