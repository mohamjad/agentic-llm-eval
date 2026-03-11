# agentic-llm-eval

`agentic-llm-eval` is a lightweight Python toolkit for running repeatable agent
evaluations with explicit traces, simple benchmark tasks, and configurable
scoring.

The core package is intentionally small:

- `AgentEvaluator` runs an agent against one or more tasks
- `AgentExecutionTrace` records step-by-step execution data
- metrics score outputs for accuracy, efficiency, safety, coherence,
  adaptability, and tool usage
- optional modules add semantic similarity, parameter search, and experimental
  neural policy components when extra dependencies are installed

This project is best understood as an inspectable evaluation harness, not as a
research benchmark or a claim of novel ML methodology. Several metrics are
heuristic by design, and the optional RL and optimization modules are utilities
for parameter tuning rather than a substitute for full-scale training systems.

## Install

Install the base dependencies:

```bash
pip install -r requirements.txt
```

Install developer tooling:

```bash
pip install -e .[dev]
```

## Quick Start

```python
from src.agents import BaseAgent
from src.benchmarks import TaskBenchmark
from src.evaluators import AgentEvaluator


class DemoAgent(BaseAgent):
    def execute(self, task, trace=None):
        return {"output": task.expected_output}


benchmark = TaskBenchmark.load("basic")
evaluator = AgentEvaluator({"use_tracer": True})
result = evaluator.evaluate(DemoAgent(), benchmark.tasks[0])

print(result.score)
print(result.metrics)
```

## What Is Included

- Core evaluation flow with task benchmarks and structured results
- Trace capture for agent actions and tool calls
- Configurable metric weights and success thresholds
- Optional semantic, Bayesian, and neural components behind graceful imports
- Test, lint, and type-check pipelines for the supported code paths

## What This Repo Does Not Claim

- It is not a drop-in replacement for production-scale eval platforms
- It does not present heuristic metrics as validated research instruments
- It does not treat the optional neural path as the default or required workflow

For a fuller description of the implementation and tradeoffs, see
[`docs/METHODS_AND_LIMITATIONS.md`](docs/METHODS_AND_LIMITATIONS.md).

## Development

Run the local checks:

```bash
pytest -q
black --check src tests
flake8 src tests --max-line-length=127 --max-complexity=10
mypy src --ignore-missing-imports
```

## Docs

- [`docs/API.md`](docs/API.md)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/METHODS_AND_LIMITATIONS.md`](docs/METHODS_AND_LIMITATIONS.md)
- [`METRICS.md`](METRICS.md)
- [`RL_GUIDE.md`](RL_GUIDE.md)

License: MIT.
