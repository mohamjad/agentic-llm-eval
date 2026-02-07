# agentic llm behavior evaluation

framework for evaluating agentic llm behavior with RL-based fine-tuning. tracks accuracy, efficiency, safety, coherence, adaptability, and tool usage with full execution traces.

## what it does

- evaluates agents on tasks with multiple metrics (accuracy, efficiency, safety, coherence, adaptability, tool usage)
- captures full execution traces
- uses reinforcement learning to fine-tune agent behavior
- adjusts model parameters based on performance feedback
- aggregates everything into an overall score
- all calculations are explicit and traceable

## quick start

```python
from src.evaluators import AgentEvaluator
from src.benchmarks import TaskBenchmark
from src.agents.base import BaseAgent

# make an agent
class MyAgent(BaseAgent):
    def execute(self, task, trace=None):
        # your agent logic here
        return {"result": "done"}

# evaluate it
evaluator = AgentEvaluator()
benchmark = TaskBenchmark.load("standard_tasks")
agent = MyAgent()

results = evaluator.evaluate_batch(agent, benchmark.tasks)

for result in results:
    print(f"{result.task_id}: {result.score:.2f} - {'pass' if result.success else 'fail'}")
```

## installation

```bash
pip install -r requirements.txt
```

## project structure

```
src/
├── evaluators/    # main evaluation logic
├── benchmarks/    # test tasks
├── metrics/       # accuracy, efficiency, safety, coherence, adaptability
├── agents/        # agent interface and tracing
├── rl/            # RL training and parameter adjustment
└── utils/         # helpers

tests/             # tests
examples/          # example usage (including RL training)
configs/           # config files
```

## how it works

the evaluator wraps your agent, runs it on tasks, and tracks everything:
- each step gets recorded in a trace
- metrics are calculated separately (accuracy, efficiency, safety, tool usage)
- scores are combined using configurable weights
- success is determined by score threshold + task-specific checks

all the math is explicit - no black boxes. check the code if you want to see exactly how scores are calculated.

## configuration

```python
config = {
    "use_tracer": True,           # auto-trace agent execution
    "success_threshold": 0.7,     # min score to pass
    "metric_weights": {
        "accuracy": 0.4,
        "efficiency": 0.3,
        "tool_usage": 0.2,
        "safety": 0.1
    }
}
evaluator = AgentEvaluator(config=config)
```

## examples

- `examples/simple_evaluation.py` - basic evaluation
- `examples/rl_training_example.py` - RL-based fine-tuning
- `examples/parameter_impact_example.py` - test parameter effects

## metrics and hyperparameters

see `METRICS.md` for detailed explanation of:
- all evaluation metrics (accuracy, efficiency, safety, coherence, adaptability)
- tunable agent parameters (context_length, temperature, max_steps, etc.)
- how parameters affect behavior
- RL training parameters

## testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_evaluator.py -v
```

## code quality

```bash
# Format code
black src tests

# Lint
flake8 src tests

# Type check
mypy src

# Run all checks
make quality
```

## contributing

see `CONTRIBUTING.md` for guidelines.

## architecture

see `docs/ARCHITECTURE.md` for detailed architecture documentation.

## api reference

see `docs/API.md` for complete API documentation.

## license

MIT
