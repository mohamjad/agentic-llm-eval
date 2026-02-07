# Examples

This directory contains example scripts demonstrating how to use the agentic-llm-eval framework.

## Available Examples

### `simple_evaluation.py`
Basic evaluation workflow - evaluates a simple agent on standard tasks.

**Run:**
```bash
python examples/simple_evaluation.py
```

**What it demonstrates:**
- Creating a custom agent
- Setting up an evaluator
- Running evaluation on tasks
- Viewing detailed results

### `rl_training_example.py`
RL-based fine-tuning - shows how to use reinforcement learning to improve agent parameters.

**Run:**
```bash
python examples/rl_training_example.py
```

**What it demonstrates:**
- Creating a parameterized agent
- Setting up RL trainer
- Training over multiple episodes
- Tracking performance improvements

### `parameter_impact_example.py`
Parameter testing - demonstrates how different agent parameters affect performance metrics.

**Run:**
```bash
python examples/parameter_impact_example.py
```

**What it demonstrates:**
- Testing different parameter values
- Comparing metric impacts
- Understanding parameter trade-offs

## Creating Your Own Agent

```python
from src.agents.base import BaseAgent, AgentExecutionTrace
from src.evaluators import AgentEvaluator
from src.benchmarks import TaskBenchmark

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

# Evaluate
evaluator = AgentEvaluator()
benchmark = TaskBenchmark.load("standard_tasks")
agent = MyAgent()
results = evaluator.evaluate_batch(agent, benchmark.tasks)
```

## Next Steps

- Read the [API Documentation](../docs/API.md) for detailed API reference
- Check [Architecture Documentation](../docs/ARCHITECTURE.md) for design details
- See [METRICS.md](../METRICS.md) for metric explanations
- Review [RL_GUIDE.md](../RL_GUIDE.md) for RL training guidance
