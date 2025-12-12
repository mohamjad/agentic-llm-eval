# RL-based fine-tuning guide

how to use reinforcement learning to fine-tune agent behavior.

## overview

the RL system adjusts agent parameters based on evaluation feedback to improve performance across different metrics.

## quick start

```python
from src.rl import RLTrainer, AgentParameters
from src.evaluators import AgentEvaluator
from src.benchmarks import TaskBenchmark
from src.agents.base import BaseAgent

# create agent that supports parameters
class MyAgent(BaseAgent):
    def __init__(self):
        self.params = AgentParameters()
    
    def execute(self, task, trace=None):
        # use self.params to adjust behavior
        return {"result": "done"}
    
    def set_parameters(self, params_dict):
        self.params = AgentParameters.from_dict(params_dict)

# setup
agent = MyAgent()
evaluator = AgentEvaluator()
benchmark = TaskBenchmark.load("standard_tasks")

# train
trainer = RLTrainer(agent, evaluator, benchmark)
result = trainer.train(episodes=10)

# get best parameters
best_params = trainer.get_best_parameters()
```

## how it works

1. **evaluate**: agent runs on tasks, metrics are calculated
2. **adjust**: policy network calculates parameter adjustments based on metrics
3. **apply**: parameters are updated on agent
4. **repeat**: process repeats for multiple episodes
5. **learn**: policy weights update based on improvements

## tunable parameters

### context_length
how much context the agent processes (100-5000).

- **affects**: accuracy, efficiency
- **higher**: more context = better accuracy but slower
- **lower**: faster but might miss info

### temperature
randomness/creativity (0.1-2.0).

- **affects**: coherence, adaptability
- **lower**: more deterministic, precise
- **higher**: more creative, varied

### max_steps
maximum execution steps (1-50).

- **affects**: efficiency, accuracy
- **higher**: more thorough but slower
- **lower**: faster but might be incomplete

### tool_usage_threshold
when to use tools (0.0-1.0).

- **affects**: tool_usage, efficiency
- **higher**: use tools conservatively
- **lower**: use tools aggressively

### reasoning_depth
depth of reasoning steps (1-10).

- **affects**: coherence, adaptability
- **higher**: deeper reasoning
- **lower**: faster but less thorough

## policy network

the policy network maps metrics to parameter adjustments:

- low accuracy → increase context_length, max_steps
- low efficiency → decrease max_steps, tool_usage_threshold
- low coherence → increase reasoning_depth, adjust temperature
- low adaptability → increase temperature, reasoning_depth

weights are updated based on actual improvements during training.

## training parameters

- **episodes**: number of training iterations (default: 10)
- **tasks_per_episode**: tasks per episode (None = all)
- **learning_rate**: how fast parameters adjust (default: 0.01)
- **update_policy**: whether to update policy weights (default: True)

## examples

see `examples/rl_training_example.py` for complete RL training example.

see `examples/parameter_impact_example.py` for testing how parameters affect behavior.

## best practices

1. start with default parameters
2. train on diverse task set
3. monitor improvement per episode
4. use best parameters from training history
5. test on held-out tasks

## interpreting results

- **improvement > 0**: parameters are helping
- **improvement ≈ 0**: parameters stable, might be converged
- **improvement < 0**: might need to adjust learning rate or reset

check `training_history` for detailed per-episode metrics.
