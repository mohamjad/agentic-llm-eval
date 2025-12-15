# evaluation metrics

explanation of all metrics used to evaluate agent behavior.

## core metrics

### accuracy
measures how correct the agent's output is compared to expected results.

- **calculation**: compares result to expected output
- **range**: 0.0 to 1.0 (1.0 = perfect match)
- **handles**: dicts (partial matching), lists (element-wise), primitives (exact match)
- **use case**: factual correctness, task completion

### efficiency
measures how efficiently the agent uses resources (steps, time).

- **calculation**: based on number of steps and execution time
- **range**: 0.0 to 1.0 (fewer steps = higher score)
- **factors**: step count, average step duration
- **use case**: resource optimization, speed

### safety
checks for unsafe or harmful content in agent responses and traces.

- **calculation**: keyword detection for unsafe content
- **range**: 0.0 to 1.0 (1.0 = safe)
- **penalties**: unsafe result = 0.5x, unsafe trace step = 0.7x
- **use case**: content moderation, safety checks

### tool usage
measures appropriateness of tool usage patterns.

- **calculation**: ratio of tool calls to total steps
- **range**: 0.0 to 1.0
- **ideal ratio**: 20-60% of steps should be tool calls
- **use case**: tool usage optimization

## advanced metrics

### coherence
measures consistency, logical flow, and topic relevance.

- **components**:
  - logical flow (0.4): checks for logical connectors (therefore, thus, etc.)
  - contradiction score (0.3): penalizes contradictions
  - topic consistency (0.3): checks if response stays on topic
- **range**: 0.0 to 1.0
- **use case**: response quality, logical reasoning

### adaptability
measures how well agent adapts to different contexts and task structures.

- **components**:
  - approach adaptation (0.4): diversity of action types used
  - context handling (0.4): appropriateness for task category
  - complexity adaptation (0.2): matches complexity to task difficulty
- **range**: 0.0 to 1.0
- **use case**: generalization, context awareness

## overall score

weighted average of all metrics:

```
overall_score = sum(metric_value * weight) / sum(weights)
```

default weights:
- accuracy: 0.4
- efficiency: 0.3
- tool_usage: 0.2
- safety: 0.1

you can customize weights in evaluator config.

## hyperparameters

### agent parameters

these can be tuned to improve performance:

- **context_length** (100-5000): how much context agent processes
  - higher = more context, potentially better accuracy but slower
  - lower = faster but might miss important info

- **temperature** (0.1-2.0): randomness/creativity in responses
  - lower = more deterministic, precise
  - higher = more creative, varied

- **max_steps** (1-50): maximum execution steps
  - higher = more thorough but slower
  - lower = faster but might be incomplete

- **tool_usage_threshold** (0.0-1.0): when to use tools
  - higher = use tools more conservatively
  - lower = use tools more aggressively

- **reasoning_depth** (1-10): depth of reasoning steps
  - higher = deeper reasoning, better coherence
  - lower = faster but potentially less thorough

### RL training parameters

- **learning_rate** (0.001-0.1): how fast parameters adjust
  - higher = faster learning but less stable
  - lower = more stable but slower convergence

- **episodes**: number of training iterations
- **tasks_per_episode**: how many tasks per training episode

## parameter impact

different parameters affect different metrics:

- **context_length**: affects accuracy, efficiency
- **temperature**: affects coherence, adaptability
- **max_steps**: affects efficiency, accuracy
- **tool_usage_threshold**: affects tool_usage, efficiency
- **reasoning_depth**: affects coherence, adaptability

see `examples/parameter_impact_example.py` for how to test parameter effects.
