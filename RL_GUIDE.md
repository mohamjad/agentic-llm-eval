# Parameter Tuning Guide

This repo includes a small training loop for iterating on agent parameters using
evaluation feedback.

The base workflow is:

1. Evaluate an agent on a task set
2. Adjust the agent parameters
3. Re-run the evaluation
4. Keep the best-performing settings

This is useful for controlled tuning experiments. It should not be confused with
full-scale RL training infrastructure.

## Required agent behavior

Your agent should:

- extend `BaseAgent`
- implement `execute`
- expose a `set_parameters` method if you want to use `RLTrainer`

## Tuned parameters

- `context_length`
- `temperature`
- `max_steps`
- `tool_usage_threshold`
- `reasoning_depth`

These parameters are bounded and validated by `AgentParameters`.

## Trainer behavior

`RLTrainer` runs repeated episodes, records the before/after metrics, and keeps
training history that you can inspect later.

The default policy logic uses explicit parameter adjustments driven by score
changes. Optional neural and Bayesian modules can be enabled when their extra
dependencies are available.

## Good use cases

- comparing prompt or parameter presets
- finding a stable operating point for a benchmark
- regression testing after agent changes

## Cautions

- benchmark overfitting is easy if you tune on too few tasks
- improvements on heuristic metrics do not guarantee better real-world behavior
- the optional neural path is experimental and should be treated that way

See `examples/rl_training_example.py` and
`examples/parameter_impact_example.py` for concrete usage.
