# Methods and Limitations

This repository is designed as a practical evaluation toolkit for agent-like
LLM workflows. The implementation is intentionally inspectable and should be
read as engineering infrastructure, not as a research paper.

## What the package does well

- Runs repeatable task evaluations with structured results
- Captures execution traces for later inspection
- Provides a small library of scoring functions with configurable weights
- Supports optional semantic, statistical, Bayesian, and neural modules when
  extra dependencies are installed

## Metric caveats

Most built-in metrics are heuristic. They are useful for smoke tests, regression
checks, and internal comparisons, but they are not validated scientific
instruments.

- `AccuracyMetric` is the strongest of the defaults because it compares outputs
  against task expectations directly
- `EfficiencyMetric`, `CoherenceMetric`, `AdaptabilityMetric`, and
  `SafetyMetric` rely on simplified rules and pattern matching
- metric weights are policy choices, not objective truth

## Parameter-tuning caveats

The trainer and policy modules focus on tuning a small parameter set for an
agent wrapper. They are helpful when you want a reproducible evaluate-adjust
loop, but they should not be described as a replacement for full RL training
systems.

- the default path is lightweight and table-driven
- the neural path is optional and dependency-heavy
- the Bayesian path is a search utility, not a claim of closed-form optimality

## Expected use cases

- local agent regressions
- benchmark harnesses for small experimental tasks
- explicit trace capture for debugging
- side-by-side comparisons between prompt or parameter variants

## Non-goals

- proving benchmark validity
- presenting heuristic metrics as universal evaluation standards
- claiming state-of-the-art RL or optimization results
- acting as a full hosted eval platform
