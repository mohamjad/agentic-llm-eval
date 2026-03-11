# Examples

This directory contains example scripts demonstrating how to use the project.

simple_evaluation.py shows the basic evaluation workflow by evaluating a simple agent on standard tasks. Run it with python examples/simple_evaluation.py. It demonstrates creating a custom agent, setting up an evaluator, running evaluation on tasks, and viewing detailed results.

rl_training_example.py shows the parameter-tuning workflow. Run it with python examples/rl_training_example.py. It demonstrates creating a parameterized agent, setting up the trainer, iterating over episodes, and tracking performance changes.

parameter_impact_example.py demonstrates how different agent parameters affect performance metrics. Run it with python examples/parameter_impact_example.py. It demonstrates testing different parameter values, comparing metric impacts, and understanding parameter trade-offs.

advanced_features.py demonstrates optional components including semantic metrics, parameter search, statistics helpers, and the experimental neural path. Run it with python examples/advanced_features.py. It requires extra packages such as torch, sentence-transformers, scikit-optimize, and scipy.

To create your own agent, extend BaseAgent and implement the execute method. The agent receives a task and optional trace parameter. If you want automatic tracing, use the trace parameter to record steps. Return a result dictionary. Then evaluate it using AgentEvaluator with a TaskBenchmark.

Next steps include reading docs/API.md for the public surface, docs/ARCHITECTURE.md for the package layout, METRICS.md for metric behavior, and RL_GUIDE.md for parameter-tuning guidance.
