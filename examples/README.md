# Examples

This directory contains example scripts demonstrating how to use the agentic-llm-eval framework.

simple_evaluation.py shows the basic evaluation workflow by evaluating a simple agent on standard tasks. Run it with python examples/simple_evaluation.py. It demonstrates creating a custom agent, setting up an evaluator, running evaluation on tasks, and viewing detailed results.

rl_training_example.py shows RL-based fine-tuning by using reinforcement learning to improve agent parameters. Run it with python examples/rl_training_example.py. It demonstrates creating a parameterized agent, setting up RL trainer, training over multiple episodes, and tracking performance improvements.

parameter_impact_example.py demonstrates how different agent parameters affect performance metrics. Run it with python examples/parameter_impact_example.py. It demonstrates testing different parameter values, comparing metric impacts, and understanding parameter trade-offs.

advanced_features.py demonstrates advanced capabilities including neural network RL, semantic metrics, Bayesian optimization, and statistical analysis. Run it with python examples/advanced_features.py. It requires torch for neural network RL, sentence-transformers for semantic metrics, scikit-optimize for Bayesian optimization, and scipy for statistical analysis. It demonstrates neural network-based policy optimization using PPO, semantic similarity using transformer embeddings, Bayesian hyperparameter optimization, and statistical significance testing.

To create your own agent, extend BaseAgent and implement the execute method. The agent receives a task and optional trace parameter. If you want automatic tracing, use the trace parameter to record steps. Return a result dictionary. Then evaluate it using AgentEvaluator with a TaskBenchmark.

Next steps include reading the API Documentation in docs/API.md for detailed API reference, checking Architecture Documentation in docs/ARCHITECTURE.md for design details, seeing METRICS.md for metric explanations, and reviewing RL_GUIDE.md for RL training guidance.
