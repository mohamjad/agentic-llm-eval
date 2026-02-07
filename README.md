# agentic llm behavior evaluation

A comprehensive framework for evaluating agentic LLM behavior with deep reinforcement learning, semantic analysis, and Bayesian optimization. Features neural network-based policy gradients (PPO), embedding-based metrics, statistical significance testing, and theoretical foundations.

The framework evaluates agents on tasks using multiple metrics like accuracy, efficiency, safety, coherence, adaptability, and tool usage. It captures full execution traces, uses deep reinforcement learning to fine-tune agent behavior through neural network-based policy optimization with proper backpropagation. It includes semantic similarity metrics using transformer embeddings, Bayesian optimization for hyperparameter tuning, and statistical significance testing with confidence intervals. The system adjusts model parameters based on performance feedback and aggregates everything into an overall score. All calculations are explicit and traceable, so you can see exactly how scores are computed.

To get started, create an agent by extending BaseAgent and implementing the execute method. The evaluator wraps your agent, runs it on tasks, and tracks everything. Each step gets recorded in a trace, metrics are calculated separately, scores are combined using configurable weights, and success is determined by score threshold plus task-specific checks. The math is explicit with no black boxes.

Installation is straightforward. Run pip install -r requirements.txt to get all dependencies. The project structure is clean with source code in src, tests in tests, examples in examples, documentation in docs, and configuration files in configs.

The evaluator can be configured with options like use_tracer for automatic tracing, success_threshold for minimum passing score, and metric_weights to customize how different metrics contribute to the overall score. By default accuracy gets 40% weight, efficiency 30%, tool usage 20%, and safety 10%, but you can adjust these to match your priorities.

Examples are provided in the examples directory. The simple_evaluation script shows basic usage, rl_training_example demonstrates RL-based fine-tuning, parameter_impact_example shows how to test parameter effects, and advanced_features covers neural network RL, semantic metrics, and Bayesian optimization.

For detailed metric explanations including how accuracy, efficiency, safety, coherence, and adaptability work, see METRICS.md. That document also covers tunable agent parameters like context_length, temperature, max_steps, tool_usage_threshold, and reasoning_depth, along with how these parameters affect behavior and what RL training parameters are available.

Testing is done with pytest. Run pytest tests/ -v for all tests, or pytest tests/ -v --cov=src --cov-report=html for coverage reports. You can also run specific test files like pytest tests/test_evaluator.py -v.

Code quality checks use black for formatting, flake8 for linting, and mypy for type checking. Run black src tests to format, flake8 src tests to lint, mypy src to type check, or make quality to run all checks at once.

For contribution guidelines see CONTRIBUTING.md. Architecture details are in docs/ARCHITECTURE.md. Complete API documentation is in docs/API.md. Mathematical foundations including PPO algorithm derivation, Generalized Advantage Estimation, Bayesian optimization theory, statistical test formulations, and convergence analysis are documented in docs/THEORY.md.

The framework includes advanced capabilities like neural network RL using PyTorch-based policy and value networks with PPO, semantic metrics using transformer embeddings for semantic similarity, Bayesian optimization with Gaussian Process-based hyperparameter tuning, and statistical analysis with hypothesis testing, confidence intervals, and effect sizes. See examples/advanced_features.py for usage examples.

License is MIT.
