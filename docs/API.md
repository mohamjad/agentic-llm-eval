# API Documentation

Complete API reference for the agentic-llm-eval framework.

AgentEvaluator is the main evaluator class for assessing agent performance. Create one with an optional config dictionary that can include use_tracer for automatic tracing, success_threshold for minimum passing score, and metric_weights to customize how metrics contribute to the overall score.

The evaluate method takes an agent and task, runs the agent on the task, collects metrics, and returns an EvaluationResult with score, metrics, and trace. The evaluate_batch method does the same for multiple tasks, returning a list of EvaluationResult objects.

EvaluationResult contains the task_id, a success boolean indicating if the task passed, a score float between 0.0 and 1.0, a metrics dictionary with detailed metric values, an optional error string if something went wrong, and an optional execution_trace list showing what the agent did.

BaseAgent is the abstract base class that all agents must extend. Implement the execute method that takes a task and optional trace parameter. If you want automatic tracing, the AgentTracer wrapper can handle that for you. The trace parameter is an AgentExecutionTrace object that you can add steps to using add_step with action_type, action_name, input_data, output_data, and duration.

TaskBenchmark loads and manages task sets. Use TaskBenchmark.load with a benchmark name to get a benchmark instance. The tasks attribute contains the list of Task objects. Each Task has an id, input, expected_output, description, and optional metadata.

Metrics are collected through metric classes. AccuracyMetric compares results to expected outputs, handling dictionaries with partial matching, lists with element-wise comparison, and primitives with exact matching. EfficiencyMetric calculates efficiency based on steps and execution time. SafetyMetric detects unsafe content using keyword detection and security patterns. CoherenceMetric measures logical consistency and topic relevance. AdaptabilityMetric measures how well agents adapt to different contexts.

RL training uses RLTrainer which takes an agent, evaluator, and benchmark. The train method runs training episodes, adjusting parameters based on performance. You can specify episodes, tasks_per_episode, and whether to update policy weights. The get_best_parameters method returns the parameters from the best performing episode.

AgentParameters defines tunable parameters with validation. Context_length ranges from 100 to 5000, temperature from 0.1 to 2.0, max_steps from 1 to 50, tool_usage_threshold from 0.0 to 1.0, and reasoning_depth from 1 to 10. All parameters are validated on initialization to ensure they stay within bounds.

Configuration uses the Config class which loads from defaults, files, and environment variables. Use Config.get to retrieve values, Config.set to update them, and Config.save_to_file to persist changes. Environment variables use a prefix and are automatically loaded.

Validation utilities help ensure data integrity. validate_agent_parameters checks parameter values are within bounds, validate_metrics ensures metric values are valid, validate_task_input checks task inputs for security issues, and validate_config validates configuration dictionaries.

For advanced features, DeepPolicyNetwork provides neural network-based RL with PyTorch. SemanticMetric uses sentence transformers for semantic similarity. BayesianParameterOptimizer uses Gaussian Process regression for hyperparameter tuning. StatisticalAnalyzer provides hypothesis testing, confidence intervals, and effect size calculations.

All optional features gracefully degrade if dependencies aren't available. The framework works fine without PyTorch, sentence-transformers, or scikit-optimize, but those specific features won't be available.
