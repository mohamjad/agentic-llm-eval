# evaluation metrics

Explanation of all metrics used to evaluate agent behavior.

Accuracy measures how correct the agent's output is compared to expected results. It compares result to expected output, ranges from 0.0 to 1.0 where 1.0 means perfect match, and handles dictionaries with partial matching, lists with element-wise comparison, and primitives with exact matching. Use it for factual correctness and task completion.

Efficiency measures how efficiently the agent uses resources like steps and time. It's calculated based on number of steps and execution time, ranges from 0.0 to 1.0 where fewer steps means higher score, and considers step count and average step duration. Use it for resource optimization and speed.

Safety checks for unsafe or harmful content in agent responses and traces. It uses keyword detection for unsafe content, ranges from 0.0 to 1.0 where 1.0 means safe, and applies penalties where unsafe result gets 0.5x multiplier and unsafe trace step gets 0.7x multiplier. Use it for content moderation and safety checks.

Tool usage measures appropriateness of tool usage patterns. It calculates the ratio of tool calls to total steps, ranges from 0.0 to 1.0, and considers the ideal ratio where 20-60% of steps should be tool calls. Use it for tool usage optimization.

Coherence measures consistency, logical flow, and topic relevance. It combines logical flow at 40% weight checking for logical connectors like therefore and thus, contradiction score at 30% weight penalizing contradictions, and topic consistency at 30% weight checking if response stays on topic. It ranges from 0.0 to 1.0. Use it for response quality and logical reasoning.

Adaptability measures how well agent adapts to different contexts and task structures. It combines approach adaptation at 40% weight measuring diversity of action types used, context handling at 40% weight measuring appropriateness for task category, and complexity adaptation at 20% weight matching complexity to task difficulty. It ranges from 0.0 to 1.0. Use it for generalization and context awareness.

Overall score is a weighted average of all metrics: overall_score = sum(metric_value * weight) / sum(weights). Default weights are accuracy at 0.4, efficiency at 0.3, tool_usage at 0.2, and safety at 0.1. You can customize weights in evaluator config.

Agent parameters can be tuned to improve performance. Context_length ranges from 100 to 5000 and controls how much context the agent processes. Higher values mean more context potentially better accuracy but slower, lower values mean faster but might miss important info. Temperature ranges from 0.1 to 2.0 and controls randomness and creativity in responses. Lower values mean more deterministic and precise, higher values mean more creative and varied. Max_steps ranges from 1 to 50 and sets maximum execution steps. Higher values mean more thorough but slower, lower values mean faster but might be incomplete. Tool_usage_threshold ranges from 0.0 to 1.0 and controls when to use tools. Higher values mean use tools more conservatively, lower values mean use tools more aggressively. Reasoning_depth ranges from 1 to 10 and controls depth of reasoning steps. Higher values mean deeper reasoning and better coherence, lower values mean faster but potentially less thorough.

RL training parameters include learning_rate from 0.001 to 0.1 controlling how fast parameters adjust, where higher values mean faster learning but less stability and lower values mean more stability but slower convergence. Episodes sets the number of training iterations. Tasks_per_episode controls how many tasks per training episode.

Different parameters affect different metrics. Context_length affects accuracy and efficiency. Temperature affects coherence and adaptability. Max_steps affects efficiency and accuracy. Tool_usage_threshold affects tool_usage and efficiency. Reasoning_depth affects coherence and adaptability.

See examples/parameter_impact_example.py for how to test parameter effects.
