# RL-based fine-tuning guide

How to use reinforcement learning to fine-tune agent behavior.

The RL system adjusts agent parameters based on evaluation feedback to improve performance across different metrics. It works by evaluating the agent on tasks and calculating metrics, then using a policy network to calculate parameter adjustments based on those metrics, applying the updated parameters to the agent, repeating this process for multiple episodes, and updating policy weights based on improvements.

To get started, create an agent that supports parameters by extending BaseAgent and implementing execute. The agent should accept parameters through a set_parameters method. Setup involves creating the agent, an evaluator, and loading a benchmark. Then create an RLTrainer with the agent, evaluator, and benchmark, call train with the number of episodes, and use get_best_parameters to retrieve the optimal settings.

Context_length ranges from 100 to 5000 and controls how much context the agent processes. It affects accuracy and efficiency. Higher values provide more context leading to better accuracy but slower execution, while lower values are faster but might miss important information.

Temperature ranges from 0.1 to 2.0 and controls randomness and creativity. It affects coherence and adaptability. Lower values are more deterministic and precise, higher values are more creative and varied.

Max_steps ranges from 1 to 50 and sets the maximum execution steps. It affects efficiency and accuracy. Higher values allow more thorough execution but are slower, lower values are faster but might be incomplete.

Tool_usage_threshold ranges from 0.0 to 1.0 and controls when to use tools. It affects tool_usage and efficiency. Higher values use tools more conservatively, lower values use tools more aggressively.

Reasoning_depth ranges from 1 to 10 and controls depth of reasoning steps. It affects coherence and adaptability. Higher values enable deeper reasoning, lower values are faster but less thorough.

The policy network maps metrics to parameter adjustments. Low accuracy triggers increases in context_length and max_steps. Low efficiency triggers decreases in max_steps and tool_usage_threshold. Low coherence triggers increases in reasoning_depth and temperature adjustments. Low adaptability triggers increases in temperature and reasoning_depth. Weights are updated based on actual improvements during training.

Training parameters include episodes for number of training iterations defaulting to 10, tasks_per_episode for tasks per episode where None means all, learning_rate for how fast parameters adjust defaulting to 0.01, and update_policy for whether to update policy weights defaulting to True.

See examples/rl_training_example.py for complete RL training example. See examples/parameter_impact_example.py for testing how parameters affect behavior.

Best practices include starting with default parameters, training on diverse task sets, monitoring improvement per episode, using best parameters from training history, and testing on held-out tasks.

Interpreting results involves understanding that improvement greater than zero means parameters are helping, improvement approximately zero means parameters are stable and might be converged, and improvement less than zero might need learning rate adjustment or reset. Check training_history for detailed per-episode metrics.
