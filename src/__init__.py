"""
Agentic LLM Behavior Evaluation Framework

A comprehensive framework for evaluating agentic LLM behavior with RL-based
fine-tuning. Tracks accuracy, efficiency, safety, coherence, adaptability,
and tool usage with full execution traces.
"""

__version__ = "0.2.0"
__author__ = "mohamjad"
__email__ = ""
__url__ = "https://github.com/mohamjad/agentic-llm-eval"

# Main exports for easy importing
from .evaluators import AgentEvaluator, BaseEvaluator, EvaluationResult
from .benchmarks import TaskBenchmark, Task
from .agents import BaseAgent, AgentExecutionTrace
from .rl import RLTrainer, AgentParameters

__all__ = [
    "__version__",
    "AgentEvaluator",
    "BaseEvaluator",
    "EvaluationResult",
    "TaskBenchmark",
    "Task",
    "BaseAgent",
    "AgentExecutionTrace",
    "RLTrainer",
    "AgentParameters",
]
