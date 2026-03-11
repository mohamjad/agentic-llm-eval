"""
Lightweight tools for evaluating agent-like LLM workflows.

The package focuses on explicit traces, simple benchmarks, configurable metric
aggregation, and optional experimental components that degrade gracefully when
their heavy dependencies are unavailable.
"""

__version__ = "0.3.0"
__author__ = "mohamjad"
__email__ = "mohammedamjad526@gmail.com"
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
