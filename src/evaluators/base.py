"""base evaluator interface"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """evaluation result"""
    task_id: str
    success: bool
    score: float
    metrics: Dict[str, Any]
    error: Optional[str] = None
    execution_trace: Optional[List[Dict[str, Any]]] = None


class BaseEvaluator(ABC):
    """base evaluator class"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def evaluate(self, agent: Any, task: Any) -> EvaluationResult:
        """evaluate agent on task"""
        pass
    
    def evaluate_batch(self, agent: Any, tasks: List[Any]) -> List[EvaluationResult]:
        """evaluate agent on multiple tasks"""
        return [self.evaluate(agent, task) for task in tasks]
