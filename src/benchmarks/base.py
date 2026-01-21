"""benchmark and task base classes"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class Task:
    """single evaluation task"""
    id: str
    description: str
    input: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def check_success(self, result: Any) -> bool:
        """check if result matches expected output"""
        if self.expected_output is None:
            return True  # No expected output means any result is acceptable
        
        # Handle dictionary comparison explicitly
        if isinstance(self.expected_output, dict) and isinstance(result, dict):
            return self._compare_dicts(self.expected_output, result)
        
        # Direct equality comparison for other types
        return result == self.expected_output
    
    def _compare_dicts(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """compare dicts - all expected keys must match"""
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            if actual[key] != expected_value:
                return False
        return True


class BaseBenchmark(ABC):
    """base class for benchmarks"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.tasks: List[Task] = []
    
    @abstractmethod
    def load_tasks(self) -> List[Task]:
        """load tasks"""
        pass
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """get task by id"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __iter__(self):
        return iter(self.tasks)
