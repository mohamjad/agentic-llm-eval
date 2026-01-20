"""standard task benchmark"""

from typing import List, Optional, Dict, Any
from .base import BaseBenchmark, Task
import json
import os


class TaskBenchmark(BaseBenchmark):
    """standard task benchmark"""
    
    def __init__(self, name: str = "standard_tasks", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.tasks = self.load_tasks()
    
    def load_tasks(self) -> List[Task]:
        """load tasks from config file or use defaults"""
        # Try to load from config file
        config_path = self.config.get("tasks_file")
        if config_path and os.path.exists(config_path):
            return self._load_from_file(config_path)
        
        # Return default tasks
        return self._get_default_tasks()
    
    def _load_from_file(self, filepath: str) -> List[Task]:
        """load tasks from json file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tasks = []
        for task_data in data.get("tasks", []):
            task = Task(
                id=task_data["id"],
                description=task_data["description"],
                input=task_data["input"],
                expected_output=task_data.get("expected_output"),
                metadata=task_data.get("metadata")
            )
            tasks.append(task)
        
        return tasks
    
    def _get_default_tasks(self) -> List[Task]:
        """default test tasks"""
        return [
            Task(
                id="task_001",
                description="Simple information retrieval",
                input={"query": "What is the capital of France?"},
                expected_output={"answer": "Paris"},
                metadata={"category": "knowledge", "difficulty": "easy"}
            ),
            Task(
                id="task_002",
                description="Multi-step reasoning",
                input={"problem": "If Alice has 5 apples and gives 2 to Bob, how many does she have left?"},
                expected_output={"answer": 3},
                metadata={"category": "reasoning", "difficulty": "easy"}
            ),
            Task(
                id="task_003",
                description="Tool usage task",
                input={"action": "calculate", "expression": "2 + 2 * 3"},
                expected_output={"result": 8},
                metadata={"category": "tool_usage", "difficulty": "medium"}
            ),
        ]
    
    @classmethod
    def load(cls, name: str, config: Optional[Dict[str, Any]] = None) -> "TaskBenchmark":
        """load benchmark by name"""
        return cls(name=name, config=config)
