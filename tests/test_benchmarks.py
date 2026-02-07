"""
Tests for benchmark modules
"""

import pytest
import json
import tempfile
import os
from src.benchmarks import Task, TaskBenchmark, BaseBenchmark


class TestTask:
    """Test Task dataclass"""
    
    def test_create_task(self):
        """Test creating a task"""
        task = Task(
            id="test_001",
            description="Test task",
            input={"test": "data"},
            expected_output={"result": "success"}
        )
        assert task.id == "test_001"
        assert task.description == "Test task"
        assert task.input == {"test": "data"}
        assert task.expected_output == {"result": "success"}
    
    def test_task_with_metadata(self):
        """Test task with metadata"""
        task = Task(
            id="test_002",
            description="Test",
            input={},
            metadata={"category": "test", "difficulty": "easy"}
        )
        assert task.metadata == {"category": "test", "difficulty": "easy"}
    
    def test_check_success(self):
        """Test task success checking"""
        task = Task(
            id="test_003",
            description="Test",
            input={},
            expected_output={"result": "success"}
        )
        
        # Test with matching result
        assert task.check_success({"result": "success"}) == True
        
        # Test with non-matching result
        assert task.check_success({"result": "failure"}) == False


class TestTaskBenchmark:
    """Test TaskBenchmark class"""
    
    def test_default_tasks(self):
        """Test default task loading"""
        benchmark = TaskBenchmark()
        assert len(benchmark.tasks) > 0
        assert all(isinstance(task, Task) for task in benchmark.tasks)
    
    def test_load_from_file(self):
        """Test loading tasks from file"""
        # Create temporary config file
        tasks_data = {
            "tasks": [
                {
                    "id": "file_task_001",
                    "description": "Task from file",
                    "input": {"test": "data"},
                    "expected_output": {"result": "success"}
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(tasks_data, f)
            temp_path = f.name
        
        try:
            benchmark = TaskBenchmark(config={"tasks_file": temp_path})
            assert len(benchmark.tasks) == 1
            assert benchmark.tasks[0].id == "file_task_001"
        finally:
            os.unlink(temp_path)
    
    def test_load_classmethod(self):
        """Test load classmethod"""
        benchmark = TaskBenchmark.load("standard_tasks")
        assert isinstance(benchmark, TaskBenchmark)
        assert len(benchmark.tasks) > 0
    
    def test_benchmark_iteration(self):
        """Test iterating over benchmark"""
        benchmark = TaskBenchmark()
        tasks = list(benchmark.tasks)
        assert len(tasks) > 0


class TestBaseBenchmark:
    """Test BaseBenchmark abstract class"""
    
    def test_base_benchmark_abstract(self):
        """Test that BaseBenchmark is abstract"""
        with pytest.raises(TypeError):
            BaseBenchmark("test")
