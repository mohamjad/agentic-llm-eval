"""
Tests for evaluator module

These tests verify that the evaluator works correctly with explicit,
explainable behavior.
"""

import pytest
from src.evaluators import AgentEvaluator, EvaluationResult
from src.benchmarks import Task
from src.agents.base import BaseAgent, AgentExecutionTrace


class MockAgent(BaseAgent):
    """
    Mock agent for testing that implements BaseAgent interface.
    
    This agent explicitly implements the interface and can optionally
    record execution traces.
    """
    
    def execute(self, task, trace: AgentExecutionTrace = None):
        """
        Execute a task and optionally record trace.
        
        Args:
            task: Task to execute
            trace: Optional trace object
            
        Returns:
            Result dictionary
        """
        if trace:
            trace.add_step(
                action_type="execution",
                action_name="mock_execute",
                input_data={"task_input": getattr(task, "input", {})},
                output_data={"result": "success"},
                duration=0.1
            )
        return {"result": "success"}


def test_evaluator_initialization():
    """Test evaluator can be initialized with explicit configuration"""
    evaluator = AgentEvaluator()
    assert evaluator is not None
    assert hasattr(evaluator, 'use_tracer')
    assert hasattr(evaluator, 'success_threshold')
    assert hasattr(evaluator, 'metric_weights')
    assert evaluator.use_tracer is True  # Default value
    assert evaluator.success_threshold == 0.7  # Default value
    assert "accuracy" in evaluator.metric_weights


def test_evaluator_custom_config():
    """Test evaluator accepts explicit configuration"""
    config = {
        "use_tracer": False,
        "success_threshold": 0.8,
        "metric_weights": {
            "accuracy": 0.5,
            "efficiency": 0.5
        }
    }
    evaluator = AgentEvaluator(config=config)
    assert evaluator.use_tracer is False
    assert evaluator.success_threshold == 0.8
    assert evaluator.metric_weights["accuracy"] == 0.5


def test_evaluate_success():
    """Test successful evaluation with explicit trace capture"""
    evaluator = AgentEvaluator()
    agent = MockAgent()
    task = Task(
        id="test_001",
        description="Test task",
        input={"test": "data"},
        expected_output={"result": "success"}
    )
    
    result = evaluator.evaluate(agent, task)
    
    assert isinstance(result, EvaluationResult)
    assert result.task_id == "test_001"
    assert result.execution_trace is not None
    assert len(result.execution_trace) > 0
    assert "accuracy" in result.metrics
    assert "efficiency" in result.metrics
    assert "safety_score" in result.metrics


def test_evaluate_batch():
    """Test batch evaluation processes all tasks"""
    evaluator = AgentEvaluator()
    agent = MockAgent()
    tasks = [
        Task(
            id=f"task_{i}",
            description=f"Task {i}",
            input={"test": i}
        )
        for i in range(3)
    ]
    
    results = evaluator.evaluate_batch(agent, tasks)
    
    assert len(results) == 3
    assert all(isinstance(r, EvaluationResult) for r in results)
    assert all(r.task_id.startswith("task_") for r in results)


def test_metrics_calculation():
    """Test that all metrics are calculated explicitly"""
    evaluator = AgentEvaluator()
    agent = MockAgent()
    task = Task(
        id="test_metrics",
        description="Test metrics",
        input={"test": "data"},
        expected_output={"result": "success"}
    )
    
    result = evaluator.evaluate(agent, task)
    
    # Verify all expected metrics are present
    assert "accuracy" in result.metrics
    assert "efficiency" in result.metrics
    assert "safety_score" in result.metrics
    assert "tool_usage" in result.metrics
    assert "overall_score" in result.metrics
    
    # Verify scores are in valid range
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert 0.0 <= result.metrics["efficiency"] <= 1.0
    assert 0.0 <= result.metrics["safety_score"] <= 1.0
    assert 0.0 <= result.score <= 1.0


def test_trace_capture():
    """Test that execution traces are captured correctly"""
    evaluator = AgentEvaluator()
    agent = MockAgent()
    task = Task(
        id="test_trace",
        description="Test trace",
        input={"test": "data"}
    )
    
    result = evaluator.evaluate(agent, task)
    
    assert result.execution_trace is not None
    assert isinstance(result.execution_trace, list)
    assert len(result.execution_trace) > 0
    
    # Verify trace structure
    first_step = result.execution_trace[0]
    assert "step_number" in first_step
    assert "action_type" in first_step
    assert "action_name" in first_step
    assert "input_data" in first_step
    assert "output_data" in first_step
