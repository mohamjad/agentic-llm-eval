"""Regression tests for public imports and optional dependency fallbacks."""

from src import AgentEvaluator, AgentExecutionTrace, RLTrainer
from src.evaluators import EvaluationResult
from src.rl import AgentParameters


def test_public_api_imports_without_optional_ml_stack():
    assert AgentEvaluator is not None
    assert RLTrainer is not None
    assert AgentParameters is not None
    assert AgentExecutionTrace is not None


def test_default_metric_weights_include_safety_score():
    evaluator = AgentEvaluator()

    assert evaluator.metric_weights["safety_score"] == 0.1
    assert "safety" not in evaluator.metric_weights


def test_evaluation_result_stays_constructible_from_public_api():
    result = EvaluationResult(task_id="task", success=True, score=0.9, metrics={"accuracy": 1.0})

    assert result.task_id == "task"
    assert result.success is True
