"""Tests for optional modules and runtime helpers."""

from datetime import datetime
import logging

import numpy as np
import pytest

from src.agents.base import AgentExecutionTrace
from src.agents.tracer import trace_tool_call
from src.metrics import semantic as semantic_module
from src.rl.bayesian_optimizer import BayesianParameterOptimizer
from src.utils import StatisticalAnalyzer, load_config, setup_logger


class DummyEvaluator:
    def evaluate_batch(self, agent, tasks):
        return [type("Result", (), {"score": 0.4})(), type("Result", (), {"score": 0.8})()]


class DummyAgent:
    def __init__(self):
        self.params = None

    def set_parameters(self, params):
        self.params = params


class DummyBenchmark:
    def __init__(self):
        self.tasks = list(range(12))


class FakeSentenceTransformer:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        rows = []
        for text in texts:
            rows.append([
                float(len(text)),
                float(sum(1 for char in text.lower() if char in "aeiou")),
                float(len(set(text.split()))),
            ])
        return np.array(rows, dtype=float)


def build_semantic_metric(monkeypatch):
    monkeypatch.setattr(semantic_module, "EMBEDDINGS_AVAILABLE", True)
    monkeypatch.setattr(semantic_module, "SentenceTransformer", FakeSentenceTransformer)
    return semantic_module.SemanticMetric(model_name="fake-mini", device="cpu")


def test_semantic_metric_collects_expected_scores(monkeypatch):
    metric = build_semantic_metric(monkeypatch)
    task = type("Task", (), {"description": "Explain Python", "expected_output": "Python is a language"})()
    trace = [{"output_data": {"thought": "Python is useful"}}, {"output_data": {"thought": "Python has syntax"}}]

    metrics = metric.collect(None, task, {"answer": "Python is a programming language"}, trace)

    assert metrics["semantic_accuracy"] >= 0.5
    assert metrics["semantic_coherence"] >= 0.5
    assert metrics["topic_consistency"] >= 0.5


def test_semantic_metric_helpers_and_cache(monkeypatch):
    metric = build_semantic_metric(monkeypatch)
    cache = semantic_module.EmbeddingCache(max_size=1)

    assert metric.encode([]).size == 0
    assert metric.semantic_coherence(["only one"]) == 1.0
    assert metric.topic_consistency("", "response") == 0.0

    cache.set("alpha", np.array([1.0, 0.0]))
    assert np.array_equal(cache.get("alpha"), np.array([1.0, 0.0]))
    cache.set("beta", np.array([0.0, 1.0]))
    assert cache.get("alpha") is None
    cache.clear()
    assert cache.get("beta") is None


def test_bayesian_optimizer_tracks_history(monkeypatch):
    optimizer = BayesianParameterOptimizer(n_calls=4, n_initial_points=2, random_state=7)
    agent = DummyAgent()
    benchmark = DummyBenchmark()

    result = optimizer.objective_function((512, 0.8, 12, 0.4, 3), DummyEvaluator(), agent, benchmark)

    assert result == pytest.approx(-0.6)
    assert agent.params["context_length"] == 512
    assert optimizer.best_score == pytest.approx(0.6)
    assert len(optimizer.optimization_history) == 1


def test_bayesian_optimizer_optimize_and_uncertainty(monkeypatch):
    optimizer = BayesianParameterOptimizer(n_calls=3, n_initial_points=1, random_state=3)
    calls = {}

    class FakeResult:
        x = [300, 0.5, 7, 0.2, 2]
        fun = -0.77
        func_vals = np.array([-0.2, -0.4, -0.77])

    def fake_gp_minimize(func, dimensions, n_calls, n_initial_points, acq_func, random_state, callback):
        calls["dimensions"] = dimensions
        calls["n_calls"] = n_calls
        value = func([300, 0.5, 7, 0.2, 2])
        assert value == pytest.approx(-0.6)
        return FakeResult()

    monkeypatch.setattr("src.rl.bayesian_optimizer.gp_minimize", fake_gp_minimize)

    result = optimizer.optimize(DummyEvaluator(), DummyAgent(), DummyBenchmark())
    uncertainty = optimizer.get_uncertainty_estimate(result["best_params"])

    assert calls["n_calls"] == 3
    assert len(calls["dimensions"]) == 5
    assert result["best_score"] == pytest.approx(0.77)
    assert result["convergence"] == [-0.2, -0.4, -0.77]
    assert 0.0 <= uncertainty <= 1.0


def test_trace_tool_call_records_success_and_error():
    trace = AgentExecutionTrace(task_id="task", start_time=datetime.now())

    @trace_tool_call(trace, "calculator")
    def add(left, right):
        return left + right

    @trace_tool_call(trace, "unstable")
    def explode():
        raise RuntimeError("boom")

    assert add(2, 3) == 5
    with pytest.raises(RuntimeError):
        explode()

    assert trace.steps[0].action_type == "tool_call"
    assert trace.steps[0].action_name == "calculator"
    assert trace.steps[1].action_type == "tool_call_error"
    assert trace.steps[1].action_name == "unstable"


def test_logger_and_utils_exports(tmp_path):
    logger = setup_logger("agentic.tests.logger", level=logging.DEBUG)
    logger_again = setup_logger("agentic.tests.logger", level=logging.ERROR)

    config_path = tmp_path / "config.json"
    config_path.write_text('{"evaluator": {"success_threshold": 0.65}}', encoding="utf-8")
    config = load_config(str(config_path))

    assert logger is logger_again
    assert logger.level == logging.ERROR
    assert logger.propagate is False
    assert StatisticalAnalyzer is not None
    assert config.get("evaluator.success_threshold") == 0.65
