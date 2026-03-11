"""Tests for configuration, validation, and statistics helpers."""

import json

import pytest

from src.utils.config import Config
from src.utils.statistics import StatisticalAnalyzer
from src.utils.validation import (
    ValidationError,
    safe_validate,
    validate_agent_parameters,
    validate_metrics,
    validate_task_input,
)


def test_config_load_from_dict_and_dot_lookup():
    config = Config().load_from_dict(
        {
            "evaluator": {"success_threshold": 0.8},
            "logging": {"level": "DEBUG"},
        }
    )

    assert config.get("evaluator.success_threshold") == 0.8
    assert config.get("logging.level") == "DEBUG"


def test_config_load_from_file_merges_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"evaluator": {"success_threshold": 0.9}, "rl": {"learning_rate": 0.02}}),
        encoding="utf-8",
    )

    config = Config().load_from_file(config_path)

    assert config.get("evaluator.success_threshold") == 0.9
    assert config.get("rl.learning_rate") == 0.02


def test_config_validate_rejects_bad_threshold():
    config = Config().load_from_dict({"evaluator": {"success_threshold": 1.2}})

    with pytest.raises(ValueError):
        config.validate()


def test_validate_agent_parameters_rejects_wrong_type():
    params = {"context_length": 50}

    with pytest.raises(ValidationError):
        validate_agent_parameters(params)


def test_validate_metrics_rejects_invalid_scores():
    with pytest.raises(ValidationError):
        validate_metrics({"accuracy": 1.4})


def test_validate_task_input_rejects_dangerous_strings():
    with pytest.raises(ValidationError):
        validate_task_input("<script>alert('x')</script>")


def test_safe_validate_returns_tuple_instead_of_raising():
    valid, message = safe_validate(validate_task_input, None)

    assert valid is False
    assert "cannot be None" in message


def test_confidence_interval_handles_single_value():
    mean, lower, upper = StatisticalAnalyzer.confidence_interval([0.7])

    assert mean == pytest.approx(0.7)
    assert lower == pytest.approx(0.7)
    assert upper == pytest.approx(0.7)


def test_compare_agents_returns_effect_size_and_sample_counts():
    comparison = StatisticalAnalyzer.compare_agents(
        [{"score": 0.9}, {"score": 0.85}, {"score": 0.88}],
        [{"score": 0.6}, {"score": 0.55}, {"score": 0.58}],
    )

    assert comparison["n1"] == 3
    assert comparison["n2"] == 3
    assert "effect_size" in comparison
    assert "t_test" in comparison
