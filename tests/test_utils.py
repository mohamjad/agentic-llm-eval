"""Tests for config and validation helpers."""

import pytest

from src.rl.policy import AgentParameters
from src.utils.config import Config
from src.utils.validation import (
    ValidationError,
    safe_validate,
    validate_agent_parameters,
    validate_config,
    validate_metrics,
    validate_score,
    validate_task_input,
)


def test_config_load_from_dict_merges_nested_values():
    """Config merging should preserve defaults while applying overrides."""
    config = Config()
    config.load_from_dict(
        {
            "evaluator": {
                "success_threshold": 0.9
            }
        }
    )

    assert config.get("evaluator.success_threshold") == 0.9
    assert config.get("metrics.efficiency.max_reasonable_steps") == 20


def test_config_load_from_env_converts_types(monkeypatch):
    """Environment values should map into nested config with basic type coercion."""
    monkeypatch.setenv("AGENTIC_EVALUATOR__SUCCESS_THRESHOLD", "0.85")
    monkeypatch.setenv("AGENTIC_LOGGING__ENABLED", "true")

    config = Config()
    config.load_from_env()

    assert config.get("evaluator.success_threshold") == 0.85
    assert config.get("logging.enabled") is True


def test_validate_agent_parameters_accepts_valid_bounds():
    """Well-formed agent parameters should pass validation."""
    params = AgentParameters(
        context_length=1200,
        temperature=0.8,
        max_steps=12,
        tool_usage_threshold=0.4,
        reasoning_depth=3,
    )

    validate_agent_parameters(params)


def test_validate_metrics_rejects_out_of_range_values():
    """Metrics outside the normalized range should fail validation."""
    with pytest.raises(ValidationError):
        validate_metrics({"accuracy": 1.5})


def test_validate_task_input_blocks_dangerous_patterns():
    """Suspicious executable content should be rejected."""
    with pytest.raises(ValidationError):
        validate_task_input("please run <script>alert(1)</script>")


def test_validate_config_requires_expected_keys():
    """Explicit required keys should be enforced."""
    with pytest.raises(ValidationError):
        validate_config({"logging": {}}, required_keys=["evaluator"])


def test_safe_validate_wraps_validation_errors():
    """safe_validate should convert validation exceptions into a tuple response."""
    is_valid, error = safe_validate(validate_score, 1.4, "overall_score")

    assert is_valid is False
    assert "overall_score" in error
