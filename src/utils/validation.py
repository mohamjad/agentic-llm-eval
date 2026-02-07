"""Input validation utilities

Provides validation functions for:
- Agent parameters
- Task inputs
- Configuration values
- Metric values
"""

from typing import Any, Dict, List, Optional, Callable
from src.rl.policy import AgentParameters


class ValidationError(ValueError):
    """Custom validation error"""
    pass


def validate_agent_parameters(params: AgentParameters) -> None:
    """
    Validate agent parameters
    
    Args:
        params: AgentParameters instance
        
    Raises:
        ValidationError: If parameters are invalid
    """
    if not isinstance(params, AgentParameters):
        raise ValidationError(f"Expected AgentParameters, got {type(params)}")
    
    if not (100 <= params.context_length <= 5000):
        raise ValidationError(
            f"context_length must be between 100 and 5000, got {params.context_length}"
        )
    
    if not (0.1 <= params.temperature <= 2.0):
        raise ValidationError(
            f"temperature must be between 0.1 and 2.0, got {params.temperature}"
        )
    
    if not (1 <= params.max_steps <= 50):
        raise ValidationError(
            f"max_steps must be between 1 and 50, got {params.max_steps}"
        )
    
    if not (0.0 <= params.tool_usage_threshold <= 1.0):
        raise ValidationError(
            f"tool_usage_threshold must be between 0.0 and 1.0, "
            f"got {params.tool_usage_threshold}"
        )
    
    if not (1 <= params.reasoning_depth <= 10):
        raise ValidationError(
            f"reasoning_depth must be between 1 and 10, got {params.reasoning_depth}"
        )


def validate_metrics(metrics: Dict[str, Any]) -> None:
    """
    Validate metric values
    
    Args:
        metrics: Dictionary of metric values
        
    Raises:
        ValidationError: If metrics are invalid
    """
    if not isinstance(metrics, dict):
        raise ValidationError(f"Expected dict, got {type(metrics)}")
    
    # Check that numeric metrics are in valid range [0, 1]
    numeric_metrics = [
        "accuracy", "efficiency", "safety_score", "coherence",
        "adaptability", "tool_usage", "overall_score"
    ]
    
    for metric_name in numeric_metrics:
        if metric_name in metrics:
            value = metrics[metric_name]
            if isinstance(value, (int, float)):
                if not (0.0 <= value <= 1.0):
                    raise ValidationError(
                        f"{metric_name} must be between 0.0 and 1.0, got {value}"
                    )


def validate_task_input(task_input: Any) -> None:
    """
    Validate task input
    
    Args:
        task_input: Task input data
        
    Raises:
        ValidationError: If input is invalid
    """
    if task_input is None:
        raise ValidationError("Task input cannot be None")
    
    # Check for common security issues
    if isinstance(task_input, str):
        # Check for potential injection attempts
        dangerous_patterns = ["<script", "eval(", "exec(", "__import__"]
        for pattern in dangerous_patterns:
            if pattern in task_input.lower():
                raise ValidationError(
                    f"Potentially dangerous pattern detected in task input: {pattern}"
                )


def validate_config(config: Dict[str, Any], required_keys: Optional[List[str]] = None) -> None:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys (optional)
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError(f"Expected dict, got {type(config)}")
    
    if required_keys:
        for key in required_keys:
            if key not in config:
                raise ValidationError(f"Required configuration key missing: {key}")
    
    # Validate evaluator config if present
    if "evaluator" in config:
        eval_config = config["evaluator"]
        if isinstance(eval_config, dict):
            if "success_threshold" in eval_config:
                threshold = eval_config["success_threshold"]
                if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                    raise ValidationError(
                        f"success_threshold must be between 0 and 1, got {threshold}"
                    )


def validate_score(score: float, name: str = "score") -> None:
    """
    Validate a score value
    
    Args:
        score: Score value
        name: Name of the score (for error messages)
        
    Raises:
        ValidationError: If score is invalid
    """
    if not isinstance(score, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(score)}")
    
    if not (0.0 <= score <= 1.0):
        raise ValidationError(f"{name} must be between 0.0 and 1.0, got {score}")


def safe_validate(func: Callable, *args, **kwargs) -> tuple[bool, Optional[str]]:
    """
    Safely run validation function
    
    Args:
        func: Validation function to call
        *args: Positional arguments for validation function
        **kwargs: Keyword arguments for validation function
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        func(*args, **kwargs)
        return True, None
    except ValidationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error during validation: {str(e)}"
