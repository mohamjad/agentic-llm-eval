"""policy network for RL-based agent behavior

Enhanced RL implementation with:
- Adaptive learning rates
- Momentum-based updates
- Multi-objective optimization
- Experience replay buffer
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
try:
    import numpy as np
except ImportError:
    # Fallback if numpy not available
    import math
    class np:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0.0
from collections import deque


@dataclass
class AgentParameters:
    """tunable agent parameters with validation"""
    context_length: int = 1000
    temperature: float = 0.7
    max_steps: int = 10
    tool_usage_threshold: float = 0.5
    reasoning_depth: int = 2
    
    def __post_init__(self):
        """Validate parameters on initialization"""
        self.context_length = max(100, min(5000, self.context_length))
        self.temperature = max(0.1, min(2.0, self.temperature))
        self.max_steps = max(1, min(50, self.max_steps))
        self.tool_usage_threshold = max(0.0, min(1.0, self.tool_usage_threshold))
        self.reasoning_depth = max(1, min(10, self.reasoning_depth))
    
    def to_dict(self) -> Dict[str, Any]:
        """convert to dict"""
        return {
            "context_length": self.context_length,
            "temperature": self.temperature,
            "max_steps": self.max_steps,
            "tool_usage_threshold": self.tool_usage_threshold,
            "reasoning_depth": self.reasoning_depth
        }
    
    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "AgentParameters":
        """create from dict with validation"""
        return cls(**params)


@dataclass
class Experience:
    """Experience tuple for replay buffer"""
    state: Dict[str, float]  # metrics before
    action: Dict[str, float]  # parameter adjustments
    reward: float  # improvement in metrics
    next_state: Dict[str, float]  # metrics after
    timestamp: float = field(default_factory=lambda: __import__('time').time())


class PolicyNetwork:
    """Enhanced policy network with adaptive learning and experience replay"""
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        decay_rate: float = 0.95,
        replay_buffer_size: int = 100
    ):
        """
        Initialize policy network
        
        Args:
            learning_rate: Base learning rate for weight updates
            momentum: Momentum factor for gradient updates
            decay_rate: Learning rate decay factor
            replay_buffer_size: Size of experience replay buffer
        """
        self.base_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.momentum = momentum
        self.decay_rate = decay_rate
        
        # Enhanced weight matrix with more sophisticated mappings
        self.weights = {
            "context_length": {
                "accuracy": 0.1, "efficiency": -0.05,
                "coherence": 0.02, "adaptability": 0.01
            },
            "temperature": {
                "coherence": 0.02, "adaptability": 0.03,
                "accuracy": -0.01, "safety_score": -0.01
            },
            "max_steps": {
                "efficiency": -0.1, "accuracy": 0.05,
                "tool_usage": 0.02, "adaptability": 0.01
            },
            "tool_usage_threshold": {
                "tool_usage": 0.15, "efficiency": -0.08,
                "accuracy": 0.03, "safety_score": 0.02
            },
            "reasoning_depth": {
                "coherence": 0.1, "adaptability": 0.05,
                "accuracy": 0.02, "efficiency": -0.03
            }
        }
        
        # Momentum buffers for each parameter
        self.velocity = {
            param: {metric: 0.0 for metric in metrics.keys()}
            for param, metrics in self.weights.items()
        }
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        
        # Performance tracking
        self.update_count = 0
        self.best_performance = 0.0
    
    def get_parameter_adjustment(
        self,
        current_params: AgentParameters,
        metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate parameter adjustments based on current metrics using multi-objective optimization
        
        Args:
            current_params: Current agent parameters
            metrics: Current performance metrics
            
        Returns:
            Dictionary of parameter adjustments
        """
        adjustments = {}
        
        for param_name, metric_weights in self.weights.items():
            adjustment = 0.0
            
            # Multi-objective: consider all relevant metrics
            for metric_name, weight in metric_weights.items():
                if metric_name in metrics:
                    metric_value = metrics[metric_name]
                    # Normalize metric value and calculate gradient
                    # Lower metrics need more adjustment
                    gradient = weight * (1.0 - metric_value)
                    adjustment += gradient
            
            # Apply momentum for smoother updates
            if param_name in self.velocity:
                # Average momentum across metrics
                avg_velocity = np.mean([
                    self.velocity[param_name].get(m, 0.0)
                    for m in metric_weights.keys()
                ])
                adjustment = self.momentum * avg_velocity + (1 - self.momentum) * adjustment
            
            adjustments[param_name] = adjustment * self.current_learning_rate
        
        return adjustments
    
    def update_weights(
        self,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
        adjustments: Dict[str, float]
    ):
        """
        Update policy weights using experience replay and adaptive learning
        
        Args:
            metrics_before: Metrics before parameter adjustment
            metrics_after: Metrics after parameter adjustment
            adjustments: Parameter adjustments that were applied
        """
        # Calculate reward (overall improvement)
        improvement = self._calculate_improvement(metrics_before, metrics_after)
        
        # Store experience in replay buffer
        experience = Experience(
            state=metrics_before.copy(),
            action=adjustments.copy(),
            reward=improvement,
            next_state=metrics_after.copy()
        )
        self.replay_buffer.append(experience)
        
        # Update weights if improvement occurred
        if improvement > 0:
            # Reinforce successful adjustments
            for param_name, adj_value in adjustments.items():
                if param_name in self.weights:
                    for metric_name in self.weights[param_name]:
                        metric_improvement = (
                            metrics_after.get(metric_name, 0) -
                            metrics_before.get(metric_name, 0)
                        )
                        
                        if metric_improvement > 0:
                            # Update weight with momentum
                            gradient = adj_value * metric_improvement * 0.01
                            self.velocity[param_name][metric_name] = (
                                self.momentum * self.velocity[param_name].get(metric_name, 0) +
                                (1 - self.momentum) * gradient
                            )
                            self.weights[param_name][metric_name] += self.velocity[param_name][metric_name]
        
        # Update learning rate with decay
        self.update_count += 1
        if self.update_count % 10 == 0:
            self.current_learning_rate *= self.decay_rate
            self.current_learning_rate = max(0.001, self.current_learning_rate)
        
        # Track best performance
        current_performance = sum(metrics_after.values()) / len(metrics_after) if metrics_after else 0.0
        if current_performance > self.best_performance:
            self.best_performance = current_performance
    
    def _calculate_improvement(
        self,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float]
    ) -> float:
        """Calculate overall improvement across all metrics"""
        common_metrics = set(metrics_before.keys()) & set(metrics_after.keys())
        if not common_metrics:
            return 0.0
        
        improvements = [
            metrics_after[k] - metrics_before[k]
            for k in common_metrics
        ]
        
        # Weighted average improvement
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def sample_experiences(self, batch_size: int = 10) -> List[Experience]:
        """Sample experiences from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return list(self.replay_buffer)
        
        # Sample recent experiences (prioritize recent)
        recent_size = min(batch_size, len(self.replay_buffer))
        return list(self.replay_buffer)[-recent_size:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            "update_count": self.update_count,
            "learning_rate": self.current_learning_rate,
            "best_performance": self.best_performance,
            "replay_buffer_size": len(self.replay_buffer)
        }


class AgentPolicy:
    """Enhanced wrapper that applies policy to agent parameters with validation"""
    
    def __init__(self, policy_network: PolicyNetwork):
        self.policy = policy_network
    
    def adjust_parameters(
        self,
        current_params: AgentParameters,
        metrics: Dict[str, float]
    ) -> AgentParameters:
        """
        Adjust parameters based on current performance with bounds checking
        
        Args:
            current_params: Current agent parameters
            metrics: Current performance metrics
            
        Returns:
            New parameters with adjustments applied
        """
        adjustments = self.policy.get_parameter_adjustment(current_params, metrics)
        
        # Apply adjustments with bounds and validation
        new_params = AgentParameters(
            context_length=int(
                current_params.context_length + adjustments.get("context_length", 0) * 100
            ),
            temperature=current_params.temperature + adjustments.get("temperature", 0),
            max_steps=int(
                current_params.max_steps + adjustments.get("max_steps", 0) * 2
            ),
            tool_usage_threshold=(
                current_params.tool_usage_threshold +
                adjustments.get("tool_usage_threshold", 0)
            ),
            reasoning_depth=int(
                current_params.reasoning_depth + adjustments.get("reasoning_depth", 0)
            )
        )
        
        return new_params
