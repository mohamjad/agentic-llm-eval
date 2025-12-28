"""policy network for RL-based agent behavior"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class AgentParameters:
    """tunable agent parameters"""
    context_length: int = 1000
    temperature: float = 0.7
    max_steps: int = 10
    tool_usage_threshold: float = 0.5
    reasoning_depth: int = 2
    
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
        """create from dict"""
        return cls(**params)


class PolicyNetwork:
    """simple policy network that maps state to parameter adjustments"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        # simple linear policy: state -> parameter deltas
        self.weights = {
            "context_length": {"accuracy": 0.1, "efficiency": -0.05},
            "temperature": {"coherence": 0.02, "adaptability": 0.03},
            "max_steps": {"efficiency": -0.1, "accuracy": 0.05},
            "tool_usage_threshold": {"tool_usage": 0.15, "efficiency": -0.08},
            "reasoning_depth": {"coherence": 0.1, "adaptability": 0.05}
        }
    
    def get_parameter_adjustment(
        self,
        current_params: AgentParameters,
        metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        calculate parameter adjustments based on current metrics
        
        metrics should include: accuracy, efficiency, coherence, adaptability, tool_usage
        """
        adjustments = {}
        
        for param_name, metric_weights in self.weights.items():
            adjustment = 0.0
            for metric_name, weight in metric_weights.items():
                if metric_name in metrics:
                    # if metric is low, adjust parameter
                    metric_value = metrics[metric_name]
                    adjustment += weight * (1.0 - metric_value)
            
            adjustments[param_name] = adjustment * self.learning_rate
        
        return adjustments
    
    def update_weights(
        self,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
        adjustments: Dict[str, float]
    ):
        """update policy weights based on improvement"""
        # simple update: if metrics improved, reinforce the adjustments
        improvement = sum(metrics_after.get(k, 0) - metrics_before.get(k, 0) 
                         for k in set(metrics_after.keys()) & set(metrics_before.keys()))
        
        if improvement > 0:
            # reinforce successful adjustments
            for param_name, adj_value in adjustments.items():
                if param_name in self.weights:
                    for metric_name in self.weights[param_name]:
                        if metrics_after.get(metric_name, 0) > metrics_before.get(metric_name, 0):
                            self.weights[param_name][metric_name] += adj_value * 0.01


class AgentPolicy:
    """wrapper that applies policy to agent parameters"""
    
    def __init__(self, policy_network: PolicyNetwork):
        self.policy = policy_network
    
    def adjust_parameters(
        self,
        current_params: AgentParameters,
        metrics: Dict[str, float]
    ) -> AgentParameters:
        """adjust parameters based on current performance"""
        adjustments = self.policy.get_parameter_adjustment(current_params, metrics)
        
        # apply adjustments with bounds
        new_params = AgentParameters(
            context_length=max(100, min(5000, 
                int(current_params.context_length + adjustments.get("context_length", 0) * 100))),
            temperature=max(0.1, min(2.0, 
                current_params.temperature + adjustments.get("temperature", 0))),
            max_steps=max(1, min(50, 
                int(current_params.max_steps + adjustments.get("max_steps", 0) * 2))),
            tool_usage_threshold=max(0.0, min(1.0, 
                current_params.tool_usage_threshold + adjustments.get("tool_usage_threshold", 0))),
            reasoning_depth=max(1, min(10, 
                int(current_params.reasoning_depth + adjustments.get("reasoning_depth", 0))))
        )
        
        return new_params
