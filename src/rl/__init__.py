"""reinforcement learning components for agent fine-tuning"""

from .trainer import RLTrainer
from .policy import PolicyNetwork, AgentPolicy, AgentParameters

# Optional advanced components
try:
    from .neural_policy import DeepPolicyNetwork
    from .bayesian_optimizer import BayesianParameterOptimizer
    __all__ = [
        "RLTrainer", "PolicyNetwork", "AgentPolicy", "AgentParameters",
        "DeepPolicyNetwork", "BayesianParameterOptimizer"
    ]
except ImportError:
    __all__ = ["RLTrainer", "PolicyNetwork", "AgentPolicy", "AgentParameters"]
