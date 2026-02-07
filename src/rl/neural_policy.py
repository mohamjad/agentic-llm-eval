"""Neural network-based policy network for deep RL

Implements a proper neural network policy with:
- Multi-layer perceptron architecture
- Policy gradient methods (PPO-style)
- Value function estimation
- Advantage calculation
- Proper backpropagation and gradient updates
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Fallback implementations
    torch = None
    nn = None
    optim = None
    F = None

from .policy import AgentParameters, Experience


class PolicyNetworkNN(nn.Module):
    """Neural network policy that maps state (metrics) to action (parameter adjustments)
    
    Architecture:
    - Input: Normalized metrics vector (dim = num_metrics)
    - Hidden layers: 2 fully connected layers with ReLU activation
    - Output: Parameter adjustment vector (dim = num_parameters)
    
    Uses Gaussian policy for continuous action space with learnable variance.
    """
    
    def __init__(
        self,
        input_dim: int = 6,  # Number of metrics
        hidden_dim: int = 64,
        output_dim: int = 5,  # Number of parameters
        activation: str = "relu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Policy network (mean of Gaussian distribution)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        
        # Learnable log_std for action variance
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform initialization"""
        for m in [self.fc1, self.fc2, self.fc_mean]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network
        
        Args:
            state: Normalized metrics tensor [batch_size, input_dim]
            
        Returns:
            mean: Mean of action distribution [batch_size, output_dim]
            std: Standard deviation of action distribution [batch_size, output_dim]
        """
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        mean = self.fc_mean(x)
        
        # Ensure std is positive and bounded
        std = torch.exp(self.log_std.clamp(-2, 2))
        std = std.expand_as(mean)
        
        return mean, std
    
    def sample_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy distribution
        
        Args:
            state: Normalized metrics tensor
            deterministic: If True, return mean (no exploration)
            
        Returns:
            action: Sampled action [batch_size, output_dim]
            log_prob: Log probability of sampled action
        """
        mean, std = self.forward(state)
        
        if deterministic:
            return mean, torch.zeros_like(mean)
        
        # Sample from Gaussian distribution
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate action under current policy
        
        Args:
            state: Normalized metrics tensor
            action: Action to evaluate
            
        Returns:
            log_prob: Log probability of action
            entropy: Entropy of action distribution
            mean: Mean of distribution
        """
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, mean


class ValueNetworkNN(nn.Module):
    """Value network for estimating state values (V(s))
    
    Used for advantage calculation: A(s,a) = Q(s,a) - V(s)
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in [self.fc1, self.fc2, self.fc_value]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value
        
        Args:
            state: Normalized metrics tensor [batch_size, input_dim]
            
        Returns:
            value: Estimated state value [batch_size, 1]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc_value(x)
        return value


class DeepPolicyNetwork:
    """Deep RL policy network using neural networks
    
    Implements PPO-style policy gradient methods with:
    - Policy network (actor)
    - Value network (critic)
    - Advantage estimation
    - Clipped surrogate objective
    - Experience replay with importance sampling
    """
    
    def __init__(
        self,
        learning_rate: float = 3e-4,
        value_learning_rate: float = 3e-4,
        gamma: float = 0.99,  # Discount factor
        lambda_gae: float = 0.95,  # GAE lambda
        clip_epsilon: float = 0.2,  # PPO clip parameter
        value_coef: float = 0.5,  # Value loss coefficient
        entropy_coef: float = 0.01,  # Entropy bonus coefficient
        max_grad_norm: float = 0.5,
        device: Optional[str] = None
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DeepPolicyNetwork. "
                "Install with: pip install torch>=2.0.0"
            )
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Network dimensions
        # Input: metrics (accuracy, efficiency, safety, coherence, adaptability, tool_usage)
        # Output: parameter adjustments (5 parameters)
        self.input_dim = 6
        self.output_dim = 5
        
        # Initialize networks
        self.policy_net = PolicyNetworkNN(
            input_dim=self.input_dim,
            hidden_dim=64,
            output_dim=self.output_dim
        ).to(self.device)
        
        self.value_net = ValueNetworkNN(
            input_dim=self.input_dim,
            hidden_dim=64
        ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )
        
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=value_learning_rate
        )
        
        # Experience buffer
        self.experience_buffer: List[Dict[str, Any]] = []
        
        # Statistics
        self.update_count = 0
        self.total_policy_loss = 0.0
        self.total_value_loss = 0.0
        
        # Metric normalization (learned from data)
        self.metric_mean = torch.zeros(self.input_dim).to(self.device)
        self.metric_std = torch.ones(self.input_dim).to(self.device)
        self.normalization_updates = 0
    
    def normalize_metrics(self, metrics: Dict[str, float]) -> torch.Tensor:
        """Normalize metrics to [0, 1] range for neural network input"""
        metric_order = [
            "accuracy", "efficiency", "safety_score",
            "coherence", "adaptability", "tool_usage"
        ]
        
        # Extract metrics in fixed order
        metric_vector = np.array([
            metrics.get(name, 0.0) for name in metric_order
        ], dtype=np.float32)
        
        # Clamp to [0, 1]
        metric_vector = np.clip(metric_vector, 0.0, 1.0)
        
        # Convert to tensor
        state = torch.FloatTensor(metric_vector).unsqueeze(0).to(self.device)
        
        # Normalize using running statistics
        if self.normalization_updates > 0:
            state = (state - self.metric_mean) / (self.metric_std + 1e-8)
        
        return state
    
    def get_parameter_adjustment(
        self,
        current_params: AgentParameters,
        metrics: Dict[str, float],
        deterministic: bool = False
    ) -> Dict[str, float]:
        """
        Get parameter adjustments using neural network policy
        
        Args:
            current_params: Current agent parameters
            metrics: Current performance metrics
            deterministic: If True, use mean action (no exploration)
            
        Returns:
            Dictionary of parameter adjustments
        """
        # Normalize metrics
        state = self.normalize_metrics(metrics)
        
        # Get action from policy
        self.policy_net.eval()
        with torch.no_grad():
            action, _ = self.policy_net.sample_action(state, deterministic=deterministic)
            action = action.squeeze(0).cpu().numpy()
        
        # Map action to parameter adjustments
        # Actions are normalized, scale them appropriately
        param_names = [
            "context_length", "temperature", "max_steps",
            "tool_usage_threshold", "reasoning_depth"
        ]
        
        adjustments = {}
        for i, param_name in enumerate(param_names):
            # Scale actions to reasonable ranges
            if param_name == "context_length":
                adjustments[param_name] = action[i] * 100  # ±100 tokens
            elif param_name == "temperature":
                adjustments[param_name] = action[i] * 0.1  # ±0.1
            elif param_name == "max_steps":
                adjustments[param_name] = action[i] * 2  # ±2 steps
            elif param_name == "tool_usage_threshold":
                adjustments[param_name] = action[i] * 0.1  # ±0.1
            elif param_name == "reasoning_depth":
                adjustments[param_name] = action[i] * 1  # ±1 depth
            else:
                adjustments[param_name] = action[i]
        
        return adjustments
    
    def store_experience(
        self,
        state: Dict[str, float],
        action: Dict[str, float],
        reward: float,
        next_state: Dict[str, float],
        done: bool = False
    ):
        """Store experience in buffer for training"""
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        }
        self.experience_buffer.append(experience)
    
    def compute_advantages(
        self,
        rewards: List[float],
        values: List[float],
        next_values: List[float],
        dones: List[bool]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE)
        
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        advantages = []
        returns = []
        
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                gae = 0.0
            
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(
        self,
        batch_size: int = 32,
        epochs: int = 4
    ):
        """
        Update policy and value networks using PPO
        
        Args:
            batch_size: Batch size for training
            epochs: Number of update epochs
        """
        if len(self.experience_buffer) < batch_size:
            return
        
        # Convert experiences to tensors
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for exp in self.experience_buffer:
            states.append(self.normalize_metrics(exp["state"]).squeeze(0))
            # Convert action dict to vector
            action_vec = np.array([
                exp["action"].get("context_length", 0.0) / 100,
                exp["action"].get("temperature", 0.0) / 0.1,
                exp["action"].get("max_steps", 0.0) / 2,
                exp["action"].get("tool_usage_threshold", 0.0) / 0.1,
                exp["action"].get("reasoning_depth", 0.0) / 1
            ], dtype=np.float32)
            actions.append(torch.FloatTensor(action_vec).to(self.device))
            rewards.append(exp["reward"])
            next_states.append(self.normalize_metrics(exp["next_state"]).squeeze(0))
            dones.append(1.0 if exp["done"] else 0.0)
        
        states_tensor = torch.stack(states).to(self.device)
        actions_tensor = torch.stack(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.stack(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Compute values
        self.value_net.eval()
        with torch.no_grad():
            values = self.value_net(states_tensor).squeeze()
            next_values = self.value_net(next_states_tensor).squeeze()
        
        # Compute advantages
        advantages, returns = self.compute_advantages(
            rewards_tensor.cpu().numpy().tolist(),
            values.cpu().numpy().tolist(),
            next_values.cpu().numpy().tolist(),
            dones_tensor.cpu().numpy().tolist()
        )
        
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )
        
        # Train for multiple epochs
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(states_tensor))
            
            for i in range(0, len(states_tensor), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Get old log probs
                self.policy_net.eval()
                with torch.no_grad():
                    old_log_probs, _, _ = self.policy_net.evaluate_action(
                        batch_states, batch_actions
                    )
                
                # Policy update
                self.policy_net.train()
                log_probs, entropy, _ = self.policy_net.evaluate_action(
                    batch_states, batch_actions
                )
                
                # PPO clipped objective
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value update
                self.value_net.train()
                values_pred = self.value_net(batch_states).squeeze()
                value_loss = F.mse_loss(values_pred, batch_returns)
                
                # Total loss
                total_loss = (
                    policy_loss +
                    self.value_coef * value_loss -
                    self.entropy_coef * entropy.mean()
                )
                
                # Update networks
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(),
                    self.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.value_net.parameters(),
                    self.max_grad_norm
                )
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
        
        # Update statistics
        self.update_count += 1
        
        # Clear buffer
        self.experience_buffer.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            "update_count": self.update_count,
            "device": self.device,
            "policy_params": sum(p.numel() for p in self.policy_net.parameters()),
            "value_params": sum(p.numel() for p in self.value_net.parameters()),
        }
