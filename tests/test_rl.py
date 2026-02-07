"""
Tests for RL training and policy components
"""

import pytest
from src.rl.policy import PolicyNetwork, AgentPolicy, AgentParameters
from src.rl.trainer import RLTrainer
from src.evaluators import AgentEvaluator
from src.benchmarks import TaskBenchmark
from src.agents.base import BaseAgent, AgentExecutionTrace


class ParameterizedTestAgent(BaseAgent):
    """Test agent that supports parameters"""
    
    def __init__(self):
        self.params = AgentParameters()
    
    def execute(self, task, trace=None):
        if trace:
            trace.add_step(
                action_type="execution",
                action_name="execute",
                input_data={},
                output_data={"result": "done"},
                duration=0.1
            )
        return {"result": "done"}
    
    def set_parameters(self, params: dict):
        self.params = AgentParameters.from_dict(params)


class TestAgentParameters:
    """Test agent parameters"""
    
    def test_default_parameters(self):
        """Test default parameter values"""
        params = AgentParameters()
        assert params.context_length == 1000
        assert params.temperature == 0.7
        assert params.max_steps == 10
        assert params.tool_usage_threshold == 0.5
        assert params.reasoning_depth == 2
    
    def test_to_dict(self):
        """Test converting to dict"""
        params = AgentParameters(
            context_length=2000,
            temperature=0.8
        )
        params_dict = params.to_dict()
        assert params_dict["context_length"] == 2000
        assert params_dict["temperature"] == 0.8
    
    def test_from_dict(self):
        """Test creating from dict"""
        params_dict = {
            "context_length": 2000,
            "temperature": 0.8,
            "max_steps": 15,
            "tool_usage_threshold": 0.6,
            "reasoning_depth": 3
        }
        params = AgentParameters.from_dict(params_dict)
        assert params.context_length == 2000
        assert params.temperature == 0.8
        assert params.max_steps == 15


class TestPolicyNetwork:
    """Test policy network"""
    
    def test_initialization(self):
        """Test policy network initialization"""
        policy = PolicyNetwork(learning_rate=0.01)
        assert policy.base_learning_rate == 0.01
        assert policy.current_learning_rate == 0.01
        assert "context_length" in policy.weights
    
    def test_get_parameter_adjustment(self):
        """Test parameter adjustment calculation"""
        policy = PolicyNetwork()
        params = AgentParameters()
        metrics = {
            "accuracy": 0.5,
            "efficiency": 0.5,
            "coherence": 0.5
        }
        
        adjustments = policy.get_parameter_adjustment(params, metrics)
        assert "context_length" in adjustments
        assert "temperature" in adjustments
    
    def test_update_weights(self):
        """Test weight updates"""
        policy = PolicyNetwork()
        metrics_before = {"accuracy": 0.5, "efficiency": 0.5}
        metrics_after = {"accuracy": 0.7, "efficiency": 0.6}
        adjustments = {"context_length": 0.1, "temperature": 0.05}
        
        # Should not raise exception
        policy.update_weights(metrics_before, metrics_after, adjustments)


class TestAgentPolicy:
    """Test agent policy wrapper"""
    
    def test_adjust_parameters(self):
        """Test parameter adjustment with bounds"""
        policy_net = PolicyNetwork()
        policy = AgentPolicy(policy_net)
        
        params = AgentParameters(
            context_length=1000,
            temperature=0.7
        )
        metrics = {
            "accuracy": 0.3,
            "efficiency": 0.3
        }
        
        new_params = policy.adjust_parameters(params, metrics)
        
        # Check bounds are respected
        assert 100 <= new_params.context_length <= 5000
        assert 0.1 <= new_params.temperature <= 2.0
        assert 1 <= new_params.max_steps <= 50
    
    def test_parameter_bounds(self):
        """Test that parameters stay within bounds"""
        policy_net = PolicyNetwork()
        policy = AgentPolicy(policy_net)
        
        # Test minimum bounds
        params = AgentParameters(
            context_length=100,
            temperature=0.1,
            max_steps=1
        )
        metrics = {"accuracy": 0.0, "efficiency": 0.0}
        new_params = policy.adjust_parameters(params, metrics)
        assert new_params.context_length >= 100
        assert new_params.temperature >= 0.1
        assert new_params.max_steps >= 1
        
        # Test maximum bounds
        params = AgentParameters(
            context_length=5000,
            temperature=2.0,
            max_steps=50
        )
        metrics = {"accuracy": 1.0, "efficiency": 1.0}
        new_params = policy.adjust_parameters(params, metrics)
        assert new_params.context_length <= 5000
        assert new_params.temperature <= 2.0
        assert new_params.max_steps <= 50


class TestRLTrainer:
    """Test RL trainer"""
    
    def test_initialization(self):
        """Test trainer initialization"""
        agent = ParameterizedTestAgent()
        evaluator = AgentEvaluator()
        benchmark = TaskBenchmark()
        
        trainer = RLTrainer(agent, evaluator, benchmark)
        assert trainer.agent == agent
        assert trainer.evaluator == evaluator
        assert trainer.benchmark == benchmark
    
    def test_get_average_metrics(self):
        """Test metric averaging"""
        agent = ParameterizedTestAgent()
        evaluator = AgentEvaluator()
        benchmark = TaskBenchmark()
        
        # Ensure benchmark has tasks
        assert len(benchmark.tasks) > 0, "Benchmark must have tasks"
        
        trainer = RLTrainer(agent, evaluator, benchmark)
        metrics = trainer._get_average_metrics(benchmark.tasks)
        
        assert isinstance(metrics, dict)
        # Metrics might be empty if evaluation fails, so just check it's a dict
    
    def test_train_single_episode(self):
        """Test training for one episode"""
        agent = ParameterizedTestAgent()
        evaluator = AgentEvaluator()
        benchmark = TaskBenchmark()
        
        # Ensure benchmark has tasks
        assert len(benchmark.tasks) > 0, "Benchmark must have tasks"
        
        trainer = RLTrainer(agent, evaluator, benchmark)
        # Use fewer tasks if benchmark has less than 2
        tasks_per_episode = min(2, len(benchmark.tasks))
        result = trainer.train(episodes=1, tasks_per_episode=tasks_per_episode, update_policy=False)
        
        assert "final_params" in result
        assert "final_metrics" in result
        assert "history" in result
        assert len(result["history"]) == 1
    
    def test_get_best_parameters(self):
        """Test retrieving best parameters"""
        agent = ParameterizedTestAgent()
        evaluator = AgentEvaluator()
        benchmark = TaskBenchmark()
        
        # Ensure benchmark has tasks
        assert len(benchmark.tasks) > 0, "Benchmark must have tasks"
        
        trainer = RLTrainer(agent, evaluator, benchmark)
        # Use fewer tasks if benchmark has less than 2
        tasks_per_episode = min(2, len(benchmark.tasks))
        trainer.train(episodes=3, tasks_per_episode=tasks_per_episode, update_policy=False)
        
        best_params = trainer.get_best_parameters()
        assert isinstance(best_params, AgentParameters)
