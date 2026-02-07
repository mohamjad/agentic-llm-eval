"""Example demonstrating advanced features:
- Neural network-based RL
- Semantic metrics
- Bayesian optimization
- Statistical analysis
"""

from src.evaluators import AgentEvaluator
from src.benchmarks import TaskBenchmark
from src.agents.base import BaseAgent
from src.rl import RLTrainer, BayesianParameterOptimizer
from src.metrics import SemanticMetric
from src.utils import StatisticalAnalyzer
from src.rl.policy import AgentParameters


class SimpleAgent(BaseAgent):
    """Simple agent for demonstration"""
    
    def __init__(self, params: dict = None):
        self.params = params or {}
    
    def execute(self, task, trace=None):
        # Simulate agent execution
        return {"result": f"Completed task: {getattr(task, 'id', 'unknown')}"}
    
    def set_parameters(self, params: dict):
        self.params = params


def example_neural_rl():
    """Example: Training with neural network-based RL"""
    print("=" * 60)
    print("Neural Network RL Example")
    print("=" * 60)
    
    try:
        agent = SimpleAgent()
        evaluator = AgentEvaluator()
        benchmark = TaskBenchmark.load("standard_tasks")
        
        # Use neural network-based RL
        trainer = RLTrainer(
            agent=agent,
            evaluator=evaluator,
            benchmark=benchmark,
            use_neural_network=True,  # Enable neural network
            device="cpu"  # Use CPU (or "cuda" for GPU)
        )
        
        # Train for a few episodes
        results = trainer.train(episodes=5, tasks_per_episode=3)
        
        print(f"Best parameters: {results['final_params']}")
        print(f"Final metrics: {results['final_metrics']}")
        print(f"Training statistics: {trainer.policy_network.get_statistics()}")
        
    except ImportError as e:
        print(f"Neural network RL not available: {e}")
        print("Install PyTorch: pip install torch>=2.0.0")


def example_semantic_metrics():
    """Example: Using semantic similarity metrics"""
    print("\n" + "=" * 60)
    print("Semantic Metrics Example")
    print("=" * 60)
    
    try:
        semantic_metric = SemanticMetric(model_name="all-MiniLM-L6-v2")
        
        # Example texts
        expected = "The agent successfully completed the task"
        actual = "The agent finished the task successfully"
        
        # Compute semantic accuracy
        accuracy = semantic_metric.semantic_accuracy(expected, actual)
        print(f"Semantic accuracy: {accuracy:.3f}")
        
        # Compute coherence
        texts = [
            "First, I need to understand the problem",
            "Then, I'll analyze the requirements",
            "Finally, I'll implement the solution"
        ]
        coherence = semantic_metric.semantic_coherence(texts)
        print(f"Semantic coherence: {coherence:.3f}")
        
    except ImportError as e:
        print(f"Semantic metrics not available: {e}")
        print("Install: pip install sentence-transformers>=2.2.0")


def example_bayesian_optimization():
    """Example: Bayesian optimization for hyperparameter tuning"""
    print("\n" + "=" * 60)
    print("Bayesian Optimization Example")
    print("=" * 60)
    
    try:
        agent = SimpleAgent()
        evaluator = AgentEvaluator()
        benchmark = TaskBenchmark.load("standard_tasks")
        
        optimizer = BayesianParameterOptimizer(
            n_calls=20,  # Number of optimization iterations
            n_initial_points=5  # Random exploration before optimization
        )
        
        # Run optimization
        results = optimizer.optimize(
            evaluator=evaluator,
            agent=agent,
            benchmark=benchmark
        )
        
        print(f"Best parameters: {results['best_params']}")
        print(f"Best score: {results['best_score']:.3f}")
        print(f"Optimization iterations: {results['n_iterations']}")
        
    except ImportError as e:
        print(f"Bayesian optimization not available: {e}")
        print("Install: pip install scikit-optimize>=0.9.0")


def example_statistical_analysis():
    """Example: Statistical comparison of agents"""
    print("\n" + "=" * 60)
    print("Statistical Analysis Example")
    print("=" * 60)
    
    try:
        # Simulate results from two agents
        agent1_results = [
            {"score": 0.85}, {"score": 0.82}, {"score": 0.88},
            {"score": 0.79}, {"score": 0.86}
        ]
        agent2_results = [
            {"score": 0.78}, {"score": 0.75}, {"score": 0.81},
            {"score": 0.73}, {"score": 0.79}
        ]
        
        # Compare agents
        comparison = StatisticalAnalyzer.compare_agents(
            agent1_results,
            agent2_results,
            metric_name="score"
        )
        
        print("Agent Comparison:")
        print(f"  Agent 1 mean: {comparison['agent1_ci']['mean']:.3f}")
        print(f"  Agent 2 mean: {comparison['agent2_ci']['mean']:.3f}")
        print(f"  p-value: {comparison['t_test']['pvalue']:.4f}")
        print(f"  Significant: {comparison['t_test']['significant']}")
        print(f"  Effect size (Cohen's d): {comparison['effect_size']['cohens_d']:.3f}")
        print(f"  Effect magnitude: {comparison['effect_size']['magnitude']}")
        
        # Confidence intervals
        print("\nConfidence Intervals (95%):")
        print(f"  Agent 1: [{comparison['agent1_ci']['lower']:.3f}, {comparison['agent1_ci']['upper']:.3f}]")
        print(f"  Agent 2: [{comparison['agent2_ci']['lower']:.3f}, {comparison['agent2_ci']['upper']:.3f}]")
        
    except ImportError as e:
        print(f"Statistical analysis not available: {e}")
        print("Install: pip install scipy>=1.11.0")


if __name__ == "__main__":
    print("Advanced Features Demonstration\n")
    
    example_neural_rl()
    example_semantic_metrics()
    example_bayesian_optimization()
    example_statistical_analysis()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
