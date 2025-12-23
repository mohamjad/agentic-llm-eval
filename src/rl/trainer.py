"""RL trainer for fine-tuning agent behavior"""

from typing import Dict, Any, List, Optional, Callable
from ..agents.base import BaseAgent
from ..evaluators import AgentEvaluator
from ..benchmarks import TaskBenchmark
from .policy import PolicyNetwork, AgentPolicy, AgentParameters


class RLTrainer:
    """train agent using reinforcement learning based on evaluation feedback"""
    
    def __init__(
        self,
        agent: BaseAgent,
        evaluator: AgentEvaluator,
        benchmark: TaskBenchmark,
        initial_params: Optional[AgentParameters] = None,
        learning_rate: float = 0.01
    ):
        self.agent = agent
        self.evaluator = evaluator
        self.benchmark = benchmark
        self.policy_network = PolicyNetwork(learning_rate=learning_rate)
        self.agent_policy = AgentPolicy(self.policy_network)
        self.current_params = initial_params or AgentParameters()
        self.training_history: List[Dict[str, Any]] = []
    
    def train(
        self,
        episodes: int = 10,
        tasks_per_episode: Optional[int] = None,
        update_policy: bool = True
    ) -> Dict[str, Any]:
        """
        train agent over multiple episodes
        
        episodes: number of training episodes
        tasks_per_episode: how many tasks per episode (None = all)
        update_policy: whether to update policy weights
        """
        all_results = []
        
        for episode in range(episodes):
            # select tasks for this episode
            tasks = self.benchmark.tasks
            if tasks_per_episode:
                import random
                tasks = random.sample(tasks, min(tasks_per_episode, len(tasks)))
            
            # evaluate with current parameters
            metrics_before = self._get_average_metrics(tasks)
            
            # adjust parameters based on performance
            self.current_params = self.agent_policy.adjust_parameters(
                self.current_params,
                metrics_before
            )
            
            # apply parameters to agent (if agent supports it)
            self._apply_parameters_to_agent()
            
            # evaluate again with new parameters
            metrics_after = self._get_average_metrics(tasks)
            
            # update policy if metrics improved
            if update_policy:
                adjustments = self.policy_network.get_parameter_adjustment(
                    self.current_params,
                    metrics_before
                )
                self.policy_network.update_weights(
                    metrics_before,
                    metrics_after,
                    adjustments
                )
            
            # record episode
            episode_data = {
                "episode": episode,
                "params": self.current_params.to_dict(),
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
                "improvement": sum(metrics_after.get(k, 0) - metrics_before.get(k, 0)
                                 for k in set(metrics_after.keys()) & set(metrics_before.keys()))
            }
            self.training_history.append(episode_data)
            all_results.append(episode_data)
            
            print(f"episode {episode}: improvement = {episode_data['improvement']:.4f}")
        
        return {
            "final_params": self.current_params.to_dict(),
            "final_metrics": metrics_after,
            "history": all_results
        }
    
    def _get_average_metrics(self, tasks: List[Any]) -> Dict[str, float]:
        """get average metrics across tasks"""
        results = self.evaluator.evaluate_batch(self.agent, tasks)
        
        # aggregate metrics
        all_metrics = {}
        metric_counts = {}
        
        for result in results:
            for metric_name, metric_value in result.metrics.items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = 0.0
                        metric_counts[metric_name] = 0
                    all_metrics[metric_name] += metric_value
                    metric_counts[metric_name] += 1
        
        # average
        avg_metrics = {
            name: value / metric_counts[name]
            for name, value in all_metrics.items()
            if metric_counts[name] > 0
        }
        
        return avg_metrics
    
    def _apply_parameters_to_agent(self):
        """apply current parameters to agent if it supports it"""
        if hasattr(self.agent, "set_parameters"):
            self.agent.set_parameters(self.current_params.to_dict())
    
    def get_best_parameters(self) -> AgentParameters:
        """get parameters from best performing episode"""
        if not self.training_history:
            return self.current_params
        
        best_episode = max(self.training_history, key=lambda x: x["improvement"])
        return AgentParameters.from_dict(best_episode["params"])
    
    def plot_training_progress(self):
        """plot training progress (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            episodes = [h["episode"] for h in self.training_history]
            improvements = [h["improvement"] for h in self.training_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(episodes, improvements, marker='o')
            plt.xlabel("episode")
            plt.ylabel("improvement")
            plt.title("training progress")
            plt.grid(True)
            plt.show()
        except ImportError:
            print("matplotlib not available for plotting")
