"""example: showing how parameters affect agent behavior"""

from src.evaluators import AgentEvaluator
from src.benchmarks import TaskBenchmark
from src.agents.base import BaseAgent
from src.rl import AgentParameters


class TestAgent(BaseAgent):
    """simple agent for testing parameter impact"""
    
    def __init__(self, params: AgentParameters):
        self.params = params
    
    def execute(self, task, trace=None):
        """execute with current parameters"""
        # simulate how parameters affect behavior
        result = {
            "context_length": self.params.context_length,
            "temperature": self.params.temperature,
            "max_steps": self.params.max_steps,
            "result": f"processed with params: {self.params.temperature:.2f} temp"
        }
        return result


def test_parameter_impact():
    """test how different parameters affect evaluation"""
    print("=== Parameter Impact Analysis ===\n")
    
    evaluator = AgentEvaluator()
    benchmark = TaskBenchmark.load("standard_tasks")
    
    # test different parameter configurations
    test_configs = [
        ("low temp, short context", AgentParameters(temperature=0.3, context_length=500)),
        ("high temp, long context", AgentParameters(temperature=1.5, context_length=2000)),
        ("many steps", AgentParameters(max_steps=20)),
        ("few steps", AgentParameters(max_steps=3)),
        ("default", AgentParameters()),
    ]
    
    results_summary = []
    
    for config_name, params in test_configs:
        agent = TestAgent(params)
        results = evaluator.evaluate_batch(agent, benchmark.tasks)
        
        avg_score = sum(r.score for r in results) / len(results)
        avg_accuracy = sum(r.metrics.get("accuracy", 0) for r in results) / len(results)
        avg_efficiency = sum(r.metrics.get("efficiency", 0) for r in results) / len(results)
        avg_coherence = sum(r.metrics.get("coherence", 0) for r in results) / len(results)
        avg_adaptability = sum(r.metrics.get("adaptability", 0) for r in results) / len(results)
        
        results_summary.append({
            "config": config_name,
            "params": params.to_dict(),
            "overall": avg_score,
            "accuracy": avg_accuracy,
            "efficiency": avg_efficiency,
            "coherence": avg_coherence,
            "adaptability": avg_adaptability
        })
        
        print(f"{config_name}:")
        print(f"  overall: {avg_score:.4f}")
        print(f"  accuracy: {avg_accuracy:.4f}, efficiency: {avg_efficiency:.4f}")
        print(f"  coherence: {avg_coherence:.4f}, adaptability: {avg_adaptability:.4f}")
        print()
    
    # find best config
    best = max(results_summary, key=lambda x: x["overall"])
    print(f"best configuration: {best['config']}")
    print(f"best score: {best['overall']:.4f}")


if __name__ == "__main__":
    test_parameter_impact()
