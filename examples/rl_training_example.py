"""example: RL-based fine-tuning of agent behavior"""

from src.evaluators import AgentEvaluator
from src.benchmarks import TaskBenchmark
from src.agents.base import BaseAgent, AgentExecutionTrace
from src.rl import RLTrainer, AgentParameters


class ParameterizedAgent(BaseAgent):
    """agent that can adjust its parameters"""
    
    def __init__(self):
        self.params = AgentParameters()
    
    def execute(self, task, trace: AgentExecutionTrace = None):
        """execute with current parameters"""
        task_input = getattr(task, "input", {})
        
        # simulate parameter effects
        # context_length affects how much info we process
        max_info = min(self.params.context_length // 100, len(str(task_input)))
        
        # temperature affects randomness/creativity
        if self.params.temperature > 1.0:
            result = {"result": f"creative response: {task_input}"}
        else:
            result = {"result": f"precise response: {task_input[:max_info]}"}
        
        # max_steps affects how many operations we do
        steps_to_take = min(self.params.max_steps, 5)
        if trace:
            for i in range(steps_to_take):
                trace.add_step(
                    action_type="processing",
                    action_name=f"step_{i}",
                    input_data={"iteration": i},
                    output_data={"status": "processing"},
                    duration=0.01
                )
        
        return result
    
    def set_parameters(self, params: dict):
        """set agent parameters"""
        self.params = AgentParameters.from_dict(params)


def main():
    """run RL training example"""
    print("=== RL Training Example ===\n")
    
    # setup
    agent = ParameterizedAgent()
    evaluator = AgentEvaluator()
    benchmark = TaskBenchmark.load("standard_tasks")
    
    # initial evaluation
    print("initial evaluation:")
    initial_results = evaluator.evaluate_batch(agent, benchmark.tasks)
    initial_avg_score = sum(r.score for r in initial_results) / len(initial_results)
    print(f"average score: {initial_avg_score:.4f}\n")
    
    # RL training
    print("starting RL training...")
    trainer = RLTrainer(
        agent=agent,
        evaluator=evaluator,
        benchmark=benchmark,
        initial_params=AgentParameters(),
        learning_rate=0.02
    )
    
    training_result = trainer.train(episodes=5, tasks_per_episode=3)
    
    print(f"\nfinal parameters:")
    for param, value in training_result["final_params"].items():
        print(f"  {param}: {value}")
    
    print(f"\nfinal metrics:")
    for metric, value in training_result["final_metrics"].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
    
    # final evaluation
    print("\nfinal evaluation:")
    final_results = evaluator.evaluate_batch(agent, benchmark.tasks)
    final_avg_score = sum(r.score for r in final_results) / len(final_results)
    print(f"average score: {final_avg_score:.4f}")
    print(f"improvement: {final_avg_score - initial_avg_score:.4f}")


if __name__ == "__main__":
    main()
