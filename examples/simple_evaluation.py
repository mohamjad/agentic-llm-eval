"""simple example - evaluate an agent on tasks"""

from src.evaluators import AgentEvaluator
from src.benchmarks import TaskBenchmark
from src.agents.base import BaseAgent, AgentExecutionTrace


class SimpleAgent(BaseAgent):
    """basic agent that processes task input"""
    
    def execute(self, task, trace: AgentExecutionTrace = None):
        """process task and return result"""
        task_input = getattr(task, "input", {})
        
        # Record reasoning step if tracing
        if trace:
            trace.add_step(
                action_type="reasoning",
                action_name="process_input",
                input_data={"task_input": task_input},
                output_data={"processed": True},
                duration=0.01,
                metadata={"agent": "SimpleAgent"}
            )
        
        # Simple processing - just return formatted input
        result = {"result": f"Processed: {task_input}"}
        
        # Record completion if tracing
        if trace:
            trace.add_step(
                action_type="completion",
                action_name="return_result",
                input_data={},
                output_data={"result": result},
                duration=0.0
            )
        
        return result


def main():
    """run evaluation example"""
    # Initialize evaluator with explicit configuration
    evaluator_config = {
        "use_tracer": True,  # Enable automatic tracing
        "success_threshold": 0.7,  # Minimum score for success
        "metric_weights": {
            "accuracy": 0.4,
            "efficiency": 0.3,
            "tool_usage": 0.2,
            "safety": 0.1
        }
    }
    evaluator = AgentEvaluator(config=evaluator_config)
    
    # Load benchmark
    benchmark = TaskBenchmark.load("standard_tasks")
    
    # Create agent
    agent = SimpleAgent()
    
    # Run evaluation
    print(f"Running evaluation on {len(benchmark)} tasks...")
    print(f"Agent: {agent.get_name()}")
    print(f"Evaluator configuration:")
    print(f"  Success threshold: {evaluator.success_threshold}")
    print(f"  Metric weights: {evaluator.metric_weights}")
    print()
    
    results = evaluator.evaluate_batch(agent, benchmark.tasks)
    
    # Print detailed results
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Task {i}: {result.task_id} ---")
        print(f"Success: {result.success}")
        print(f"Overall Score: {result.score:.4f}")
        print(f"\nDetailed Metrics:")
        print(f"  Accuracy: {result.metrics.get('accuracy', 0.0):.4f}")
        print(f"  Efficiency: {result.metrics.get('efficiency', 0.0):.4f}")
        print(f"  Tool Usage: {result.metrics.get('tool_usage', 0.0):.4f}")
        print(f"  Safety: {result.metrics.get('safety_score', 0.0):.4f}")
        print(f"  Steps Taken: {result.metrics.get('steps_taken', 0)}")
        print(f"  Execution Time: {result.metrics.get('execution_time', 0.0):.4f}s")
        
        if result.error:
            print(f"\nError: {result.error}")
        
        if result.execution_trace:
            print(f"\nExecution Trace ({len(result.execution_trace)} steps):")
            for step in result.execution_trace[:3]:  # Show first 3 steps
                print(f"  Step {step.get('step_number')}: {step.get('action_type')} - {step.get('action_name')}")
            if len(result.execution_trace) > 3:
                print(f"  ... and {len(result.execution_trace) - 3} more steps")
    
    # Calculate and display overall statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r.success)
    success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
    avg_score = sum(r.score for r in results) / total_tasks if total_tasks > 0 else 0.0
    
    # Calculate average metrics
    avg_accuracy = sum(r.metrics.get('accuracy', 0.0) for r in results) / total_tasks
    avg_efficiency = sum(r.metrics.get('efficiency', 0.0) for r in results) / total_tasks
    avg_safety = sum(r.metrics.get('safety_score', 0.0) for r in results) / total_tasks
    
    print(f"Total Tasks: {total_tasks}")
    print(f"Successful Tasks: {successful_tasks}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"\nAverage Scores:")
    print(f"  Overall: {avg_score:.4f}")
    print(f"  Accuracy: {avg_accuracy:.4f}")
    print(f"  Efficiency: {avg_efficiency:.4f}")
    print(f"  Safety: {avg_safety:.4f}")


if __name__ == "__main__":
    main()
