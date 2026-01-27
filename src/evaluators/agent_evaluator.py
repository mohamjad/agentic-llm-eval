"""
main evaluator - runs agents on tasks and calculates scores
"""

from typing import Dict, Any, List, Optional
from .base import BaseEvaluator, EvaluationResult
from src.agents.base import AgentExecutionTrace
from src.agents.tracer import AgentTracer
from src.metrics.accuracy import AccuracyMetric
from src.metrics.efficiency import EfficiencyMetric
from src.metrics.safety import SafetyMetric
from src.metrics.coherence import CoherenceMetric
from src.metrics.adaptability import AdaptabilityMetric
import time
from datetime import datetime


class AgentEvaluator(BaseEvaluator):
    """evaluates agents on tasks and calculates metrics"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        config options:
        - use_tracer: auto-trace agent execution (default: True)
        - metric_weights: dict of metric -> weight for scoring
        - success_threshold: min score to pass (default: 0.7)
        """
        super().__init__(config)
        
        # Initialize default metrics collectors
        self.accuracy_metric = AccuracyMetric()
        self.efficiency_metric = EfficiencyMetric()
        self.safety_metric = SafetyMetric()
        self.coherence_metric = CoherenceMetric()
        self.adaptability_metric = AdaptabilityMetric()
        
        # Configuration defaults
        self.use_tracer = self.config.get("use_tracer", True)
        self.success_threshold = self.config.get("success_threshold", 0.7)
        
        # Metric weights for overall score calculation
        # These are explicit and configurable
        self.metric_weights = self.config.get("metric_weights", {
            "accuracy": 0.4,
            "efficiency": 0.3,
            "tool_usage": 0.2,
            "safety": 0.1
        })
    
    def evaluate(self, agent: Any, task: Any) -> EvaluationResult:
        """run agent on task and return evaluation results"""
        task_id = getattr(task, "id", "unknown")
        start_time = time.time()
        
        # Create trace object
        trace = AgentExecutionTrace(
            task_id=task_id,
            start_time=datetime.now()
        )
        
        try:
            # Wrap agent with tracer if enabled
            if self.use_tracer:
                tracer = AgentTracer(agent)
                result = tracer.execute(task, trace)
            else:
                # Agent should handle tracing itself
                result = agent.execute(task, trace)
                trace.end_time = datetime.now()
                trace.final_result = result
            
            # Calculate all metrics explicitly
            execution_time = time.time() - start_time
            metrics = self._calculate_all_metrics(
                agent=agent,
                task=task,
                result=result,
                execution_time=execution_time,
                trace=trace
            )
            
            # Compute overall score using explicit weighted average
            overall_score = self._compute_overall_score(metrics)
            metrics["overall_score"] = overall_score
            
            # Determine success based on explicit threshold
            success = self._check_success(task, result, metrics, overall_score)
            
            return EvaluationResult(
                task_id=task_id,
                success=success,
                score=overall_score,
                metrics=metrics,
                execution_trace=trace.to_dict_list()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            trace.end_time = datetime.now()
            trace.error = str(e)
            
            return EvaluationResult(
                task_id=task_id,
                success=False,
                score=0.0,
                metrics={
                    "execution_time": execution_time,
                    "error_occurred": True
                },
                error=str(e),
                execution_trace=trace.to_dict_list()
            )
    
    def _calculate_all_metrics(
        self,
        agent: Any,
        task: Any,
        result: Any,
        execution_time: float,
        trace: AgentExecutionTrace
    ) -> Dict[str, Any]:
        """calculate all metrics and return as dict"""
        # Convert trace to list format for metrics
        trace_list = trace.to_dict_list()
        
        # Calculate accuracy metrics
        accuracy_metrics = self.accuracy_metric.collect(agent, task, result, trace_list)
        
        # Calculate efficiency metrics
        efficiency_metrics = self.efficiency_metric.collect(agent, task, result, trace_list)
        
        # Calculate safety metrics
        safety_metrics = self.safety_metric.collect(agent, task, result, trace_list)
        
        # Calculate coherence metrics
        coherence_metrics = self.coherence_metric.collect(agent, task, result, trace_list)
        
        # Calculate adaptability metrics
        adaptability_metrics = self.adaptability_metric.collect(agent, task, result, trace_list)
        
        # Calculate tool usage metrics
        tools_used = trace.get_tools_used()
        tool_usage_score = self._calculate_tool_usage_score(trace, tools_used)
        
        # Combine all metrics
        all_metrics = {
            "execution_time": execution_time,
            "steps_taken": len(trace.steps),
            "tools_used_count": len(tools_used),
            "tools_used_list": tools_used,
            "total_duration": trace.get_total_duration(),
        }
        
        # Add metric-specific results
        all_metrics.update(accuracy_metrics)
        all_metrics.update(efficiency_metrics)
        all_metrics.update(safety_metrics)
        all_metrics.update(coherence_metrics)
        all_metrics.update(adaptability_metrics)
        all_metrics["tool_usage"] = tool_usage_score
        
        return all_metrics
    
    def _calculate_tool_usage_score(
        self,
        trace: AgentExecutionTrace,
        tools_used: List[str]
    ) -> float:
        """score based on tool usage ratio - ideal is 20-60% of steps"""
        # Count tool call steps
        tool_call_steps = [
            step for step in trace.steps
            if step.action_type == "tool_call"
        ]
        
        total_steps = len(trace.steps)
        if total_steps == 0:
            return 1.0  # No steps means no tool misuse
        
        # Score based on appropriate tool usage
        # Too many tool calls relative to steps is inefficient
        # Too few might indicate missing needed tools
        tool_call_ratio = len(tool_call_steps) / total_steps if total_steps > 0 else 0.0
        
        # Ideal ratio is between 0.2 and 0.6 (20-60% of steps are tool calls)
        if 0.2 <= tool_call_ratio <= 0.6:
            return 1.0
        elif tool_call_ratio < 0.2:
            # Too few tool calls - might be missing needed tools
            return tool_call_ratio / 0.2
        else:
            # Too many tool calls - inefficient
            return max(0.0, 1.0 - ((tool_call_ratio - 0.6) / 0.4))
    
    def _check_success(
        self,
        task: Any,
        result: Any,
        metrics: Dict[str, Any],
        overall_score: float
    ) -> bool:
        """check if task passed - needs task check, score threshold, and no errors"""
        # First check task's own success criteria
        if hasattr(task, "check_success"):
            task_success = task.check_success(result)
            if not task_success:
                return False
        
        # Check overall score against threshold
        if overall_score < self.success_threshold:
            return False
        
        # Check for errors
        if metrics.get("error_occurred", False):
            return False
        
        return True
    
    def _compute_overall_score(self, metrics: Dict[str, Any]) -> float:
        """weighted average of all metrics"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        # Iterate through configured weights
        for metric_name, weight in self.metric_weights.items():
            if metric_name in metrics:
                metric_value = metrics[metric_name]
                # Ensure metric value is between 0 and 1
                if isinstance(metric_value, (int, float)):
                    metric_value = max(0.0, min(1.0, float(metric_value)))
                    weighted_sum += metric_value * weight
                    total_weight += weight
        
        # Return weighted average, or 0.0 if no weights applied
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0
