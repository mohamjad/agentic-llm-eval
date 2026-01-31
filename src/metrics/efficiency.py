"""efficiency metrics - fewer steps and less time = better"""

from typing import Dict, Any, List


class EfficiencyMetric:
    """calculate efficiency based on steps and time"""
    
    def __init__(self, max_reasonable_steps: int = 20):
        """max steps before score starts dropping (default: 20)"""
        self.max_reasonable_steps = max_reasonable_steps
    
    def collect(
        self,
        agent: Any,
        task: Any,
        result: Any,
        trace: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """return efficiency score, step count, and avg step time"""
        steps_count = len(trace)
        
        # Calculate efficiency score based on steps
        # Formula: max(0, 1.0 - (steps / max_reasonable_steps))
        # This gives 1.0 for 0 steps, decreasing linearly to 0.0 at max_reasonable_steps
        if steps_count == 0:
            efficiency_score = 1.0
        elif steps_count <= self.max_reasonable_steps:
            efficiency_score = max(0.0, 1.0 - (steps_count / self.max_reasonable_steps))
        else:
            # Beyond max_reasonable_steps, score becomes negative, clamp to 0
            efficiency_score = 0.0
        
        # Calculate average step time
        avg_step_time = self._calculate_avg_step_time(trace)
        
        return {
            "efficiency": efficiency_score,
            "steps_count": float(steps_count),
            "avg_step_time": avg_step_time
        }
    
    def _calculate_avg_step_time(self, trace: List[Dict[str, Any]]) -> float:
        """average time per step"""
        if not trace:
            return 0.0
        
        # Extract durations from trace steps
        durations = []
        for step in trace:
            duration = step.get("duration", 0.0)
            if isinstance(duration, (int, float)) and duration > 0:
                durations.append(float(duration))
        
        if not durations:
            return 0.0
        
        # Calculate average
        total_time = sum(durations)
        return total_time / len(durations)
