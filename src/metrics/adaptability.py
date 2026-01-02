"""adaptability metrics - measures how well agent adapts to different contexts"""

from typing import Dict, Any, List
import statistics


class AdaptabilityMetric:
    """measure adaptability - how agent handles varied contexts and task structures"""
    
    def __init__(self):
        self.context_types = ["knowledge", "reasoning", "tool_usage", "creative"]
    
    def collect(
        self,
        agent: Any,
        task: Any,
        result: Any,
        trace: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """calculate adaptability score"""
        task_metadata = getattr(task, "metadata", {}) or {}
        task_category = task_metadata.get("category", "unknown")
        task_difficulty = task_metadata.get("difficulty", "medium")
        
        # check if agent adapted approach based on task type
        approach_adaptation = self._check_approach_adaptation(trace, task_category)
        
        # check performance across different contexts
        context_handling = self._check_context_handling(task_category, result)
        
        # check if agent adjusted complexity based on difficulty
        complexity_adaptation = self._check_complexity_adaptation(trace, task_difficulty)
        
        # overall adaptability
        adaptability_score = (
            approach_adaptation * 0.4 +
            context_handling * 0.4 +
            complexity_adaptation * 0.2
        )
        
        return {
            "adaptability": adaptability_score,
            "approach_adaptation": approach_adaptation,
            "context_handling": context_handling,
            "complexity_adaptation": complexity_adaptation
        }
    
    def _check_approach_adaptation(self, trace: List[Dict[str, Any]], category: str) -> float:
        """check if agent adapted approach to task category"""
        if not trace:
            return 0.5  # neutral if no trace
        
        # count different action types used
        action_types = [step.get("action_type", "") for step in trace]
        unique_actions = len(set(action_types))
        
        # more diverse actions = better adaptation
        # normalize: 1-2 actions = 0.5, 3-4 = 0.7, 5+ = 1.0
        if unique_actions <= 2:
            return 0.5
        elif unique_actions <= 4:
            return 0.7
        else:
            return 1.0
    
    def _check_context_handling(self, category: str, result: Any) -> float:
        """check if result is appropriate for context"""
        result_str = str(result).lower()
        
        # different categories need different response characteristics
        if category == "knowledge":
            # knowledge tasks should be factual and direct
            has_facts = any(word in result_str for word in ["is", "are", "was", "were", "the"])
            return 1.0 if has_facts else 0.6
        
        elif category == "reasoning":
            # reasoning tasks should show steps or logic
            has_reasoning = any(word in result_str for word in ["because", "since", "therefore", "step"])
            return 1.0 if has_reasoning else 0.6
        
        elif category == "tool_usage":
            # tool usage should show tool calls in trace
            return 0.8  # assume good if we got here
        
        else:
            # creative or unknown - be flexible
            return 0.7
    
    def _check_complexity_adaptation(self, trace: List[Dict[str, Any]], difficulty: str) -> float:
        """check if agent adjusted complexity based on difficulty"""
        steps = len(trace)
        
        # easy tasks should be simpler (fewer steps)
        # hard tasks can be more complex (more steps)
        if difficulty == "easy":
            # fewer steps = better adaptation
            return 1.0 if steps <= 3 else max(0.5, 1.0 - (steps - 3) * 0.1)
        elif difficulty == "hard":
            # more steps acceptable
            return 1.0 if steps >= 3 else 0.7
        else:
            # medium - balanced
            return 1.0 if 2 <= steps <= 5 else 0.7
