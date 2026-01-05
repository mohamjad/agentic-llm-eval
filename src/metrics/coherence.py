"""coherence metrics - measures consistency and logical flow"""

from typing import Dict, Any, List
import re


class CoherenceMetric:
    """measure coherence - consistency, logical flow, topic relevance"""
    
    def __init__(self):
        # coherence indicators
        self.contradiction_keywords = ["but", "however", "although", "despite", "contradict"]
        self.continuation_keywords = ["therefore", "thus", "hence", "consequently", "as a result"]
    
    def collect(
        self,
        agent: Any,
        task: Any,
        result: Any,
        trace: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """calculate coherence score from result and trace"""
        result_str = str(result).lower()
        
        # check for logical flow
        logical_flow = self._check_logical_flow(result_str, trace)
        
        # check for contradictions
        contradiction_score = self._check_contradictions(result_str, trace)
        
        # check topic consistency
        topic_consistency = self._check_topic_consistency(result_str, task)
        
        # overall coherence (weighted average)
        coherence_score = (
            logical_flow * 0.4 +
            contradiction_score * 0.3 +
            topic_consistency * 0.3
        )
        
        return {
            "coherence": coherence_score,
            "logical_flow": logical_flow,
            "contradiction_score": contradiction_score,
            "topic_consistency": topic_consistency
        }
    
    def _check_logical_flow(self, result_str: str, trace: List[Dict[str, Any]]) -> float:
        """check if response has logical flow indicators"""
        if not result_str:
            return 0.0
        
        # count continuation keywords (good)
        continuation_count = sum(1 for kw in self.continuation_keywords if kw in result_str)
        
        # normalize based on text length
        words = len(result_str.split())
        if words == 0:
            return 0.0
        
        # more continuation keywords relative to text length = better flow
        flow_score = min(1.0, continuation_count / max(1, words / 50))
        
        return flow_score
    
    def _check_contradictions(self, result_str: str, trace: List[Dict[str, Any]]) -> float:
        """check for contradictions - fewer is better"""
        contradiction_count = sum(1 for kw in self.contradiction_keywords if kw in result_str)
        
        # score decreases with more contradictions
        # 0 contradictions = 1.0, 1-2 = 0.7, 3+ = 0.3
        if contradiction_count == 0:
            return 1.0
        elif contradiction_count <= 2:
            return 0.7
        else:
            return 0.3
    
    def _check_topic_consistency(self, result_str: str, task: Any) -> float:
        """check if result stays on topic"""
        task_input = str(getattr(task, "input", {})).lower()
        task_desc = str(getattr(task, "description", "")).lower()
        
        # extract key terms from task
        task_terms = set(re.findall(r'\b\w{4,}\b', task_input + " " + task_desc))
        result_terms = set(re.findall(r'\b\w{4,}\b', result_str))
        
        if not task_terms:
            return 1.0  # no topic to check
        
        # calculate overlap
        overlap = len(task_terms & result_terms)
        total_task_terms = len(task_terms)
        
        # consistency = overlap ratio
        consistency = overlap / total_task_terms if total_task_terms > 0 else 0.0
        
        return min(1.0, consistency * 1.5)  # allow some flexibility
