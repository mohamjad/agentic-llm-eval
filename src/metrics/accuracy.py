"""accuracy metrics - compares results to expected outputs"""

from typing import Dict, Any, List, Union


class AccuracyMetric:
    """calculate accuracy by comparing result to expected"""
    
    @staticmethod
    def calculate(task: Any, result: Any, expected: Any) -> float:
        """compare result to expected - handles dicts, lists, primitives"""
        if expected is None:
            return 1.0  # No expected output means perfect score
        
        # Dictionary comparison
        if isinstance(expected, dict) and isinstance(result, dict):
            return AccuracyMetric._dict_accuracy(expected, result)
        
        # List comparison
        if isinstance(expected, list) and isinstance(result, list):
            return AccuracyMetric._list_accuracy(expected, result)
        
        # Direct equality for primitives
        return 1.0 if result == expected else 0.0
    
    @staticmethod
    def _dict_accuracy(expected: Dict[str, Any], result: Dict[str, Any]) -> float:
        """dict comparison - matches / total keys"""
        if not expected:
            return 1.0  # Empty expected dict means any result is correct
        
        matches = 0
        total_keys = len(expected)
        
        # Check each expected key-value pair
        for key, expected_value in expected.items():
            if key in result:
                actual_value = result[key]
                if actual_value == expected_value:
                    matches += 1
        
        # Return ratio of matches to total expected keys
        return matches / total_keys if total_keys > 0 else 0.0
    
    @staticmethod
    def _list_accuracy(expected: List[Any], result: List[Any]) -> float:
        """list comparison - element-wise matching"""
        if len(expected) != len(result):
            return 0.0  # Different lengths = no match
        
        if len(expected) == 0:
            return 1.0  # Both empty = match
        
        matches = sum(1 for e, r in zip(expected, result) if e == r)
        return matches / len(expected)
    
    def collect(
        self,
        agent: Any,
        task: Any,
        result: Any,
        trace: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """return accuracy and exact_match scores"""
        expected = getattr(task, "expected_output", None)
        accuracy = self.calculate(task, result, expected)
        
        return {
            "accuracy": accuracy,
            "exact_match": 1.0 if accuracy == 1.0 else 0.0
        }
