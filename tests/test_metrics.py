"""
Comprehensive tests for all metric modules
"""

import pytest
from src.metrics.accuracy import AccuracyMetric
from src.metrics.efficiency import EfficiencyMetric
from src.metrics.safety import SafetyMetric
from src.metrics.coherence import CoherenceMetric
from src.metrics.adaptability import AdaptabilityMetric
from src.benchmarks import Task


class TestAccuracyMetric:
    """Test accuracy metric calculations"""
    
    def test_exact_match_primitive(self):
        """Test exact match for primitive values"""
        metric = AccuracyMetric()
        assert metric.calculate(None, "test", "test") == 1.0
        assert metric.calculate(None, "test", "wrong") == 0.0
        assert metric.calculate(None, 42, 42) == 1.0
        assert metric.calculate(None, 42, 43) == 0.0
    
    def test_dict_accuracy(self):
        """Test dictionary comparison"""
        metric = AccuracyMetric()
        expected = {"a": 1, "b": 2, "c": 3}
        result = {"a": 1, "b": 2, "c": 3}
        assert metric.calculate(None, result, expected) == 1.0
        
        result_partial = {"a": 1, "b": 2}
        assert metric.calculate(None, result_partial, expected) == pytest.approx(2/3)
        
        result_wrong = {"a": 1, "b": 3, "c": 3}
        assert metric.calculate(None, result_wrong, expected) == pytest.approx(2/3)
    
    def test_list_accuracy(self):
        """Test list comparison"""
        metric = AccuracyMetric()
        expected = [1, 2, 3]
        result = [1, 2, 3]
        assert metric.calculate(None, result, expected) == 1.0
        
        result_partial = [1, 2, 4]
        assert metric.calculate(None, result_partial, expected) == pytest.approx(2/3)
        
        result_wrong_length = [1, 2]
        assert metric.calculate(None, result_wrong_length, expected) == 0.0
    
    def test_none_expected(self):
        """Test when expected is None"""
        metric = AccuracyMetric()
        assert metric.calculate(None, "any result", None) == 1.0
    
    def test_collect_with_task(self):
        """Test collect method with task"""
        metric = AccuracyMetric()
        task = Task(
            id="test",
            description="Test",
            input={"test": "data"},
            expected_output={"result": "success"}
        )
        result = {"result": "success"}
        trace = []
        
        metrics = metric.collect(None, task, result, trace)
        assert metrics["accuracy"] == 1.0
        assert metrics["exact_match"] == 1.0


class TestEfficiencyMetric:
    """Test efficiency metric calculations"""
    
    def test_zero_steps(self):
        """Test efficiency with zero steps"""
        metric = EfficiencyMetric()
        trace = []
        result = metric.collect(None, None, None, trace)
        assert result["efficiency"] == 1.0
        assert result["steps_count"] == 0.0
    
    def test_few_steps(self):
        """Test efficiency with few steps"""
        metric = EfficiencyMetric(max_reasonable_steps=20)
        trace = [{"duration": 0.1} for _ in range(5)]
        result = metric.collect(None, None, None, trace)
        assert result["efficiency"] > 0.5
        assert result["steps_count"] == 5.0
    
    def test_many_steps(self):
        """Test efficiency with many steps"""
        metric = EfficiencyMetric(max_reasonable_steps=20)
        trace = [{"duration": 0.1} for _ in range(25)]
        result = metric.collect(None, None, None, trace)
        assert result["efficiency"] == 0.0
        assert result["steps_count"] == 25.0
    
    def test_avg_step_time(self):
        """Test average step time calculation"""
        metric = EfficiencyMetric()
        trace = [
            {"duration": 0.1},
            {"duration": 0.2},
            {"duration": 0.3}
        ]
        result = metric.collect(None, None, None, trace)
        assert result["avg_step_time"] == pytest.approx(0.2)


class TestSafetyMetric:
    """Test safety metric detection"""
    
    def test_safe_content(self):
        """Test safe content gets full score"""
        metric = SafetyMetric()
        result = {"output": "This is a safe response"}
        trace = [{"action_type": "reasoning", "action_name": "think"}]
        
        metrics = metric.collect(None, None, result, trace)
        assert metrics["safety_score"] == 1.0
        assert metrics["has_unsafe_content"] == 1.0
    
    def test_unsafe_result(self):
        """Test unsafe result gets penalty"""
        metric = SafetyMetric()
        result = {"output": "This contains harmful content"}
        trace = []
        
        metrics = metric.collect(None, None, result, trace)
        # "harmful" is a MODERATE keyword, so severity=0.5 -> safety_score=0.3
        assert metrics["safety_score"] == 0.3
        assert metrics["has_unsafe_content"] == 0.0
    
    def test_unsafe_trace(self):
        """Test unsafe trace step gets penalty"""
        metric = SafetyMetric()
        result = {"output": "Safe output"}
        trace = [{"action_type": "reasoning", "action_name": "think", "output_data": {"thought": "violence"}}]
        
        metrics = metric.collect(None, None, result, trace)
        # "violence" is SEVERE (1.0), trace_severity * 0.7 = 0.7, overall_severity = 0.7
        # Since 0.7 >= 0.5 and < 0.8, safety_score = 0.3
        assert metrics["safety_score"] == 0.3
        assert metrics["has_unsafe_content"] == 0.0
    
    def test_custom_keywords(self):
        """Test custom unsafe keywords"""
        custom_keywords = ["badword1", "badword2"]
        metric = SafetyMetric(unsafe_keywords=custom_keywords)
        result = {"output": "This has badword1 in it"}
        
        metrics = metric.collect(None, None, result, [])
        # Custom keywords are treated as MODERATE, so severity=0.5 -> safety_score=0.3
        assert metrics["safety_score"] == 0.3
    
    def test_case_insensitive(self):
        """Test case-insensitive keyword detection"""
        metric = SafetyMetric()
        result = {"output": "This has HARMful content"}
        
        metrics = metric.collect(None, None, result, [])
        # "HARMful" contains "harmful" (MODERATE), so severity=0.5 -> safety_score=0.3
        assert metrics["safety_score"] == 0.3


class TestCoherenceMetric:
    """Test coherence metric calculations"""
    
    def test_logical_flow(self):
        """Test logical flow detection"""
        metric = CoherenceMetric()
        result = "First step. Therefore, second step. Thus, conclusion."
        trace = []
        
        metrics = metric.collect(None, None, result, trace)
        assert metrics["logical_flow"] > 0.0
        assert "coherence" in metrics
    
    def test_contradictions(self):
        """Test contradiction detection"""
        metric = CoherenceMetric()
        result = "This is true. However, this contradicts it. But wait..."
        trace = []
        
        metrics = metric.collect(None, None, result, trace)
        assert metrics["contradiction_score"] < 1.0
    
    def test_topic_consistency(self):
        """Test topic consistency"""
        metric = CoherenceMetric()
        task = Task(
            id="test",
            description="Test task about Python programming",
            input={"topic": "Python"}
        )
        result = "Python is a programming language used for data science"
        trace = []
        
        metrics = metric.collect(None, task, result, trace)
        assert metrics["topic_consistency"] > 0.0


class TestAdaptabilityMetric:
    """Test adaptability metric calculations"""
    
    def test_approach_adaptation(self):
        """Test approach adaptation scoring"""
        metric = AdaptabilityMetric()
        trace = [
            {"action_type": "reasoning"},
            {"action_type": "tool_call"},
            {"action_type": "completion"}
        ]
        result = {}
        
        metrics = metric.collect(None, None, result, trace)
        assert metrics["approach_adaptation"] > 0.0
    
    def test_context_handling(self):
        """Test context handling"""
        metric = AdaptabilityMetric()
        task = Task(
            id="test",
            description="Knowledge task",
            input={},
            metadata={"category": "knowledge"}
        )
        result = "The answer is correct"
        trace = []
        
        metrics = metric.collect(None, task, result, trace)
        assert metrics["context_handling"] > 0.0
    
    def test_complexity_adaptation(self):
        """Test complexity adaptation"""
        metric = AdaptabilityMetric()
        task = Task(
            id="test",
            description="Easy task",
            input={},
            metadata={"difficulty": "easy"}
        )
        result = {}
        trace = [{"action_type": "step"} for _ in range(2)]
        
        metrics = metric.collect(None, task, result, trace)
        assert metrics["complexity_adaptation"] > 0.0
