"""
Tests for agent base classes and tracing
"""

import pytest
from datetime import datetime
from src.agents.base import BaseAgent, AgentExecutionTrace, TraceStep
from src.agents.tracer import AgentTracer
from src.benchmarks import Task


class TestAgent(BaseAgent):
    """Test agent implementation"""
    
    def execute(self, task, trace=None):
        if trace:
            trace.add_step(
                action_type="test",
                action_name="execute",
                input_data={"task": str(task)},
                output_data={"result": "done"},
                duration=0.1
            )
        return {"result": "done"}


class TestAgentExecutionTrace:
    """Test execution trace functionality"""
    
    def test_create_trace(self):
        """Test trace creation"""
        trace = AgentExecutionTrace(
            task_id="test_001",
            start_time=datetime.now()
        )
        assert trace.task_id == "test_001"
        assert len(trace.steps) == 0
    
    def test_add_step(self):
        """Test adding steps to trace"""
        trace = AgentExecutionTrace(
            task_id="test_001",
            start_time=datetime.now()
        )
        step = trace.add_step(
            action_type="test",
            action_name="test_action",
            input_data={"input": "data"},
            output_data={"output": "result"},
            duration=0.5
        )
        
        assert len(trace.steps) == 1
        assert step.step_number == 1
        assert step.action_type == "test"
        assert step.duration == 0.5
    
    def test_get_tools_used(self):
        """Test extracting tools used from trace"""
        trace = AgentExecutionTrace(
            task_id="test_001",
            start_time=datetime.now()
        )
        trace.add_step(
            action_type="tool_call",
            action_name="calculator",
            input_data={},
            output_data={}
        )
        trace.add_step(
            action_type="tool_call",
            action_name="search",
            input_data={},
            output_data={}
        )
        trace.add_step(
            action_type="reasoning",
            action_name="think",
            input_data={},
            output_data={}
        )
        
        tools = trace.get_tools_used()
        assert "calculator" in tools
        assert "search" in tools
        assert len(tools) == 2
    
    def test_to_dict_list(self):
        """Test converting trace to dict list"""
        trace = AgentExecutionTrace(
            task_id="test_001",
            start_time=datetime.now()
        )
        trace.add_step(
            action_type="test",
            action_name="test_action",
            input_data={"input": "data"},
            output_data={"output": "result"},
            duration=0.5
        )
        
        dict_list = trace.to_dict_list()
        assert len(dict_list) == 1
        assert dict_list[0]["action_type"] == "test"
        assert dict_list[0]["step_number"] == 1
    
    def test_get_total_duration(self):
        """Test total duration calculation"""
        trace = AgentExecutionTrace(
            task_id="test_001",
            start_time=datetime.now()
        )
        trace.add_step(
            action_type="test",
            action_name="test_action",
            input_data={},
            output_data={},
            duration=0.5
        )
        trace.add_step(
            action_type="test",
            action_name="test_action2",
            input_data={},
            output_data={},
            duration=0.3
        )
        
        duration = trace.get_total_duration()
        assert duration == pytest.approx(0.8)


class TestAgentTracer:
    """Test agent tracer functionality"""
    
    def test_trace_execution(self):
        """Test tracing agent execution"""
        agent = TestAgent()
        tracer = AgentTracer(agent)
        task = Task(
            id="test_001",
            description="Test task",
            input={"test": "data"}
        )
        
        trace = AgentExecutionTrace(
            task_id="test_001",
            start_time=datetime.now()
        )
        
        result = tracer.execute(task, trace)
        
        assert result == {"result": "done"}
        assert len(trace.steps) > 0
        assert trace.final_result == result
    
    def test_trace_error(self):
        """Test tracing errors"""
        class FailingAgent(BaseAgent):
            def execute(self, task, trace=None):
                raise ValueError("Test error")
        
        agent = FailingAgent()
        tracer = AgentTracer(agent)
        task = Task(id="test", description="Test", input={})
        trace = AgentExecutionTrace(
            task_id="test",
            start_time=datetime.now()
        )
        
        with pytest.raises(ValueError):
            tracer.execute(task, trace)
        
        assert trace.error == "Test error"
        assert any(step.action_type == "error" for step in trace.steps)
    
    def test_auto_create_trace(self):
        """Test auto-creating trace if not provided"""
        agent = TestAgent()
        tracer = AgentTracer(agent)
        task = Task(id="test", description="Test", input={})
        
        result = tracer.execute(task)
        
        assert result == {"result": "done"}
        assert tracer.get_trace() is not None


class TestBaseAgent:
    """Test base agent interface"""
    
    def test_abstract_method(self):
        """Test that BaseAgent is abstract"""
        with pytest.raises(TypeError):
            BaseAgent()
    
    def test_get_name(self):
        """Test agent name"""
        agent = TestAgent()
        assert agent.get_name() == "TestAgent"
