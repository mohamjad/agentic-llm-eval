"""auto-trace agent execution"""

from typing import Dict, Any, Optional, Callable
from .base import BaseAgent, AgentExecutionTrace, TraceStep
from datetime import datetime
import time
import functools


class AgentTracer:
    """wraps agent and auto-traces execution"""
    
    def __init__(self, agent: BaseAgent):
        """wrap an agent for tracing"""
        self.agent = agent
        self._current_trace: Optional[AgentExecutionTrace] = None
    
    def execute(self, task: Any, trace: Optional[AgentExecutionTrace] = None) -> Any:
        """execute task and record trace"""
        if trace is None:
            trace = AgentExecutionTrace(
                task_id=getattr(task, "id", "unknown"),
                start_time=datetime.now()
            )
        
        self._current_trace = trace
        
        try:
            # Record start of execution
            trace.add_step(
                action_type="execution_start",
                action_name="execute",
                input_data={"task_id": trace.task_id, "task_input": getattr(task, "input", {})},
                output_data={},
                duration=0.0
            )
            
            # Execute the agent
            start_time = time.time()
            result = self.agent.execute(task, trace)
            execution_time = time.time() - start_time
            
            # Record completion
            trace.add_step(
                action_type="execution_complete",
                action_name="execute",
                input_data={},
                output_data={"result": result},
                duration=execution_time
            )
            
            trace.end_time = datetime.now()
            trace.final_result = result
            
            return result
            
        except Exception as e:
            # Record error
            error_msg = str(e)
            trace.add_step(
                action_type="error",
                action_name="execute",
                input_data={},
                output_data={"error": error_msg},
                duration=0.0
            )
            trace.end_time = datetime.now()
            trace.error = error_msg
            raise
    
    def get_trace(self) -> Optional[AgentExecutionTrace]:
        """get current trace"""
        return self._current_trace
    
    def get_name(self) -> str:
        """wrapped agent name"""
        return self.agent.get_name()


def trace_tool_call(trace: Optional[AgentExecutionTrace], tool_name: str):
    """decorator to auto-trace tool calls"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if trace is None:
                return func(*args, **kwargs)
            
            start_time = time.time()
            input_data = {"args": args, "kwargs": kwargs}
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                trace.add_step(
                    action_type="tool_call",
                    action_name=tool_name,
                    input_data=input_data,
                    output_data={"result": result},
                    duration=duration
                )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                trace.add_step(
                    action_type="tool_call_error",
                    action_name=tool_name,
                    input_data=input_data,
                    output_data={"error": str(e)},
                    duration=duration
                )
                raise
        
        return wrapper
    return decorator
