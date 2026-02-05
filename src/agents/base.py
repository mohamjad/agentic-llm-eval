"""agent interface and execution tracing"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TraceStep:
    """single step in execution trace"""
    step_number: int
    action_type: str
    action_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentExecutionTrace:
    """complete trace of agent execution"""
    task_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    steps: List[TraceStep] = field(default_factory=list)
    final_result: Optional[Any] = None
    error: Optional[str] = None
    
    def add_step(
        self,
        action_type: str,
        action_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        duration: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TraceStep:
        """add a step to the trace"""
        step = TraceStep(
            step_number=len(self.steps) + 1,
            action_type=action_type,
            action_name=action_name,
            input_data=input_data,
            output_data=output_data,
            duration=duration,
            metadata=metadata or {}
        )
        self.steps.append(step)
        return step
    
    def get_total_duration(self) -> float:
        """total execution time in seconds"""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return sum(step.duration for step in self.steps)
    
    def get_tools_used(self) -> List[str]:
        """list of unique tools used"""
        tools = set()
        for step in self.steps:
            if step.action_type == "tool_call":
                tools.add(step.action_name)
        return sorted(list(tools))
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """convert trace to list of dicts"""
        return [
            {
                "step_number": step.step_number,
                "action_type": step.action_type,
                "action_name": step.action_name,
                "input_data": step.input_data,
                "output_data": step.output_data,
                "timestamp": step.timestamp.isoformat(),
                "duration": step.duration,
                "metadata": step.metadata,
            }
            for step in self.steps
        ]


class BaseAgent(ABC):
    """base interface for agents - implement execute() method"""
    
    @abstractmethod
    def execute(self, task: Any, trace: Optional[AgentExecutionTrace] = None) -> Any:
        """execute a task, optionally recording steps in trace"""
        pass
    
    def get_name(self) -> str:
        """agent name"""
        return self.__class__.__name__
