from typing import Optional, Literal, List, TypedDict, Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

AorB = Literal["A", "B"]
YESorNO = Literal["YES", "NO"]

StepEvaluationStatus = Literal["COMPLETED", "RUNNING", "CANCEL", "HUMAN"]


class FinalMessage(TypedDict):
    """Final message to be sent to the user, leave empty if it is not needed."""
    message: Optional[str]


class StepEvaluation(TypedDict):
    """Evaluation of an active task step."""
    evaluation: StepEvaluationStatus


class UserIntent(TypedDict):
    """Represents the user's intent to either be conversational (A) or directive (B) with a specific goal."""
    intent: Literal["A", "B"]
    current_goal: str


class Step(TypedDict):
    """Step is a part of a planned task."""
    ordinal_id: int
    title: str
    description: Optional[str]
    status: Literal["pending", "ongoing", "done"]


class TaskDescriptor(TypedDict):
    steps: List[Step]
    active_step: int | None
    overall_status: StepEvaluationStatus


class Summarization(TypedDict):
    summarization_bucket: List[str]
    rolling_summary: str
    artifacts: str
    generation: int


class SummarizationAnswer(TypedDict):
    """Summarization details"""
    summary: str
    artifacts: str


class ToolChunks(TypedDict):
    tool: str
    chunks: List[str]


class RipdudeState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    user_intent: UserIntent = UserIntent(intent="A", current_goal="Topic research")
    task_descriptor: TaskDescriptor = TaskDescriptor(steps=[], active_step=None, overall_status="COMPLETED")
    summarization: Summarization = Summarization(summarization_bucket=[], rolling_summary="", artifacts="", generation=1)
    tool_chunks: ToolChunks = ToolChunks(tool="", chunks=[])


class AppState(TypedDict):
    graph: CompiledStateGraph[RipdudeState]