from typing import Any

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from ripdude.checkpointer import redis_saver
from ripdude.models import reasoning_model
from ripdude.nodes import AgentNodes
from ripdude.schema import RipdudeState


def build_graph(tools: list[Any]):
    nodes = AgentNodes(reasoning_model=reasoning_model, tools=tools)

    agent = StateGraph(RipdudeState)

    # nodes
    agent.add_node("human_tag_generation", nodes.tag_generation_on_last_message)
    agent.add_node("tool_tag_generation", nodes.tag_generation_on_last_message)
    agent.add_node("evaluate_intent", nodes.evaluate_intent)
    agent.add_node("call_model_with_tools", nodes.call_model_with_tools)
    agent.add_node("pre_tools", nodes.pre_tools)
    agent.add_node("tools", ToolNode(tools))
    agent.add_node("limiter", nodes.limiter)
    agent.add_node("summarize", nodes.summarize)
    agent.add_node("summarize_for_steps", nodes.summarize)
    agent.add_node("create_steps", nodes.create_steps)
    agent.add_node("evaluate_step", nodes.evaluate_step)
    agent.add_node("task_pre_end", nodes.remove_autoprompts)
    agent.add_node("task_end", nodes.task_end)
    agent.add_node("task_after", nodes.task_after)
    agent.add_node("pre_end", nodes.remove_autoprompts)

    # edges
    agent.add_edge(START, "human_tag_generation")
    agent.add_conditional_edges("human_tag_generation", nodes.route_step_continuation)
    agent.add_conditional_edges("evaluate_intent", nodes.route_intent)

    agent.add_edge("summarize_for_steps", "create_steps")
    agent.add_edge("create_steps", "call_model_with_tools")

    agent.add_conditional_edges("call_model_with_tools", nodes.route_tools_loop)
    agent.add_edge("pre_tools", "tools")
    agent.add_edge("tools", "tool_tag_generation")
    agent.add_edge("tool_tag_generation", "limiter")
    agent.add_edge("limiter", "evaluate_step")

    agent.add_conditional_edges("evaluate_step", nodes.route_steps)

    agent.add_edge("task_pre_end", "task_end")
    agent.add_edge("task_end", "task_after")
    agent.add_edge("task_after", "pre_end")

    agent.add_conditional_edges("pre_end", nodes.route_summary)
    agent.add_edge("summarize", END)

    return agent.compile(checkpointer=redis_saver)