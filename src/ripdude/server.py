from contextlib import asynccontextmanager
from logging import info
from typing import TypedDict

import httpx
from fastapi import FastAPI
from langchain_core.load import dumps
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from starlette.responses import StreamingResponse

from ripdude.checkpointer import redis_saver
from ripdude.config import AGENT_NAME, REASONING_MODEL_NAME, OPENAI_API_BASE, AUTOLOAD_MODEL
from ripdude.context import assemble_payload, assemble_system_message
from ripdude.graph import build_graph
from ripdude.schema import RipdudeState
from ripdude.tools import mcp_adapter, local_tools, excluded_tools, get_messages_of_generation, count_messages_tokens


class PromptRequest(TypedDict):
    prompt: str
    thread_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # keep the sessions for tools open
    async with mcp_adapter as adapter:
        tools = await adapter.list_tools()
        tools.extend(local_tools)
        filtered_tools = [tool for tool in tools if tool.name not in excluded_tools]

        # warmup model
        if AUTOLOAD_MODEL:
            await load_model()

        # redis checkpointer init
        await redis_saver.setup()

        # build graph
        app.state["graph"] = build_graph(filtered_tools)

        yield

        await redis_saver.close()


app = FastAPI(title="Agent Execution Pod", lifespan=lifespan)


def getConfig(thread_id: str):
    return RunnableConfig(
        configurable={"thread_id": thread_id},
        recursion_limit=200,
    )


# ENDPOINTS:

@app.post("/chat", response_class=StreamingResponse)
async def stream_events_endpoint(request: PromptRequest):
    graph = getattr(app.state, "graph")
    async for event in graph.astream_events({"messages": [HumanMessage(request["prompt"])]}, config=getConfig(request["thread_id"]), version="v2"):
        # default=str handles non-serializable objects (like message objects, timestamps, UUIDs)
        yield f"data: {dumps(event)}\n\n"


@app.get("/clear")
async def clear_state(thread_id: str):
    info(f"Clearing agent's cached state.")
    graph = getattr(app.state, "graph")
    await graph.aupdate_state(
        config=getConfig(thread_id),
        values={"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]},
        as_node="__start__"
    )

    return get_state(thread_id)


@app.get("/state")
async def get_state(thread_id: str):
    graph = getattr(app.state, "graph")
    return await graph.aget_state(config=getConfig(thread_id))


@app.get("/tools")
async def get_tools():
    graph = getattr(app.state, "graph")
    tool_node = graph.builder.nodes["tools"].runnable
    tools_dict = tool_node.tools_by_name
    tools_list = list(tools_dict.values())
    result = {}
    for tool in tools_list:
        result[tool.name] = {
            "description": tool.description,
            "args": tool.args
        }
    return result


@app.get("/load")
async def load_model(model: str):
    if not model:
        model = REASONING_MODEL_NAME
    async with httpx.AsyncClient(timeout=None) as client:
        await client.post(url=f"{OPENAI_API_BASE}/models/load", json={"model": model})


@app.get("/unload")
async def unload_model(model: str):
    if not model:
        model = REASONING_MODEL_NAME
    async with httpx.AsyncClient(timeout=None) as client:
        await client.post(url=f"{OPENAI_API_BASE}/models/unload", json={"model": model})


@app.get("/health")
async def health():
    return {"status": "alive", "agent": AGENT_NAME}


@app.get("/gen_tokens")
async def gen_tokens(thread_id: str, generation: str):
    try:
        gen = int(generation) if generation else None
        state = await get_state(thread_id)
        pstate = RipdudeState.model_validate(state[0])
        gen_messages = get_messages_of_generation(pstate, gen)
        tokens = count_messages_tokens(gen_messages)
        return {f"gen-{gen}-tokens": tokens}
    except ValueError:
        return {"error": "Generation must be an int number."}


@app.get("/context_tokens")
async def context_tokens(thread_id: str):
    state = await get_state(thread_id)
    pstate = RipdudeState.model_validate(state[0])
    context = await assemble_system_message(pstate)
    tokens = count_messages_tokens([context])
    return {"context-tokens": tokens}


@app.get("/print_context")
async def print_context(thread_id: str):
    state = await get_state(thread_id)
    pstate = RipdudeState.model_validate(state[0])
    context = await assemble_system_message(pstate)
    return context.content


@app.get("/total_tokens")
async def total_tokens(thread_id: str):
    state = await get_state(thread_id)
    pstate = RipdudeState.model_validate(state[0])
    payload = await assemble_payload(pstate)
    tokens = count_messages_tokens(payload)
    return {"total-tokens": tokens}


@app.get("/print_payload")
async def print_payload(thread_id: str):
    state = await get_state(thread_id)
    pstate = RipdudeState.model_validate(state[0])
    payload = await assemble_payload(pstate)
    return payload