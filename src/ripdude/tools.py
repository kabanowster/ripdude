import json
import subprocess
from typing import List, Optional

import tiktoken
from langchain.mcp import MCPAdapter
from langchain_core.messages import AnyMessage, ToolMessage, RemoveMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from ripdude.config import COGNEE_MCP_URL, MCP_TOOLS, PLAYWRIGHT_MCP_URL
from ripdude.schema import RipdudeState, ToolChunks


@tool
def shell_cmd(command: str) -> dict[str, str]:
    """Execute shell commands."""
    proc = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    return {
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


@tool
def get_tool_chunk(index: int, runtime: ToolRuntime):
    """
    Outputs of some tools hit the token limit, and their message is providing you the number of chunks.
    This tool allows you to access a specific chunk by 0-based index.
    A max of 3 chunks is kept in your context as the FIFO window, so be mindful when loading them.
    """
    state: RipdudeState = RipdudeState.model_validate(runtime.state)
    tool_chunks: ToolChunks = state.tool_chunks
    chunks: List[str] = tool_chunks.get("chunks")
    chunks_count = len(chunks)
    if index >= chunks_count:
        return f"Provided index is out of range. The list holds {chunks_count} chunks."

    update_messages: list[ToolMessage | RemoveMessage] = [
        ToolMessage(
            content=f"Tool: {tool_chunks.get('tool')}, Chunk: {index}\n\n{chunks[index]}",
            tool_call_id=runtime.tool_call_id,
            additional_kwargs={"tool_chunk": index, "managed_tool": True}
        )
    ]

    # fifo chunks
    current_chunks = [m for m in runtime.state["messages"] if m.additional_kwargs.get("tool_chunk") is not None]
    if len(current_chunks) == 3:
        update_messages.append(RemoveMessage(id=current_chunks[0].id))

    return Command(update={"messages": update_messages})


local_tools = [shell_cmd, get_tool_chunk]

# base tools
mcp_tools = {
    "cognee": {"url": COGNEE_MCP_URL},
    "playwright": {"url": PLAYWRIGHT_MCP_URL}
}

# MCP_TOOLS: List of pattern: <tool_name1>@<url1>,<tool_name2>@<url2>,...
if MCP_TOOLS:
    for t in MCP_TOOLS.split(","):
        tool = t.split("@")
        mcp_tools[tool[0]] = {"url": tool[1]}

mcp_adapter = MCPAdapter({"mcpServers": mcp_tools})

excluded_tools = [
    "playwright_browser_evaluate",
    "playwright_browser_run_code_unsafe",
    "playwright_browser_run_code",
    "cognee_search_tools",
    "cognee_call_tool",
    "playwright_browser_resize",
    "playwright_browser_install",
    "playwright_browser_network_requests",
    "playwright_browser_console_messages",
    "playwright_browser_take_screenshot"
]


def count_tokens(content: str) -> int:
    """Count tokens for provided string content"""
    encoding = tiktoken.get_encoding("cl100k_base")  # todo: env encoding
    return len(encoding.encode(content))


def count_messages_tokens(messages: List[AnyMessage]) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    for message in messages:
        total_tokens += len(encoding.encode(str(message.content) or ""))
    return total_tokens


def chunk_messages(messages: List[AnyMessage], max_chunk: int = 6000):
    chunks = []
    current_chunk = []
    current_tokens = 0

    for msg in messages:
        msg_tokens = count_messages_tokens([msg])

        if current_tokens + msg_tokens > max_chunk and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(msg)
        current_tokens += msg_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_text(text: str, max_chunk: int = 10000) -> List[str]:
    lines: List[str] = text.splitlines(keepends=True)

    if not lines:
        return []

    if len(lines) == 1:
        return chunk_text_line(text, max_chunk)

    chunks = []
    current_chunk_lines = []
    current_token_count = 0

    for line in lines:
        line_tokens = count_tokens(line)

        # Handle single lines that exceed the max limit entirely
        if line_tokens > max_chunk:
            # First, flush whatever we accumulated so far
            if current_chunk_lines:
                chunks.append("".join(current_chunk_lines))
                current_chunk_lines = []
                current_token_count = 0

            # Split and add the oversized line
            chunks.extend(chunk_text_line(line, max_chunk))
            continue

        # If adding this line exceeds max_chunk, flush current chunk
        if current_token_count + line_tokens > max_chunk:
            chunks.append("".join(current_chunk_lines))
            current_chunk_lines = [line]
            current_token_count = line_tokens
        else:
            current_chunk_lines.append(line)
            current_token_count += line_tokens

    # Remember the final leftover chunk
    if current_chunk_lines:
        chunks.append("".join(current_chunk_lines))

    return chunks


def chunk_text_line(line: str, max_chunk: int = 10000) -> List[str]:
    chunks = []
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(line)
    for i in range(0, len(tokens), max_chunk):
        chunk_tokens = tokens[i: i + max_chunk]
        chunks.append(enc.decode(chunk_tokens))
    return chunks


def get_messages_of_generation(state: RipdudeState, generation: Optional[int], exclude_bucket: bool = True) -> List[AnyMessage]:
    bucket = state.summarization["summarization_bucket"]
    selected_generation = generation or state.summarization["generation"]
    return [m for m in state.messages if m.additional_kwargs.get("generation") == selected_generation and (m.id not in bucket if exclude_bucket else True)]


def chunk_tool_output(message: AnyMessage) -> ToolChunks:
    name = message.name or "tool_used_before"
    text = ""
    if "shell" in name:
        text = json.loads(str(message.content))["stdout"]
    if "playwright" in name:
        text = message.content[0]["text"]

    return ToolChunks(tool=name, chunks=chunk_text(text))