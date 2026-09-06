import os
from pathlib import Path
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, BaseMessage
from config import HARNESS_ROOT, AGENT_NAME, OPENAI_BASE_URL, MODEL_NAME
from tools import tools, toolsByName, append_to_short_memory

# ------------------------------------------------------------------
# A. Light-Weight Context Assembler (Index Only)
# ------------------------------------------------------------------
def assemble_context(clusterRoot: str, agentName: str) -> str:
    root = Path(clusterRoot)
    agentDir = root / agentName

    def read_file(path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else ""

    globalCtx = read_file(root / "context.md")
    agentCtx = read_file(agentDir / "agent.md")
    shortMem = read_file(agentDir / "memories" / "short-memory.md")
    longMem = read_file(agentDir / "memories" / "long-memory.md")
    skillsIndex = read_file(agentDir / "skills" / "skills.md")

    return f"""
=== GLOBAL CONTEXT ===
{globalCtx}

=== AGENT IDENTITY ===
{agentCtx}

=== SHORT-TERM MEMORY ===
{shortMem}

=== LONG-TERM MEMORY ===
{longMem}

=== AVAILABLE SKILLS INDEX ===
{skillsIndex}

=== INSTRUCTIONS FOR SKILL USAGE ===
1. Check the 'AVAILABLE SKILLS INDEX' above to see if a specialized skill matches the task.
2. If a relevant skill exists, call the `get_skill` tool to fetch its detailed Standard Operating Procedure (.skill file).
3. Follow the fetched SOP instructions.
4. If no listed skill fits the request, rely on your own reasoning and available tools.

=== LONG-TERM MEMORY MANAGEMENT INSTRUCTIONS ===
1. Long-memory is a curated journal of tasks reviews, best-practices, enduring rules, architectural decisions, and preferences, etc.
2. You have tools to READ (`read_long_memory`), ADD (`append_long_memory`), UPDATE (`update_long_memory`), and DELETE (`delete_long_memory`) records.
3. If the user updates a rule or contradicts a past instruction (e.g., "We switched from PostgreSQL to SQLite"):
   a. Call `read_long_memory` to view current entries and their index numbers.
   b. Identify the outdated entry number.
   c. Call `update_long_memory` or `delete_long_memory` to revise the journal.
4. Do NOT leave conflicting or outdated rules in long-memory.md.
"""


# ------------------------------------------------------------------
# C. Multi-Turn Reasoning Loop
# ------------------------------------------------------------------
def run_agent(userPrompt: str, maxIterations: int = 5):
    llm = ChatOpenAI(
        base_url = OPENAI_BASE_URL,
        api_key = lambda: "ollama",
        model = MODEL_NAME,
        temperature=0.1
    ).bind_tools(tools)

    systemPrompt = assemble_context(HARNESS_ROOT, AGENT_NAME)
    messages: List[BaseMessage] = [
        SystemMessage(content=systemPrompt),
        HumanMessage(content=userPrompt)
    ]

    executedTools: List[str] = []

    for _ in range(maxIterations):
        response = llm.invoke(messages)
        messages.append(response)

        # Loop ends when no more tools are called
        if not response.tool_calls:
            finalContent = response.content

            # AUTOMATIC LOGGING STEP
            append_to_short_memory(HARNESS_ROOT, AGENT_NAME, userPrompt, executedTools, finalContent)
            return finalContent

        # Execute tool calls
        for toolCall in response.tool_calls:
            toolName = toolCall["name"]
            executedTools.append(toolName)

            selectedTool = toolsByName[toolName]
            toolOutput = selectedTool.invoke(toolCall["args"])

            messages.append(ToolMessage(content=str(toolOutput), tool_call_id=toolCall["id"]))

    fallbackMsg = "Reached maximum execution turns."
    append_to_short_memory(HARNESS_ROOT, AGENT_NAME, userPrompt, executedTools, fallbackMsg)
    return fallbackMsg



# ------------------------------------------------------------------
# D. Test Scenarios
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Example 1: Triggering a task that matches an existing skill index
    print("--- Scenario 1: Needs Skill ---")
    response1 = run_agent("Perform a health check on my project workspace.")
    print("Agent Response:\n", response1)

    # Example 2: Prompt with no matching skill (uses general reasoning)
    print("\n--- Scenario 2: No Matching Skill ---")
    response2 = run_agent("Write a short poem about local computing.")
    print("Agent Response:\n", response2)