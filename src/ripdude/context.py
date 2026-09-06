from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from ripdude.config import AGENT_NAME, HARNESS_ROOT
from ripdude.schema import RipdudeState
from ripdude.tools import get_messages_of_generation


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


async def get_context():
    root = Path(HARNESS_ROOT)
    agent_dir = root / AGENT_NAME.strip().replace(" ", "-")

    global_ctx = read_file(root / "context.md")
    agent_ctx = read_file(agent_dir / "agent.md")

    system_message = f"""
[GLOBAL CONTEXT]
{global_ctx}

[YOUR IDENTITY]
Your Alias is: {AGENT_NAME}
{agent_ctx}

[MEMORY MANAGEMENT]
* Before responding to questions, use cognee_recall.
* Use cognee_remember to note important information for later use - especially: milestones of given tasks, user preferences and things explicitly told to remember.
* If you say you're going to remember, acknowledge or commit to the information, use cognee_remember.

[VERY IMPORTANT]
* Do not make up information or tool usage. You need to be truthful and sincere above all.
* Do not make git commits if it is not explicitly described in the task.
* Stay focused on the current task and subjects in the prompt, limit your curiosity.
* If an error occurs or a variable is missing, do not attempt to guess or self-correct. Stop immediately and ask for human assistance.
* Provide only the direct answers. Do not speculate, do not analyze alternative scenarios, and do not explore unrequested features.
* Maximum single message or tool output is limited to 45000 tokens which is roughly 180 kb.
* When using tools make sure you fill in all required parameters.
* When using web remember to first accept cookies consent as it usually blocks any interaction.
* When user ask question try to answer and do not do follow up actions without explicit permission.

[FILESYSTEM MANAGEMENT]
Only use standard shell commands to manage files and filesystem.
Example utilities: ls, cat, mkdir, touch, rm, cp, mv, sed, awk.
    IMPORTANT RULES:
    * Your operational scopes are limited to $WORKSPACE_ROOT and $HOME. DO NOT READ or write files outside the scopes.
    * You are working in containerized environment.
    * The only stateful directory is $WORKSPACE_ROOT. 
"""

    return system_message


async def get_rolling_summary(state: RipdudeState):
    summary = state.summarization["rolling_summary"]
    return f"[THREAD SUMMARY]\n{summary}" if summary else ""


async def get_user_intent(state: RipdudeState):
    return f"[USER's GOAL]\n{state.user_intent['current_goal']}"


async def get_steps(state: RipdudeState) -> str:
    steps = state.task_descriptor["steps"]
    if len(steps) == 0:
        return ""

    lines = []

    for step in steps:
        details = None
        if step["status"] == "done":
            status_symbol = "[x]"
        elif step["status"] == "ongoing":
            status_symbol = "[>]"
            details = (
                f" (ACTIVE)\n"
                f"Instructions: {step['description'] or 'Execute necessary actions.'}"
            )
        else:
            status_symbol = "[ ]"

        lines.append(f"{status_symbol} {step['ordinal_id']}. {step['title']}{details or ''}")

    macro_tracklist = "\n".join(lines)

    return f"[GOAL EXECUTION PLAN]\n{macro_tracklist}" if macro_tracklist else ""


async def get_artifacts(state: RipdudeState):
    artifacts = state.summarization["artifacts"]
    return f"[CONTEXT ARTIFACTS (processed files, links and other variables)]\n{artifacts}" if artifacts else ""


async def assemble_system_message(state: RipdudeState):
    states = [
        await get_context(),
        await get_rolling_summary(state),
        await get_user_intent(state),
        await get_steps(state),
        await get_artifacts(state),
    ]
    return SystemMessage(content="\n\n".join(states))


async def assemble_task_context(state: RipdudeState):
    states = [
        await get_rolling_summary(state),
        await get_user_intent(state),
        await get_steps(state)
    ]
    return SystemMessage(content="\n\n".join(states))


async def assemble_payload(state: RipdudeState):
    # system msg
    context = await assemble_system_message(state)
    # generation msgs
    auto_prompts = [m for m in state.messages if m.additional_kwargs.get("auto_prompt_type") == "task_init"]
    messages = get_messages_of_generation(state, None)
    if messages and messages[-1].type == "ai":
        messages.append(HumanMessage(content="Continue executing the plan", additional_kwargs={"auto_prompt": True, "auto_prompt_type": "task_ai_loop"}))
    return [context] + auto_prompts + messages