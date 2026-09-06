from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END
from langgraph.types import Command

from ripdude.config import AGENT_NAME
from ripdude.context import assemble_payload, assemble_task_context
from ripdude.schema import RipdudeState, UserIntent, Summarization, TaskDescriptor, SummarizationAnswer, StepEvaluation, StepEvaluationStatus
from ripdude.tools import count_messages_tokens, chunk_messages, get_messages_of_generation, chunk_tool_output


class AgentNodes:
    def __init__(self, reasoning_model: BaseChatModel, tools: list[Any]):
        self.model = reasoning_model
        self.model_with_tools = reasoning_model.bind_tools(tools)
        self.tools = tools

    @staticmethod
    async def tag_generation_on_last_message(state: RipdudeState):
        # tag the incoming message with generation
        state.messages[-1].additional_kwargs["generation"] = state.summarization["generation"]
        return state

    @staticmethod
    async def route_step_continuation(state: RipdudeState) -> Literal["call_model_with_tools", "evaluate_intent"]:
        descriptor = state.task_descriptor

        if descriptor.get("overall_status") in ["HUMAN", "RUNNING"]:
            return "call_model_with_tools"
        return "evaluate_intent"

    async def evaluate_intent(self, state: RipdudeState):
        latest_user_msgs = "\n\n".join([str(m.content) for m in state.messages if m.type == "human"][-3:])
        prompt = [
            SystemMessage(content=
                          "You are a helpful assistant.\n"
                          "Your goal is to evaluate the user's current intent.\n"
                          "Do not make assumptions about the user's intent.\n"
                          "Intent options:\n"
                          "'A' (default) = user is asking questions, makes plans, fills conversation.\n"
                          "'B' = user directly calls to execute task, example: 'I need you to...', 'Please do...', 'Your task is to...', etc.\n"
                          "Current goal evaluation (based on intent):\n"
                          "If A - goal is 'Topic research'.\n"
                          "If B - define user's goal, in concise, 1-2 sentences."
                          ),
            HumanMessage(content=
                         "Evaluate the user's current intent, based on last 3 messages."
                         "User messages:"
                         f"{latest_user_msgs}"
                         )
        ]

        user_intent = await self.model.with_structured_output(schema=UserIntent).ainvoke(prompt)
        return {
            "user_intent": user_intent
        }

    async def call_model_with_tools(self, state: RipdudeState, config: RunnableConfig) -> dict:
        descriptor = state.task_descriptor
        if descriptor.get("overall_status") == "HUMAN":
            descriptor["overall_status"] = "RUNNING"

        payload = await assemble_payload(state)
        response = await self.model_with_tools.ainvoke(payload, config=config)
        response.additional_kwargs["generation"] = state.summarization["generation"]
        return {
            "messages": [response],
            "task_descriptor": descriptor
        }

    @staticmethod
    async def pre_tools(state: RipdudeState):
        last_message: AIMessage = AIMessage.model_validate(state.messages[-1])
        for tool_call in last_message.tool_calls:
            for value in tool_call["args"].values():
                if not value:
                    last_message.tool_calls = []
                    state.messages.append(ToolMessage(content="***validator: Make sure to provide non-empty values to all arguments of a tool call.", name=tool_call["name"], tool_call_id=tool_call["id"], status="error"))
                    return Command(
                        update={
                            "messages": []
                        },
                        goto="call_model_with_tools"
                    )
            if not tool_call["name"].startswith("cognee"):
                continue
            match tool_call["name"]:
                case "cognee_remember":
                    tool_call["args"]["dataset_name"] = AGENT_NAME
                case "cognee_recall":
                    tool_call["args"]["datasets"] = AGENT_NAME

        return state

    @staticmethod
    async def route_tools_loop(state: RipdudeState) -> Literal["pre_tools", "evaluate_step", "pre_end"]:
        if getattr(state.messages[-1], "tool_calls"):
            return "pre_tools"

        if state.task_descriptor.get("overall_status") == "RUNNING":
            return "evaluate_step"

        return "pre_end"

    @staticmethod
    async def route_summary(state: RipdudeState):
        bucket = state.summarization["summarization_bucket"]

        if len(bucket) > 0:
            return "summarize"

        messages = state.messages
        active_tokens = count_messages_tokens(messages)
        if active_tokens >= 45000:  # todo env MAX_CTX
            return "summarize"

        return END

    @staticmethod
    async def route_intent(state: RipdudeState) -> Literal["call_model_with_tools", "summarize_for_steps"]:
        intent = state.user_intent["intent"]
        match intent:
            case "A":
                return "call_model_with_tools"
            case "B":
                return "summarize_for_steps"

    @staticmethod
    async def limiter(state: RipdudeState):
        bucket = state.summarization["summarization_bucket"]

        # tool chunker
        # limiter is run after a tool execution, so last msg is a tool
        tool_message = state.messages[-1]

        if count_messages_tokens([tool_message]) > 45000:  # todo env MAX_CTX
            if getattr(tool_message, "status", "none") == "success":
                chunks = chunk_tool_output(tool_message)
                state.tool_chunks = chunks
                tool_message.content = f"***limiter: Tool execution succeeded and produced content exceeding tokens limit. It is split into {len(chunks)} chunks. Use get_tool_chunk tool to retrieve the content one by one."
            else:
                tool_message.content = "***limiter: Tool execution failed and the output exceeded the tokens limit, so it was discarded. Try different way."

        current_messages = get_messages_of_generation(state, None)
        # active tools limit
        tools_messages = [m for m in current_messages if m.additional_kwargs.get("managed_tool", False) == False]
        for m in tools_messages[:-3]:
            if m.id:
                bucket.append(m.id)

        # messages total rolling limit (marks last messages into bucket)
        total = current_messages
        for m in current_messages:
            if m.id and count_messages_tokens(total) > 45000:  # todo env MAX_CTX
                bucket.append(m.id)
            else:
                break
            total.remove(m)

        return state

    @staticmethod
    async def remove_autoprompts(state: RipdudeState):
        removals = []
        for m in state.messages:
            if m.id and m.additional_kwargs.get("auto_prompt") == True:
                removals.append(RemoveMessage(id=m.id))

        if len(removals) == 0:
            return state

        return {
            "messages": removals
        }

    async def summarize(self, state: RipdudeState):
        # payload evaluation
        payload_to_summarize = get_messages_of_generation(state, None, False)

        if len(payload_to_summarize) <= 1:
            return state

        chunks = chunk_messages(payload_to_summarize, 60000)  # todo env CHUNK_SIZE

        # task definition
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert context summarizer for an autonomous agent.\n"
                "Your objective is to update the rolling summary of the session between user and agent up to this point.\n\n"
                "CRITICAL REQUIREMENTS:\n"
                "1. Retain crucial technical facts, key decisions, task outputs, and active constraints.\n"
                "2. Extract filepaths, variables, urls, tokens, etc., as artifacts."
                "3. Do not store file or tools outputs."
                "4. Omit conversational filler, repetitive tool schemas, and obsolete debugging logs.\n\n"
                "CURRENT ROLLING SUMMARY:\n{current_summary}\n\n"
                "CURRENT ARTIFACTS:\n{current_artifacts}\n\n"
            )),
            ("human", "Analyze new messages and update the rolling summary and artifacts:\n\n{new_logs}")
        ])

        # summarization loop
        current_summary = state.summarization.get("rolling_summary", "No prior execution history")
        current_artifacts = state.summarization.get("artifacts", "No stored artifacts")
        for chunk in chunks:
            formatted_chunk = "\n".join([f"[{m.type.upper()}]: {m.content}" for m in chunk])
            prompt = await prompt_template.ainvoke({
                "current_summary": current_summary,
                "current_artifacts": current_artifacts,
                "new_logs": formatted_chunk
            })

            response = await self.model.with_structured_output(schema=SummarizationAnswer).ainvoke(prompt)

            current_summary = response["summary"]
            current_artifacts = response["artifacts"]

        # finalization
        new_generation = state.summarization["generation"] + 1
        return {
            "summarization": Summarization(summarization_bucket=[], rolling_summary=current_summary, artifacts=current_artifacts, generation=new_generation),
        }

    async def create_steps(self, state: RipdudeState):

        prompt = [
            SystemMessage(content=
                          "You are an expert task supervisor for AI agents.\n"
                          "Your objective is to make agents tasks easier to understand and execute by providing clear and concise steps to execute.\n"
                          "Set the first step as active_step, its status to 'ongoing' and others to 'pending'. Set overall_status to 'RUNNING'\n"
                          "Always fill in the titles\n"
                          "Do not be overly detailed about the plan - fewer generalized steps is better than listing every single action. Sometimes even one step is fairly enough.\n"
                          ),
            HumanMessage(content=
                         "Given these details create a plan for the agent:\n"
                         "SUMMARY:\n"
                         f"{state.summarization["rolling_summary"] or 'No rolling summary created yet, session is probably too short.'}\n"
                         "USER'S GOAL:\n"
                         f" {state.user_intent["current_goal"]}\n"
                         "LAST MESSAGES:\n"
                         f"{'\n'.join([str(m.content) for m in state.messages if m.type == 'human'][-3:])}"
                         )
        ]

        task_descriptor = await self.model.with_structured_output(schema=TaskDescriptor).ainvoke(prompt)

        # after summarization before creating task, the new generation has no human message (jinja error)
        return {
            "task_descriptor": task_descriptor,
            "messages": [
                HumanMessage(content=
                             "Fulfill the user's goal by following the execution plan.\n"
                             "Focus only on currently ACTIVE step - DO NOT exceed the next step if it is not marked as ACTIVE.\n"
                             "If you need more feedback or tools approval, ask for it.\n"
                             "If the step can not be completed, immediately hold and provide concise argument for the supervisor.",
                             additional_kwargs={"auto_prompt": True, "auto_prompt_type": "task_init"})
            ]
        }

    async def evaluate_step(self, state: RipdudeState):
        descriptor = state.task_descriptor
        steps = descriptor["steps"]

        if len(steps) == 0:
            return state

        step_messages = get_messages_of_generation(state, None)
        task_context = await assemble_task_context(state)
        context = SystemMessage(content=
                                "You are a plan steps execution supervisor for AI agent.\n"
                                "Your job is to analyze recent actions, compare it with the execution plan and evaluate the status of the current active step:\n"
                                "-> RUNNING if agent has not yet satisfied the step and is on the right path.\n"
                                "-> CANCEL if agent is not following the plan or repeatably failing it or explicitly makes argument for cancellation.\n"
                                "-> COMPLETED if the step's goal is achieved. Do not trust the agent telling that the step is done.\n"
                                "-> HUMAN if agent requires feedback or approval.\n"
                                "Be unprejudiced, fair and unbiased.\n"
                                f"{task_context.content}\n\n"
                                )
        prompt = [
            context,
            HumanMessage(content=
                         "Evaluate recent actions:\n\n"
                         f"{'\n'.join([str(m.content) for m in step_messages])}"
                         )]

        response: StepEvaluation = await self.model.with_structured_output(schema=StepEvaluation).ainvoke(prompt)

        evaluation: StepEvaluationStatus = response.get("evaluation")

        if evaluation in ["RUNNING", "HUMAN", "CANCEL"]:
            descriptor["overall_status"] = evaluation
            return state

        # COMPLETED

        active_step = descriptor.get("active_step")
        next_id = None

        for step in steps:
            if step["ordinal_id"] == active_step:
                step["status"] = "done"
            elif step["status"] == "pending" and next_id is None:
                step["status"] = "ongoing"
                next_id = step["ordinal_id"]

        if next_id:
            descriptor["active_step"] = next_id
        else:
            descriptor["overall_status"] = "COMPLETED"

        return state

    @staticmethod
    async def route_steps(state: RipdudeState):
        steps = state.task_descriptor.get("steps")
        status = state.task_descriptor.get("overall_status")

        if status == "CANCEL":
            return "task_pre_end"

        if status == "HUMAN":
            return END

        yet_undone = next((True for t in steps if t["status"] is not "done"), False)
        if yet_undone:
            return "call_model_with_tools"

        return "task_pre_end"

    async def task_end(self, state: RipdudeState, config: RunnableConfig):
        reset_task_descriptor = TaskDescriptor(steps=[], active_step=None, overall_status="COMPLETED")

        if state.messages[-1].type == "tool":
            # supervisor ended the task after tool
            context = await assemble_task_context(state)
            messages = [m for m in get_messages_of_generation(state, None, False) if m.type in ["human", "ai"]]
            prompt = HumanMessage(content="You have ended the task. Given the context and your recent messages, assemble a final message to the user.", additional_kwargs={"auto_prompt": True})
            response = await self.model.ainvoke([context] + messages + [prompt], config=config)
            response.additional_kwargs["generation"] = state.summarization["generation"]
            return {
                "messages": [response],
                "task_descriptor": reset_task_descriptor,
            }
        else:
            return {
                "task_descriptor": reset_task_descriptor,
            }

    @staticmethod
    async def task_after(state: RipdudeState):
        bucket = state.summarization.get("summarization_bucket")
        messages = [m.id for m in get_messages_of_generation(state, None) if m.id is not None]
        bucket.extend(messages)

        return state