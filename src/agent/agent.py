import re
from typing import Annotated, TypedDict
from transformers import pipeline as hf_pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.tools.stock_tools import (
    tool_get_stock_history,
    tool_get_fundamental_data,
    tool_plot_stock_chart,
)
from src.tools.news_tools import tool_search_financial_news
from src.tools.rag_tools import tool_search_knowledge_base

# ── Tool registry ────────────────────────────────────────────────────────────

TOOLS = [
    tool_get_stock_history,
    tool_get_fundamental_data,
    tool_plot_stock_chart,
    tool_search_financial_news,
    tool_search_knowledge_base,
]

_TOOL_MAP = {t.name: t for t in TOOLS}

_TOOL_DESCRIPTIONS = "\n".join(
    f"- {t.name}: {t.description.split(chr(10))[0]}" for t in TOOLS
)

# ── ReAct prompt ─────────────────────────────────────────────────────────────

_REACT_TEMPLATE = """You are a financial analysis assistant. Always use the available tools to look up real data before answering.
Respond in the same language as the user.

Available tools:
{tool_descriptions}

Format your response EXACTLY like this (no deviations):

Thought: <think about what to do>
Action: <one of the tool names above>
Action Input: <input for the tool>
Observation: <tool result will be inserted here>
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer.
Final Answer: <your answer to the user>

If the question is unrelated to finance or investing, respond:
Thought: This question is outside my domain.
Final Answer: I can only help with financial and investment questions.

Previous conversation:
{history}

User question: {input}

Begin:
Thought:"""


# ── Graph state ───────────────────────────────────────────────────────────────

class ReActState(TypedDict):
    input: str
    history: str
    scratchpad: str       # accumulated Thought/Action/Observation text
    output: str           # final answer
    iterations: int


# ── Helpers ───────────────────────────────────────────────────────────────────

_ACTION_RE = re.compile(r"Action:\s*(.+)", re.IGNORECASE)
_INPUT_RE  = re.compile(r"Action Input:\s*(.+)", re.IGNORECASE)
_FINAL_RE  = re.compile(r"Final Answer:\s*(.+)", re.DOTALL | re.IGNORECASE)


def _parse_action(text: str):
    """Return the last (tool_name, tool_input) pair in text, only if it's a known tool."""
    actions = _ACTION_RE.findall(text)
    inputs = _INPUT_RE.findall(text)
    if not actions or not inputs:
        return None, None
    # Only look at the very last Action/Input pair
    tool_name = actions[-1].strip()
    tool_input = inputs[-1].strip()
    if tool_name in _TOOL_MAP:
        return tool_name, tool_input
    return None, None


def _parse_final(text: str):
    m = _FINAL_RE.search(text)
    return m.group(1).strip() if m else None


# ── LLM factory ──────────────────────────────────────────────────────────────

def create_llm(tokenizer, model, max_new_tokens: int = 512, temperature: float = 0.1):
    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        return_full_text=False,
        repetition_penalty=1.1,
    )
    return HuggingFacePipeline(pipeline=pipe)


# ── Graph nodes ───────────────────────────────────────────────────────────────

def _trim_at_observation(text: str) -> str:
    """Remove hallucinated 'Observation:' and anything after it."""
    idx = text.find("\nObservation:")
    if idx == -1:
        idx = text.lower().find("observation:")
    return text[:idx].rstrip() if idx != -1 else text


def make_agent_node(llm, prompt_template: PromptTemplate):
    def agent_node(state: ReActState) -> ReActState:
        full_prompt = prompt_template.format(
            tool_descriptions=_TOOL_DESCRIPTIONS,
            history=state["history"],
            input=state["input"],
        )
        if state["scratchpad"]:
            full_prompt += " " + state["scratchpad"] + "\nThought:"

        raw = llm.invoke(full_prompt)
        new_text = "Thought:" + raw if not raw.startswith("Thought") else raw
        # Stop before any hallucinated Observation so the real tool output goes in
        new_text = _trim_at_observation(new_text)
        state["scratchpad"] += "\n" + new_text
        return state
    return agent_node


def tool_node(state: ReActState) -> ReActState:
    tool_name, tool_input = _parse_action(state["scratchpad"])
    if tool_name and tool_name in _TOOL_MAP:
        try:
            tool = _TOOL_MAP[tool_name]
            # Try dict input first (for tools with named args)
            try:
                result = tool.invoke(tool_input)
            except Exception:
                result = tool.invoke({"query": tool_input})
        except Exception as e:
            result = f"Tool error: {e}"
    else:
        result = f"Unknown tool: {tool_name}"

    observation = f"\nObservation: {result}"
    state["scratchpad"] += observation
    state["iterations"] = state.get("iterations", 0) + 1
    return state


def should_continue(state: ReActState) -> str:
    if state.get("iterations", 0) >= 6:
        return "force_final"
    final = _parse_final(state["scratchpad"])
    if final:
        state["output"] = final
        return "end"
    tool_name, _ = _parse_action(state["scratchpad"])
    if tool_name:
        return "tool"
    # No tool and no Final Answer — if we have observations, force a final answer
    if "Observation:" in state["scratchpad"]:
        return "force_final"
    return "end"


_THOUGHT_RE = re.compile(r"Thought:\s*(.+)", re.IGNORECASE)


def make_force_final_node(llm):
    """Generate Final Answer directly from accumulated observations."""
    def force_final_node(state: ReActState) -> ReActState:
        prompt = (
            state["scratchpad"]
            + "\nBased on the Observation above, answer the user's question concisely.\nFinal Answer:"
        )
        raw = llm.invoke(prompt)
        state["output"] = raw.split("\n")[0].strip()
        return state
    return force_final_node


def finalize_node(state: ReActState) -> ReActState:
    if not state.get("output"):
        final = _parse_final(state["scratchpad"])
        if final:
            state["output"] = final
        else:
            # Fall back: use the last meaningful Thought as the answer
            thoughts = _THOUGHT_RE.findall(state["scratchpad"])
            meaningful = [t.strip() for t in thoughts if "final answer" not in t.lower() and len(t.strip()) > 20]
            state["output"] = meaningful[-1] if meaningful else state["scratchpad"].strip()
    return state


# ── Public API ────────────────────────────────────────────────────────────────

def create_agent_graph(tokenizer, model, checkpointer: MemorySaver = None):
    """Build a text-based ReAct LangGraph agent.

    Usage:
        graph, memory = create_agent_graph(tokenizer, model)
        config = {"configurable": {"thread_id": "my-thread"}}
        result = graph.invoke({"input": "...", "history": "", "scratchpad": "", "output": "", "iterations": 0}, config)
        print(result["output"])
    """
    llm = create_llm(tokenizer, model)
    prompt = PromptTemplate.from_template(_REACT_TEMPLATE)

    builder = StateGraph(ReActState)
    builder.add_node("agent", make_agent_node(llm, prompt))
    builder.add_node("tool", tool_node)
    builder.add_node("force_final", make_force_final_node(llm))
    builder.add_node("finalize", finalize_node)

    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", should_continue, {
        "tool": "tool",
        "force_final": "force_final",
        "end": "finalize",
    })
    builder.add_edge("tool", "agent")
    builder.add_edge("force_final", "finalize")
    builder.add_edge("finalize", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = builder.compile(checkpointer=checkpointer)
    return graph, checkpointer
