import json
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

TOOLS_ALL = [
    tool_get_stock_history,
    tool_get_fundamental_data,
    tool_plot_stock_chart,
    tool_search_financial_news,
    tool_search_knowledge_base,
]
TOOLS = TOOLS_ALL  # keep backward compat with week3 notebook
_ALL_TOOL_NAMES = {t.name for t in TOOLS_ALL}

def _build_tool_map(tools: list) -> dict:
    return {t.name: t for t in tools}

def _build_tool_descriptions(tools: list) -> str:
    return "\n".join(f"- {t.name}: {t.description.split(chr(10))[0]}" for t in tools)

_TOOL_INPUT_FORMATS_ALL = """Tool input formats:
- tool_get_stock_history: <ticker>, <period>  (example: 2330.TW, 3mo)
- tool_plot_stock_chart: <ticker>, <period>  (example: 2330.TW, 3mo)
- tool_get_fundamental_data: <ticker>  (example: 2330.TW)
- tool_search_financial_news: <ticker>  (example: 2330.TW)
- tool_search_knowledge_base: <query>  (example: 半導體 AI 需求)

Valid periods: 1mo, 3mo, 6mo, 1y."""

_TOOL_INPUT_FORMATS_NO_RAG = """Tool input formats:
- tool_get_stock_history: <ticker>, <period>  (example: 2330.TW, 3mo)
- tool_plot_stock_chart: <ticker>, <period>  (example: 2330.TW, 3mo)
- tool_get_fundamental_data: <ticker>  (example: 2330.TW)
- tool_search_financial_news: <ticker>  (example: 2330.TW)

Valid periods: 1mo, 3mo, 6mo, 1y."""

# ── ReAct prompt ─────────────────────────────────────────────────────────────

_REACT_TEMPLATE_TOOLS = """You are a financial analysis assistant. Always use the available tools to look up real data before answering.
Respond in the same language as the user.
Do not provide guaranteed returns, guaranteed price movement, or a single stock that is certain to rise.
If the user asks for a guaranteed investment outcome, refuse without using tools.
If the question explicitly asks to use the knowledge base, use tool_search_knowledge_base once and then answer from that Observation.
Do not call extra tools after you already have enough evidence to answer.

Available tools:
{tool_descriptions}

{tool_input_formats}

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

_REACT_TEMPLATE_LLM_ONLY = """You are a financial analysis assistant.
Respond in the same language as the user.
Do not provide guaranteed returns or certain investment outcomes.
If the question is unrelated to finance or investing, say you can only help with financial questions.

Previous conversation:
{history}

User question: {input}

Thought: """


# ── Graph state ───────────────────────────────────────────────────────────────

class ReActState(TypedDict):
    input: str
    history: str
    scratchpad: str
    output: str
    iterations: int


# ── Helpers ───────────────────────────────────────────────────────────────────

_ACTION_RE = re.compile(r"Action:\s*(.+)", re.IGNORECASE)
_INPUT_RE  = re.compile(r"Action Input:\s*(.+)", re.IGNORECASE)
_FINAL_RE  = re.compile(r"Final Answer:\s*(.+)", re.DOTALL | re.IGNORECASE)
_OBSERVATION_RE = re.compile(r"Observation:\s*(.*?)(?=\nThought:|\nAction:|\nFinal Answer:|\Z)", re.IGNORECASE | re.DOTALL)

_PERIOD_ALIASES = {
    "1 month": "1mo",
    "one month": "1mo",
    "1m": "1mo",
    "1mo": "1mo",
    "3 month": "3mo",
    "3 months": "3mo",
    "three months": "3mo",
    "3m": "3mo",
    "3mo": "3mo",
    "6 month": "6mo",
    "6 months": "6mo",
    "six months": "6mo",
    "6m": "6mo",
    "6mo": "6mo",
    "1 year": "1y",
    "one year": "1y",
    "1yr": "1y",
    "1y": "1y",
}


def _parse_action(text: str, allowed_tool_names: set = None):
    """Return the last (tool_name, tool_input) pair in text, only if it's an ALLOWED tool."""
    actions = _ACTION_RE.findall(text)
    inputs = _INPUT_RE.findall(text)
    if not actions or not inputs:
        return None, None

    tool_name = actions[-1].strip()
    tool_input = inputs[-1].strip()

    # Use the filtered active set if provided, fallback to the global list if not
    valid_set = allowed_tool_names if allowed_tool_names is not None else _ALL_TOOL_NAMES
    if tool_name in valid_set:
        return tool_name, tool_input
    return None, None


def _parse_final(text: str):
    m = _FINAL_RE.search(text)
    return m.group(1).strip() if m else None


def _extract_observations(text: str) -> list[str]:
    return [item.strip() for item in _OBSERVATION_RE.findall(text) if item.strip()]


def _is_low_quality_final(final: str, user_input: str) -> bool:
    lowered = final.lower()
    bad_phrases = [
        "insert observation",
        "see above",
        "these news articles",
        "these news factors",
        "we also got",
        "please continue",
        "your answer starts here",
    ]
    if any(phrase in lowered for phrase in bad_phrases):
        return True

    if _contains_cjk(user_input) and not _contains_cjk(final):
        return True

    numeric_or_data_question = any(
        token in user_input
        for token in ["股價", "收盤價", "高低點", "漲跌幅", "本益比", "股息", "價格", "基本面", "新聞"]
    )
    if numeric_or_data_question and len(final) < 80:
        return True

    return False


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _is_guaranteed_investment_request(text: str) -> bool:
    lowered = text.lower()
    chinese_signals = ["保證", "一定", "穩賺", "漲停", "明天會漲", "直接告訴我買"]
    english_signals = ["guarantee", "guaranteed", "sure profit", "certain to rise", "will definitely rise"]
    return any(signal in text for signal in chinese_signals) or any(signal in lowered for signal in english_signals)


def _is_out_of_scope_request(text: str) -> bool:
    lowered = text.lower()
    out_of_scope_signals = ["貪吃蛇", "遊戲", "寫程式", "snake game", "write a game", "write code"]
    finance_signals = ["股票", "投資", "金融", "股價", "etf", "fundamental", "market", "stock"]
    return any(signal in lowered for signal in out_of_scope_signals) and not any(signal in lowered for signal in finance_signals)


def _is_knowledge_base_question(text: str) -> bool:
    lowered = text.lower()
    return "知識庫" in text or "knowledge base" in lowered


def _guardrail_response(user_input: str) -> str | None:
    if _is_guaranteed_investment_request(user_input):
        if _contains_cjk(user_input):
            return (
                "我不能保證或推薦明天一定會漲停的股票，也不能把不確定的市場結果包裝成確定建議。"
                "我可以改為協助你根據股價、基本面、新聞與風險因素做資料化分析。"
            )
        return (
            "I cannot guarantee or recommend a stock that will rise tomorrow. "
            "I can help analyze stocks using price data, fundamentals, news, and risk factors instead."
        )

    if _is_out_of_scope_request(user_input):
        if _contains_cjk(user_input):
            return "我只能協助金融、投資與市場分析相關問題。"
        return "I can only help with financial and investment questions."

    return None


def _normalise_period(value: str, default: str = "3mo") -> str:
    value = value.strip().lower()
    return _PERIOD_ALIASES.get(value, value if value in {"1mo", "3mo", "6mo", "1y"} else default)


def _coerce_tool_input(tool_name: str, tool_input: str):
    """Convert text ReAct inputs into LangChain tool arguments.

    Small local LLMs often emit inputs like "2330.TW, 3mo". LangChain will
    otherwise pass that whole string as the first argument, so we normalize
    common financial-tool inputs before invoking the tool.
    """
    raw = tool_input.strip().strip("`")
    if raw.startswith("{") and raw.endswith("}"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    if tool_name in {"tool_get_stock_history", "tool_plot_stock_chart"}:
        parts = [p.strip() for p in re.split(r"[,;\n]+", raw) if p.strip()]
        ticker = parts[0] if parts else raw
        period = _normalise_period(parts[1]) if len(parts) > 1 else "3mo"
        return {"ticker": ticker, "period": period}

    if tool_name == "tool_get_fundamental_data":
        ticker = re.split(r"[,;\n]+", raw)[0].strip()
        return {"ticker": ticker}

    if tool_name == "tool_search_financial_news":
        ticker = re.split(r"[,;\n]+", raw)[0].strip()
        return {"query": ticker}

    if tool_name == "tool_search_knowledge_base":
        return {"query": raw}

    return raw


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


def make_agent_node(llm, prompt_template: PromptTemplate, tool_descriptions: str = "", tool_input_formats: str = ""):
    def agent_node(state: ReActState) -> ReActState:
        fmt_kwargs = {"history": state["history"], "input": state["input"]}
        if tool_descriptions:
            fmt_kwargs["tool_descriptions"] = tool_descriptions
            fmt_kwargs["tool_input_formats"] = tool_input_formats
        full_prompt = prompt_template.format(**fmt_kwargs)
        if state["scratchpad"]:
            full_prompt += " " + state["scratchpad"] + "\nThought:"

        raw = llm.invoke(full_prompt)
        new_text = "Thought:" + raw if not raw.startswith("Thought") else raw
        # Stop before any hallucinated Observation so the real tool output goes in
        new_text = _trim_at_observation(new_text)
        state["scratchpad"] += "\n" + new_text
        return state
    return agent_node


def guardrail_node(state: ReActState) -> ReActState:
    final = _guardrail_response(state["input"])
    if final:
        state["output"] = final
        state["scratchpad"] += (
            "\nThought: This request should be refused by policy/domain guardrails."
            f"\nFinal Answer: {final}"
        )
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
    if "Observation:" in state["scratchpad"]:
        return "force_final"
    return "end"


def make_should_continue(max_iterations: int = 6, allowed_tool_names: set = None):
    def _should_continue(state: ReActState) -> str:
        if state.get("iterations", 0) >= max_iterations:
            return "force_final"
        final = _parse_final(state["scratchpad"])
        if final:
            if _is_low_quality_final(final, state["input"]):
                return "force_final"
            state["output"] = final
            return "end"
        if _is_knowledge_base_question(state["input"]) and state.get("iterations", 0) >= 1:
            return "force_final"
        tool_name, _ = _parse_action(state["scratchpad"], allowed_tool_names)
        if tool_name:
            return "tool"
        if "Observation:" in state["scratchpad"]:
            return "force_final"
        return "end"
    return _should_continue


def guardrail_route(state: ReActState) -> str:
    return "end" if state.get("output") else "agent"


_THOUGHT_RE = re.compile(r"Thought:\s*(.+)", re.IGNORECASE)


def make_force_final_node(llm):
    """Generate Final Answer directly from accumulated observations."""
    def force_final_node(state: ReActState) -> ReActState:
        language_instruction = (
            "Answer in Traditional Chinese. Use clear bullet points when listing numbers."
            if _contains_cjk(state["input"])
            else "Answer in English."
        )
        observations = _extract_observations(state["scratchpad"])
        observation_text = "\n\n".join(observations) if observations else "No Observation was available."
        prompt = (
            f"User question: {state['input']}\n"
            + "Observed tool results:\n"
            + observation_text
            + f"\n{language_instruction}\n"
            + "Create the final answer using ONLY the Observed tool results above.\n"
            + "Rules:\n"
            + "1. Include every numeric field requested by the user if it appears in the observed tool results.\n"
            + "2. If the observed tool results say a value is N/A, explicitly say the tool returned N/A; do not guess.\n"
            + "3. Do not use placeholders such as 'insert observation', 'see above', or 'these news articles'.\n"
            + "4. Do not mention a chart path unless that exact path appears in the observed tool results.\n"
            + "5. For comparison questions, compare only tickers with observed data and state if data is missing.\n"
            + "6. For news questions, summarize specific observed headlines or factors instead of saying 'these articles'.\n"
            + "7. Keep the answer concise but complete.\n"
            + "Final Answer:"
        )
        raw = llm.invoke(prompt)
        final = raw.strip()
        for marker in ["\nUser question:", "\nThought:", "\nAction:", "\nObservation:"]:
            if marker in final:
                final = final.split(marker, 1)[0].strip()
        state["output"] = final
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

def create_agent_graph(
    tokenizer,
    model,
    checkpointer: MemorySaver = None,
    max_new_tokens: int = 512,
    max_iterations: int = 6,
    ablation_mode: str = "full_suite",
):
    # Resolve tools and prompt based on mode
    if ablation_mode == "llm_only":
        active_tools = []
        prompt = PromptTemplate.from_template(_REACT_TEMPLATE_LLM_ONLY)
        tool_desc = ""
        tool_fmt = ""
    elif ablation_mode == "llm_tools":
        active_tools = [t for t in TOOLS_ALL if t.name != "tool_search_knowledge_base"]
        prompt = PromptTemplate.from_template(_REACT_TEMPLATE_TOOLS)
        tool_desc = _build_tool_descriptions(active_tools)
        tool_fmt = _TOOL_INPUT_FORMATS_NO_RAG
    else:  # "full_suite"
        active_tools = TOOLS_ALL
        prompt = PromptTemplate.from_template(_REACT_TEMPLATE_TOOLS)
        tool_desc = _build_tool_descriptions(active_tools)
        tool_fmt = _TOOL_INPUT_FORMATS_ALL

    active_tool_map = _build_tool_map(active_tools)
    active_tool_names = set(active_tool_map)

    llm = create_llm(tokenizer, model, max_new_tokens=max_new_tokens)

    # ── nodes (pass active_tool_map into tool_node via closure) ──
    def _tool_node(state: ReActState) -> ReActState:
        tool_name, tool_input = _parse_action(state["scratchpad"], active_tool_names)
        if tool_name and tool_name in active_tool_map:
            try:
                result = active_tool_map[tool_name].invoke(_coerce_tool_input(tool_name, tool_input))
            except Exception as e:
                result = f"Tool error: {e}"
        else:
            result = f"Unknown or disabled tool: {tool_name}"
        state["scratchpad"] += f"\nObservation: {result}"
        state["iterations"] = state.get("iterations", 0) + 1
        return state

    builder = StateGraph(ReActState)
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("agent", make_agent_node(llm, prompt, tool_desc, tool_fmt))
    builder.add_node("tool", _tool_node)           # ← local closure, not global tool_node
    builder.add_node("force_final", make_force_final_node(llm))
    builder.add_node("finalize", finalize_node)

    if ablation_mode == "full_suite":
        builder.set_entry_point("guardrail")
    else:
        builder.set_entry_point("agent")
    if ablation_mode == "full_suite":
        builder.add_conditional_edges("guardrail", guardrail_route, {"agent": "agent", "end": "finalize"})
    if ablation_mode == "llm_only":
        builder.add_edge("agent", "finalize")
    else:
        builder.add_conditional_edges("agent", make_should_continue(max_iterations, active_tool_names), {
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
