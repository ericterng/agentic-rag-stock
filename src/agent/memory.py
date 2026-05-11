from langgraph.checkpoint.memory import MemorySaver


def create_memory() -> MemorySaver:
    """In-memory checkpointer for LangGraph conversation history.

    Each conversation thread is identified by a thread_id passed in the
    config dict: {"configurable": {"thread_id": "<id>"}}.
    """
    return MemorySaver()
