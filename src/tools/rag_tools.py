from langchain.tools import tool
from src.rag.retriever import load_vectordb

_vectordb = None


def _get_vectordb():
    global _vectordb
    if _vectordb is None:
        _vectordb = load_vectordb()
    return _vectordb


@tool
def tool_search_knowledge_base(query: str) -> str:
    """Search the financial knowledge base for domain knowledge.

    Use this for ETF compositions, regulations, company backgrounds,
    or any information that requires retrieved context rather than real-time data.

    Args:
        query: Natural language query in Chinese or English
    """
    try:
        docs = _get_vectordb().similarity_search(query, k=4)
        if not docs:
            return "No relevant information found in the knowledge base."

        lines = [f"Knowledge base results for '{query}':\n"]
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            lines.append(f"[{i}] (Source: {source})\n{doc.page_content}\n")
        return "\n".join(lines).strip()
    except Exception as e:
        return f"Error searching knowledge base: {e}"
