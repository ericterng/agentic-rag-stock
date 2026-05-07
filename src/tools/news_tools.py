import yfinance as yf
from langchain_core.tools import tool


def get_stock_news(ticker: str, max_items: int = 5) -> list[dict]:
    stock = yf.Ticker(ticker)
    results = []
    for item in stock.news[:max_items]:
        content = item.get("content", {})
        results.append({
            "title": content.get("title", ""),
            "summary": content.get("summary", ""),
            "published": content.get("pubDate", ""),
            "url": content.get("canonicalUrl", {}).get("url", ""),
        })
    return results


@tool
def tool_search_financial_news(query: str) -> str:
    """Search for recent financial news by stock ticker.

    For broad keyword queries (e.g. '半導體', 'AI'), use tool_search_knowledge_base instead.

    Args:
        query: Stock ticker symbol (e.g. '2330.TW', '0050.TW', 'TSMC')
    """
    try:
        news = get_stock_news(query)
        if not news:
            return f"No news found for: {query}"

        lines = [f"Latest news for '{query}':\n"]
        for i, item in enumerate(news, 1):
            lines.append(f"{i}. {item['title']}")
            if item["summary"]:
                lines.append(f"   {item['summary'][:200]}...")
            lines.append(f"   Published: {item['published']}\n")
        return "\n".join(lines).strip()
    except Exception as e:
        return f"Error searching news for {query}: {e}"
