import yfinance as yf
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from src.config import RAW_DATA_DIR

DEFAULT_TICKERS = ["0050.TW", "2330.TW", "2454.TW", "2317.TW"]


def load_yfinance_news(tickers: list[str], max_per_ticker: int = 10) -> list[Document]:
    documents = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            for item in stock.news[:max_per_ticker]:
                content = item.get("content", {})
                title = content.get("title", "")
                summary = content.get("summary", "")
                text = f"Title: {title}\nSummary: {summary}".strip()
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": "yfinance_news",
                            "ticker": ticker,
                            "date": content.get("pubDate", ""),
                        },
                    ))
        except Exception as e:
            print(f"Warning: Could not load news for {ticker}: {e}")
    return documents


def load_pdf(pdf_path: str) -> list[Document]:
    return PyPDFLoader(pdf_path).load()


def load_all_documents(tickers: list[str] = None) -> list[Document]:
    tickers = tickers or DEFAULT_TICKERS
    documents = []

    print(f"Loading yfinance news for {tickers}...")
    news_docs = load_yfinance_news(tickers)
    documents.extend(news_docs)
    print(f"  Loaded {len(news_docs)} news articles")

    for pdf_path in RAW_DATA_DIR.glob("*.pdf"):
        print(f"Loading PDF: {pdf_path.name}...")
        pdf_docs = load_pdf(str(pdf_path))
        documents.extend(pdf_docs)
        print(f"  Loaded {len(pdf_docs)} pages")

    print(f"Total documents: {len(documents)}")
    return documents
