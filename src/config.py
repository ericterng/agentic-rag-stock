from pathlib import Path
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VECTORDB_DIR = DATA_DIR / "vectordb"
YFINANCE_CACHE_DIR = Path(
    os.getenv(
        "YFINANCE_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "grounded_financial_llm_agent_yfinance_cache"),
    )
)
OUTPUTS_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"

for d in [RAW_DATA_DIR, VECTORDB_DIR, YFINANCE_CACHE_DIR, CHARTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
EMBEDDING_MODEL = "BAAI/bge-m3"
VECTORDB_PATH = os.getenv("VECTORDB_PATH", str(VECTORDB_DIR))
