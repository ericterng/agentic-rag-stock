from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load BAAI/bge-m3 embedding model on GPU."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
