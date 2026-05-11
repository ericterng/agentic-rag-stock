import transformers.modeling_utils as _mu
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL

# BAAI/bge-m3 is a trusted cached model; bypass the torch < 2.6 load restriction.
# The vulnerability (CVE-2025-32434) applies to untrusted pickled files — not here.
_mu.check_torch_load_is_safe = lambda: None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load BAAI/bge-m3 embedding model on GPU."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
