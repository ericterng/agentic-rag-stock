from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.rag.embedder import get_embeddings
from src.config import VECTORDB_PATH


def build_vectordb(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = get_embeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORDB_PATH,
    )
    print(f"Vector DB saved to {VECTORDB_PATH}")
    return vectordb


def load_vectordb() -> Chroma:
    return Chroma(
        persist_directory=VECTORDB_PATH,
        embedding_function=get_embeddings(),
    )


def retrieve(query: str, vectordb: Chroma, k: int = 4) -> list[Document]:
    return vectordb.similarity_search(query, k=k)
