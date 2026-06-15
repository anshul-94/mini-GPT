"""
rag/embedder.py — Text Chunking + Embeddings + FAISS Index Builder

Splits Documents into chunks, embeds them with HuggingFace (default) or
OpenAI, and returns a FAISS vector store ready for retrieval.
"""

import os
import logging
from typing import List

from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

logger = logging.getLogger("rag.embedder")

# ── Chunking configuration ────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Embedding model ───────────────────────────────────────────────────────────
_embeddings_cache = None


def get_embeddings():
    """
    Returns the embeddings model (cached singleton).
    Uses HuggingFace by default. Set EMBEDDING_PROVIDER=openai in .env for OpenAI.
    """
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache

    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()

    if provider == "openai":
        logger.info("Using OpenAI embeddings")
        try:
            from langchain_openai import OpenAIEmbeddings
            _embeddings_cache = OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-3-small"
            )
        except ImportError:
            logger.warning("langchain-openai not installed. Falling back to HuggingFace.")
            _embeddings_cache = _load_huggingface_embeddings()
    else:
        _embeddings_cache = _load_huggingface_embeddings()

    return _embeddings_cache


def _load_huggingface_embeddings():
    """Load HuggingFace sentence-transformers embeddings (free, local)."""
    logger.info("Loading HuggingFace embeddings: sentence-transformers/all-MiniLM-L6-v2")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    except ImportError:
        # Fallback to community version
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    Preserves original metadata and adds chunk_index.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    chunks = splitter.split_documents(docs)
    logger.info(f"Chunked {len(docs)} documents → {len(chunks)} chunks")
    return chunks


def build_faiss_index(docs: List[Document]) -> FAISS:
    """
    Chunk documents and build a FAISS index.
    Returns the FAISS vector store.
    """
    if not docs:
        raise ValueError("No documents provided to build FAISS index.")

    chunks = chunk_documents(docs)

    if not chunks:
        raise ValueError("Document chunking produced no text chunks.")

    embeddings = get_embeddings()
    logger.info(f"Building FAISS index from {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS index built successfully.")
    return vectorstore


def add_to_faiss_index(vectorstore: FAISS, docs: List[Document]) -> FAISS:
    """
    Add new documents to an existing FAISS index (for additive uploads).
    """
    chunks = chunk_documents(docs)
    if not chunks:
        return vectorstore
    embeddings = get_embeddings()
    new_store = FAISS.from_documents(chunks, embeddings)
    vectorstore.merge_from(new_store)
    logger.info(f"Added {len(chunks)} new chunks to existing FAISS index.")
    return vectorstore
