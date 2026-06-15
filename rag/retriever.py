"""
rag/retriever.py — Session-Aware FAISS Retriever Store

Maintains per-session FAISS vector stores in memory.
Provides add / retrieve / analytics / clear APIs.
"""

import os
import logging
from typing import Dict, List, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger("rag.retriever")

# ── In-memory session store ────────────────────────────────────────────────────
# Schema: {
#   session_id: {
#     "vectorstore": FAISS | None,
#     "file_names": [str],
#     "file_types": [str],
#     "analytics": {},          # merged analytics from all uploaded files
#     "analytics_texts": [str], # pre-formatted analytics strings per file
#     "total_chunks": int
#   }
# }
_session_stores: Dict[str, Dict[str, Any]] = {}

# Maximum number of concurrent sessions (memory guard)
MAX_SESSIONS = 200


def _init_session(session_id: str) -> None:
    """Initialize a session store if it does not exist."""
    if session_id not in _session_stores:
        _session_stores[session_id] = {
            "vectorstore": None,
            "file_names": [],
            "file_types": [],
            "analytics": {},
            "analytics_texts": [],
            "total_chunks": 0
        }


def has_documents(session_id: str) -> bool:
    """Returns True if this session has any uploaded documents."""
    store = _session_stores.get(session_id)
    return store is not None and store["vectorstore"] is not None


def add_documents(
    session_id: str,
    docs: List[Document],
    file_name: str,
    file_type: str,
    analytics: Dict[str, Any],
    analytics_text: str
) -> int:
    """
    Embed and index documents for a session.
    If the session already has a FAISS index, the new documents are merged in.
    Returns the total chunk count after indexing.
    """
    from rag.embedder import build_faiss_index, add_to_faiss_index

    # Memory guard: evict oldest sessions if over limit
    if len(_session_stores) >= MAX_SESSIONS:
        oldest = list(_session_stores.keys())[:50]
        for sid in oldest:
            _session_stores.pop(sid, None)
        logger.info(f"Evicted 50 oldest RAG sessions. Active: {len(_session_stores)}")

    _init_session(session_id)
    store = _session_stores[session_id]

    if store["vectorstore"] is None:
        store["vectorstore"] = build_faiss_index(docs)
        # Count chunks by checking vectorstore
        try:
            chunk_count = store["vectorstore"].index.ntotal
        except Exception:
            chunk_count = len(docs)
    else:
        store["vectorstore"] = add_to_faiss_index(store["vectorstore"], docs)
        try:
            chunk_count = store["vectorstore"].index.ntotal
        except Exception:
            chunk_count = store["total_chunks"] + len(docs)

    store["file_names"].append(file_name)
    store["file_types"].append(file_type)
    store["total_chunks"] = chunk_count

    # Merge analytics
    if analytics:
        store["analytics"][file_name] = analytics

    # Store formatted analytics text (will be injected into LLM context)
    if analytics_text:
        store["analytics_texts"].append(f"[{file_name}]\n{analytics_text}")

    logger.info(
        f"Session '{session_id}': indexed '{file_name}' → "
        f"{chunk_count} total chunks, {len(store['file_names'])} files"
    )
    return chunk_count


def get_context(session_id: str, query: str, k: int = 3) -> str:
    """
    Retrieve top-k relevant document chunks for the query.
    Returns a formatted string ready for LLM injection.

    Each individual chunk is capped at MAX_CHUNK_CHARS characters to prevent
    a single large chunk from flooding the context window.
    """
    MAX_CHUNK_CHARS = 800  # per-chunk cap (~200 tokens)

    store = _session_stores.get(session_id)
    if not store or store["vectorstore"] is None:
        return ""

    try:
        results: List[Document] = store["vectorstore"].similarity_search(query, k=k)
        if not results:
            logger.info(f"Retrieval: no results for session '{session_id}' query='{query[:60]}'")
            return ""

        chunks = []
        total_chars = 0
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            # Extract just the filename from full path
            source_name = os.path.basename(source) if source else "unknown"

            content = doc.page_content.strip()
            # Cap individual chunk
            if len(content) > MAX_CHUNK_CHARS:
                content = content[:MAX_CHUNK_CHARS] + "..."

            chunk_text = f"[Chunk {i} from '{source_name}']\n{content}"
            chunks.append(chunk_text)
            total_chars += len(chunk_text)

        logger.info(
            f"Retrieval: session='{session_id}' returned {len(results)}/{k} chunks "
            f"totalling {total_chars} chars"
        )
        return "\n\n---\n\n".join(chunks)

    except Exception as e:
        logger.error(f"Retrieval failed for session '{session_id}': {e}")
        return ""


def get_analytics_text(session_id: str) -> str:
    """
    Return all pre-formatted analytics strings for the session.
    Used for CSV, Excel, WhatsApp files.
    """
    store = _session_stores.get(session_id)
    if not store or not store["analytics_texts"]:
        return ""
    return "\n\n".join(store["analytics_texts"])


def get_session_info(session_id: str) -> Dict[str, Any]:
    """Return metadata about a session's uploaded files."""
    store = _session_stores.get(session_id)
    if not store:
        return {"files": [], "total_chunks": 0, "has_documents": False}
    return {
        "files": list(zip(store["file_names"], store["file_types"])),
        "file_names": store["file_names"],
        "total_chunks": store["total_chunks"],
        "has_documents": store["vectorstore"] is not None
    }


def clear_session(session_id: str) -> None:
    """Remove all RAG data for a session."""
    _session_stores.pop(session_id, None)
    logger.info(f"Cleared RAG store for session '{session_id}'")
