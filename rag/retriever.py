"""
rag/retriever.py — Session-Aware FAISS Retriever Store

Maintains per-session FAISS vector stores in memory.
Provides add / retrieve / analytics / clear APIs.

For WhatsApp files, also stores raw parsed messages to support:
  - Direct analytics queries (no FAISS needed)
  - Entity/name/topic search with context expansion
"""

import os
import logging
from typing import Dict, List, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger("rag.retriever")

# ── In-memory session store ───────────────────────────────────────────────────
# Schema: {
#   session_id: {
#     "vectorstore":       FAISS | None,
#     "file_names":        [str],
#     "file_types":        [str],
#     "analytics":         {},    # merged analytics per file
#     "analytics_texts":   [str], # pre-formatted analytics strings
#     "raw_messages":      {},    # {file_name: [msg_dict, ...]} for WhatsApp
#     "whatsapp_analytics":{},    # {file_name: analytics_dict} for WhatsApp
#     "all_chunks":        [],    # ordered list of chunk texts (for neighbor expansion)
#     "total_chunks":      int
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
            "raw_messages": {},        # {file_name: [msg_dict, ...]}
            "whatsapp_analytics": {},  # {file_name: analytics_dict}
            "all_chunks": [],          # ordered chunk texts for neighbor expansion
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
    analytics_text: str,
    raw_messages: Optional[List[Dict]] = None,  # WhatsApp parsed messages
) -> int:
    """
    Embed and index documents for a session.
    If the session already has a FAISS index, the new documents are merged in.
    Returns the total chunk count after indexing.

    For WhatsApp files, raw_messages stores the full parsed message list
    so entity_search() and query_analytics() can operate directly on it.
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

    # Store ordered chunk texts for neighbor-expansion retrieval
    for doc in docs:
        store["all_chunks"].append(doc.page_content)

    # Merge analytics
    if analytics:
        store["analytics"][file_name] = analytics

    # Store formatted analytics text
    if analytics_text:
        store["analytics_texts"].append(f"[{file_name}]\n{analytics_text}")

    # Store raw WhatsApp messages for direct analytics + entity search
    if raw_messages:
        store["raw_messages"][file_name] = raw_messages
        store["whatsapp_analytics"][file_name] = analytics
        logger.info(
            f"Session '{session_id}': stored {len(raw_messages)} raw WhatsApp messages "
            f"for '{file_name}'"
        )

    logger.info(
        f"Session '{session_id}': indexed '{file_name}' → "
        f"{chunk_count} total chunks, {len(store['file_names'])} files"
    )
    return chunk_count


def get_context(session_id: str, query: str, k: int = 3) -> str:
    """
    Retrieve relevant document chunks.
    
    For entity-related questions (e.g. Priyansu, Chetu, Friendship, Relationship):
      - Scans the entire document's chunks.
      - Finds all matching chunks.
      - Merges nearby chunks (within 1-2 chunks of each other).
      - Returns combined context (expanded with previous/next chunk).
      
    For general questions:
      - Uses FAISS similarity search to get top-k chunks.
      - If a retrieved chunk contains person names, relationship topics, or discussion topics:
        - Automatically expands context: retrieves previous + current + next chunk.
      - Capped at MAX_CONTEXT_CHARS.
    """
    import re
    MAX_CHUNK_CHARS = 1200     # per-chunk cap
    MAX_CONTEXT_CHARS = 4000   # total cap for all chunks combined

    store = _session_stores.get(session_id)
    if not store:
        return ""

    all_chunks = store.get("all_chunks", [])

    # ── Helpers ───────────────────────────────────────────────────────────────
    def is_entity_query(q_str: str) -> bool:
        q_low = q_str.lower()
        entity_keywords = ["priyansu", "chetu", "friendship", "relationship"]
        if any(kw in q_low for kw in entity_keywords):
            return True
        # Capitalized words of length 4+ (likely names)
        for w in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", q_str):
            if w.lower() not in {
                "what", "when", "where", "which", "this", "that", "they", "them",
                "their", "have", "does", "tell", "show", "give", "find", "about",
                "with", "from", "between", "discuss", "explain", "describe", "there",
                "here", "then", "once", "upon"
            }:
                return True
        return False

    def should_expand_context(text: str) -> bool:
        keywords = [
            "relationship", "friendship", "friend", "gf", "bf", "dating", "love", "marry", "marriage",
            "fight", "argument", "discuss", "discussion", "said", "told", "plan", "deploy", "project",
            "talk", "chat", "meet", "meeting", "school", "college", "career", "job", "work"
        ]
        
        # Build set of name parts from participants + known entities
        name_parts = {"priyansu", "chetu"}
        for f_analytics in store.get("analytics", {}).values():
            participants = f_analytics.get("participants", [])
            for p in participants:
                for part in p.split():
                    clean_part = re.sub(r"[^a-zA-Z]", "", part).lower()
                    if len(clean_part) >= 3:
                        name_parts.add(clean_part)

        # Clean text by removing metadata prefix of each line (e.g. "[date time] Sender:")
        cleaned_lines = []
        for line in text.splitlines():
            parts = line.split(": ", 1)
            if len(parts) > 1:
                cleaned_lines.append(parts[1])
            else:
                cleaned_lines.append(line)
        cleaned_text = "\n".join(cleaned_lines)
        
        text_lower = cleaned_text.lower()
        if any(kw in text_lower for kw in keywords):
            return True
            
        # Check for participant or entity names in message body
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text_lower)
        if any(w in name_parts for w in words):
            return True
            
        return False

    # ── 1. Smart Retrieval for Entity-Related Questions ───────────────────────
    if is_entity_query(query) and all_chunks:
        # Search entire document for matching terms
        stop_words = {"who", "what", "when", "where", "why", "how", "the", "and", "for",
                      "are", "was", "were", "did", "does", "tell", "show", "about",
                      "this", "that", "with", "from", "said", "have", "explain", "discuss"}
        terms = [
            w.lower() for w in re.findall(r"\b\w{3,}\b", query)
            if w.lower() not in stop_words
        ]
        
        # Find all matching chunk indices
        matching_indices = []
        for idx, text in enumerate(all_chunks):
            if any(term in text.lower() for term in terms):
                matching_indices.append(idx)
        
        if matching_indices:
            # Group nearby matches (merge if they are within 2 indices of each other)
            groups = []
            current_group = [matching_indices[0]]
            for idx in matching_indices[1:]:
                if idx - current_group[-1] <= 2:
                    current_group.append(idx)
                else:
                    groups.append(current_group)
                    current_group = [idx]
            groups.append(current_group)
            
            output_sections = []
            included_texts = set()
            total_chars = 0
            
            for group_idx, group in enumerate(groups, 1):
                if total_chars >= MAX_CONTEXT_CHARS:
                    break
                
                # Retrieve previous + group chunks + next chunk
                start_idx = max(0, group[0] - 1)
                end_idx = min(len(all_chunks) - 1, group[-1] + 1)
                
                section_lines = [f"[Smart Retrieval Group {group_idx} (Chunks {start_idx+1}–{end_idx+1})]"]
                for ci in range(start_idx, end_idx + 1):
                    text = all_chunks[ci].strip()
                    if text and text not in included_texts:
                        included_texts.add(text)
                        
                        if ci in group:
                            label = "matched chunk"
                        elif ci == start_idx:
                            label = "previous context"
                        else:
                            label = "next context"
                            
                        if len(text) > MAX_CHUNK_CHARS:
                            text = text[:MAX_CHUNK_CHARS] + "..."
                        section_lines.append(f"  [{label}]\n{text}")
                
                section = "\n".join(section_lines)
                output_sections.append(section)
                total_chars += len(section)
            
            result = "\n\n---\n\n".join(output_sections)
            logger.info(
                f"Smart Retrieval: query='{query}' matched {len(matching_indices)} chunks "
                f"→ {len(output_sections)} sections, {total_chars} chars"
            )
            return result

    # ── 2. Standard FAISS Retrieval with Conditional Context Expansion ───────
    if store["vectorstore"] is None:
        return ""

    try:
        results: List[Document] = store["vectorstore"].similarity_search(query, k=k)
        if not results:
            logger.info(f"Retrieval: no FAISS results for session '{session_id}'")
            return ""

        included_texts = set()
        output_sections = []
        total_chars = 0

        for match_idx, doc in enumerate(results, 1):
            if total_chars >= MAX_CONTEXT_CHARS:
                break

            source = doc.metadata.get("source", "unknown")
            source_name = os.path.basename(source) if source else "unknown"
            chunk_idx = doc.metadata.get("chunk_index", None)

            # Determine if this chunk meets the criteria for expansion
            expand = should_expand_context(doc.page_content)
            
            neighbor_texts = []
            if expand and chunk_idx is not None and all_chunks:
                # Expand context: retrieve previous + current + next chunk
                prev_idx = chunk_idx - 1
                next_idx = chunk_idx + 1
                for ci in [prev_idx, chunk_idx, next_idx]:
                    if 0 <= ci < len(all_chunks):
                        text = all_chunks[ci].strip()
                        if text and text not in included_texts:
                            included_texts.add(text)
                            label = (
                                "previous context" if ci == prev_idx
                                else ("next context" if ci == next_idx else "matched chunk")
                            )
                            neighbor_texts.append((ci, label, text))
            else:
                # No expansion (or no metadata) — retrieve only current matched chunk
                text = doc.page_content.strip()
                if text not in included_texts:
                    included_texts.add(text)
                    neighbor_texts.append((chunk_idx or 0, "matched chunk", text))

            if not neighbor_texts:
                continue

            section_lines = [f"[Match {match_idx} from '{source_name}']"]
            for ci, label, text in neighbor_texts:
                if len(text) > MAX_CHUNK_CHARS:
                    text = text[:MAX_CHUNK_CHARS] + "..."
                section_lines.append(f"  [{label}]\n{text}")

            section = "\n".join(section_lines)
            output_sections.append(section)
            total_chars += len(section)

        result = "\n\n---\n\n".join(output_sections)
        logger.info(
            f"Retrieval: session='{session_id}' {len(results)} FAISS hits "
            f"→ {len(output_sections)} sections, {total_chars} chars (conditional expansion)"
        )
        return result

    except Exception as e:
        logger.error(f"Retrieval failed for session '{session_id}': {e}")
        return ""


def get_whatsapp_context(session_id: str, query: str) -> Dict[str, Any]:
    """
    Smart WhatsApp context router. Classifies query intent and returns
    the appropriate context:

      "analytics" intent  → Direct answer from pre-computed analytics
      "entity" intent     → Full-text entity/name search with context expansion
      "vector" intent     → Standard FAISS retrieval (handled separately)

    Returns:
      {
        "route":           str  ("analytics" | "entity" | "vector"),
        "direct_answer":   str  (non-empty for analytics/entity routes),
        "context_for_llm": str  (additional context to inject into LLM)
      }
    """
    from rag.whatsapp import classify_whatsapp_query, query_analytics, entity_search

    store = _session_stores.get(session_id)
    if not store:
        return {"route": "vector", "direct_answer": "", "context_for_llm": ""}

    # Collect WhatsApp messages and analytics across all files in this session
    all_messages: List[Dict] = []
    merged_analytics: Dict[str, Any] = {}

    for file_name, msgs in store.get("raw_messages", {}).items():
        all_messages.extend(msgs)

    # Merge analytics from all WhatsApp files
    for file_name, analytics in store.get("whatsapp_analytics", {}).items():
        for key, val in analytics.items():
            if key == "total_messages":
                merged_analytics[key] = merged_analytics.get(key, 0) + val
            elif key == "message_counts_per_sender":
                existing = merged_analytics.get(key, {})
                for sender, count in val.items():
                    existing[sender] = existing.get(sender, 0) + count
                merged_analytics[key] = existing
            elif key == "keyword_counts":
                existing = merged_analytics.get(key, {})
                for kw, count in val.items():
                    existing[kw] = existing.get(kw, 0) + count
                merged_analytics[key] = existing
            elif key == "sender_keyword_counts":
                existing = merged_analytics.get(key, {})
                for sender, kw_dict in val.items():
                    if sender not in existing:
                        existing[sender] = {}
                    for kw, count in kw_dict.items():
                        existing[sender][kw] = existing[sender].get(kw, 0) + count
                merged_analytics[key] = existing
            elif key == "top_emojis":
                # Merge emoji counters
                from collections import Counter as _Counter
                existing_c = _Counter(dict(merged_analytics.get(key, [])))
                new_c = _Counter(dict(val))
                existing_c.update(new_c)
                merged_analytics[key] = existing_c.most_common(10)
            elif key not in merged_analytics:
                merged_analytics[key] = val

    # Rebuild derived fields after merge
    if merged_analytics.get("message_counts_per_sender"):
        sorted_counts = sorted(
            merged_analytics["message_counts_per_sender"].items(),
            key=lambda x: -x[1]
        )
        merged_analytics["most_active_sender"] = sorted_counts[0] if sorted_counts else ("N/A", 0)
        merged_analytics["participants"] = [s for s, _ in sorted_counts]
        merged_analytics["total_participants"] = len(sorted_counts)

    if not all_messages and not merged_analytics:
        return {"route": "vector", "direct_answer": "", "context_for_llm": ""}

    route = classify_whatsapp_query(query)
    logger.info(f"[WhatsApp router] query='{query[:60]}' → route='{route}'")

    if route == "analytics" and merged_analytics:
        answer = query_analytics(merged_analytics, query)
        return {
            "route": "analytics",
            "direct_answer": answer,
            "context_for_llm": answer
        }

    elif route == "entity" and all_messages:
        context = entity_search(all_messages, query, context_window=5, max_results=6)
        return {
            "route": "entity",
            "direct_answer": "",
            "context_for_llm": context
        }

    else:
        return {"route": "vector", "direct_answer": "", "context_for_llm": ""}


def has_whatsapp(session_id: str) -> bool:
    """Returns True if this session has WhatsApp data (raw messages stored)."""
    store = _session_stores.get(session_id)
    return bool(store and store.get("raw_messages"))


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
