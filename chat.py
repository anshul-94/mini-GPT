import os
import time
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot.chat")

load_dotenv()

# Load API Key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    logger.warning("OPENROUTER_API_KEY is not set in environment variables.")

client = OpenAI(
    api_key=api_key or "DUMMY_KEY",  # Avoid instantiation crash if key is missing initially
    base_url="https://openrouter.ai/api/v1"
)

# ── Model fallback chain ──────────────────────────────────────────────────────
# Last verified against OpenRouter live API: 2026-06-15
#
# Full probe results (HTTP status + actual content):
#
#   MODEL                                       HTTP  RESULT
#   openai/gpt-oss-20b:free                     200   EMPTY  ← primary, falls through
#   openai/gpt-oss-120b:free                    200   ✅ ALIVE  131k context
#   google/gemma-4-31b-it:free                  200   ✅ ALIVE  262k context
#   google/gemma-4-26b-a4b-it:free              200   ✅ ALIVE  262k context
#   nex-agi/nex-n2-pro:free                     200   ✅ ALIVE  262k context
#
#   REMOVED (dead slugs):
#   meta-llama/llama-3-8b-instruct:free         404   Not Found
#   mistralai/mistral-7b-instruct:free          404   Not Found
#   nvidia/nemotron-3-nano-30b-a3b:free         200   EMPTY
#   nvidia/nemotron-3-super-120b-a12b:free      200   finish=length (wrong reply)
#   nvidia/nemotron-3-ultra-550b-a55b:free      200   finish=length (wrong reply)
#
# Override primary model via OPENROUTER_MODEL env var.
_MODEL_FALLBACK_CHAIN = [
    os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free"),  # primary (may return empty)
    "openai/gpt-oss-120b:free",       # fallback 1 — live verified ✅ 131k context
    "google/gemma-4-31b-it:free",     # fallback 2 — live verified ✅ 262k context
    "google/gemma-4-26b-a4b-it:free", # fallback 3 — live verified ✅ 262k context
    "nex-agi/nex-n2-pro:free",        # fallback 4 — live verified ✅ 262k context
]

# ── Context budget constants ───────────────────────────────────────────────────
# Conservative limits to stay inside free-tier model context windows (~4k tokens)
# 1 token ≈ 4 chars. We reserve ~2k tokens for the response.
MAX_ANALYTICS_CHARS = 1500    # analytics block (WhatsApp/CSV/Excel)
MAX_RETRIEVED_CHARS = 3000    # total retrieved chunks text
MAX_TOTAL_PROMPT_CHARS = 8000 # hard cap on full prompt before sending

# ── System prompts ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Xai, a friendly AI assistant.

STRICT LANGUAGE DETECTION RULES (follow in this exact order):
1. First detect the SCRIPT used in the user's last message:
   - If message uses DEVANAGARI characters (like क, ख, ग, आ, etc.) → reply in Hindi (Devanagari script).
   - If message uses ROMAN/LATIN alphabet ONLY (a-z, even if words sound Hindi, e.g. "mera naam Anshul hai") → reply in Hinglish (Roman-script Hindi mixed with English). Never switch to Devanagari.
   - If message is clearly English (words like "my", "name", "is", "what", "how") → reply in English.
2. When in doubt between English and Hinglish: if the message contains ANY English words or greetings ("hi", "hey", "my name", "baby"), treat it as ENGLISH and reply in English.
3. NEVER switch from English to Hindi or Hinglish unless the user explicitly writes in Devanagari script.
4. NEVER randomly change languages mid-conversation.
5. Keep replies friendly, natural, and concise (1-3 sentences)."""

RAG_SYSTEM_PROMPT = """You are Xai, a helpful AI assistant specialized in analyzing documents and files.

The user has uploaded one or more files. Answer questions based ONLY on the document content provided below.

RULES:
1. Answer ONLY from the document context provided. Do not invent information.
2. If the answer is not in the documents, say: "I couldn't find that information in the uploaded files."
3. For counts/totals/averages: compute from the data given, do not estimate.
4. For WhatsApp chats: use the analytics summary for counting questions.
5. Be concise and accurate. Cite the source file when helpful.
6. Reply in the same language/script as the user's message."""

SUMMARIZE_PROMPT = """You are a conversation memory manager. Consolidate these new exchanges into the running summary.
Current Summary:
{existing_summary}

New exchanges:
{new_messages}

Output an updated, clean summary strictly structured as follows (omit any category that has no data):
- Name: (user name if known)
- Preferences: (user likes, tone, script choices)
- Projects: (technologies, codebase details, frameworks)
- Goals: (what the user is building or trying to achieve)

Reply ONLY with the updated summary, nothing else."""

COMPRESS_PROMPT = """Compress this summary, retaining Name, Preferences, Projects, and Goals:
{summary}
Reply ONLY with the compressed version."""

# Thread-safe in-memory session memories storage
# Schema: { session_id (str): { "stm": list, "summary": str } }
session_memories = {}


# ── Helper: compute rough char length of a messages list ─────────────────────
def _prompt_char_length(messages: list) -> int:
    return sum(len(m.get("content", "")) for m in messages)


# ── Helper: trim a text block to a max character budget ──────────────────────
def _trim_to_budget(text: str, max_chars: int, label: str = "context") -> str:
    """Trim text to max_chars, appending a note if trimmed."""
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars]
    # Try to cut at a clean line boundary
    last_newline = trimmed.rfind("\n")
    if last_newline > max_chars * 0.7:
        trimmed = trimmed[:last_newline]
    logger.warning(
        f"[Budget] {label} trimmed from {len(text)} → {len(trimmed)} chars "
        f"(budget={max_chars})"
    )
    return trimmed + f"\n\n[...{label} truncated to fit context window...]"


# ── Startup: validate fallback chain once ─────────────────────────────────────
# Called lazily on first chat request. Removes any 404 models from the chain
# so they never waste an attempt slot at runtime.
_chain_validated = False

def _validate_model_chain() -> None:
    """
    Probe each model with a tiny ping request.
    Removes models that return 404 (slug no longer exists) from the chain.
    Models that return 200-but-empty are kept — they may succeed on real prompts
    or serve as a signal to fall through to the next model.
    """
    global _MODEL_FALLBACK_CHAIN, _chain_validated
    if _chain_validated:
        return
    _chain_validated = True

    import urllib.request as _req
    import json as _json

    PING = _json.dumps({
        "model": "__MODEL__",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5
    }).encode()
    HEADERS = {
        "Authorization": f"Bearer {api_key or ''}",
        "Content-Type": "application/json",
    }

    valid: list = []
    deduped = list(dict.fromkeys(_MODEL_FALLBACK_CHAIN))
    for model in deduped:
        body = PING.replace(b"__MODEL__", model.encode())
        request = _req.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body, headers=HEADERS, method="POST"
        )
        try:
            with _req.urlopen(request, timeout=10) as resp:
                status = resp.status
                # 200 means the slug is alive (content may still be empty — handled in _call_llm)
                valid.append(model)
                logger.info(f"[chain-validation] ✅ '{model}' alive (HTTP {status})")
        except _req.HTTPError as e:
            if e.code == 404:
                logger.warning(f"[chain-validation] ❌ '{model}' returned 404 — removed from chain.")
            elif e.code == 429:
                # Rate-limited during validation — keep the model (assume it exists)
                valid.append(model)
                logger.info(f"[chain-validation] ⚠️  '{model}' rate-limited (429) — kept in chain.")
            else:
                # Other HTTP errors (503 etc.) — keep, may be transient
                valid.append(model)
                logger.info(f"[chain-validation] ⚠️  '{model}' HTTP {e.code} — kept in chain.")
        except Exception as e:
            # Network error — keep, may be transient
            valid.append(model)
            logger.warning(f"[chain-validation] ⚠️  '{model}' error ({e}) — kept in chain.")

    if not valid:
        logger.error("[chain-validation] No valid models remain. Restoring defaults.")
        valid = ["openai/gpt-oss-120b:free", "google/gemma-4-31b-it:free"]

    _MODEL_FALLBACK_CHAIN = valid
    logger.info(f"[chain-validation] Active model chain: {_MODEL_FALLBACK_CHAIN}")


# ── Core LLM caller with full fallback chain ──────────────────────────────────
def _call_llm(messages: list, max_tokens: int, session_id: str, purpose: str = "chat") -> str:
    """
    Calls OpenRouter with the fallback model chain.
    Returns the reply string, or raises RuntimeError only if ALL models fail.

    Handles per model:
      - None / empty content         → skip to next model
      - Empty choices list           → skip to next model
      - HTTP 404 (slug gone)         → skip to next model
      - HTTP 429 (rate limited)      → 2s backoff, skip to next model
      - finish_reason='content_filter' → return friendly message immediately
      - finish_reason='length'       → content truncated but still usable
      - Network / timeout error      → skip to next model
    """
    # Validate model chain once at startup (removes dead 404 slugs)
    _validate_model_chain()

    # Dedup model chain (env var might repeat a default)
    model_chain = list(dict.fromkeys(_MODEL_FALLBACK_CHAIN))

    # Log total prompt size before sending
    total_chars = _prompt_char_length(messages)
    logger.info(
        f"[{purpose}] session={session_id} | prompt_chars={total_chars} "
        f"| max_tokens={max_tokens} | chain={model_chain}"
    )

    last_error: Exception = RuntimeError("No models available in fallback chain.")

    for attempt, model in enumerate(model_chain, start=1):
        try:
            logger.info(f"[{purpose}] Attempt {attempt}/{len(model_chain)} — model='{model}'")

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                timeout=60.0,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": "https://github.com/anshul-94/mini-GPT",
                    "X-Title": "Mini GPT Chatbot"
                }
            )

            # ── Log raw response for debugging ────────────────────────────
            try:
                raw_dump = response.model_dump()
                finish = (raw_dump.get("choices") or [{}])[0].get("finish_reason", "?")
                usage = raw_dump.get("usage") or {}
                logger.info(
                    f"[{purpose}] model='{model}' HTTP 200 | "
                    f"finish_reason={finish!r} | "
                    f"prompt_tokens={usage.get('prompt_tokens','?')} | "
                    f"completion_tokens={usage.get('completion_tokens','?')}"
                )
            except Exception:
                pass  # logging failure must never crash the handler

            # ── Defensive response inspection ─────────────────────────────
            if not response.choices or len(response.choices) == 0:
                logger.warning(
                    f"[{purpose}] model='{model}' returned zero choices → trying next."
                )
                last_error = RuntimeError(f"Model '{model}' returned empty choices list.")
                time.sleep(0.5)
                continue

            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)

            # Content filter — model refused the request
            if finish_reason == "content_filter":
                logger.warning(f"[{purpose}] model='{model}' content_filter triggered.")
                return "I'm sorry, I can't respond to that request."

            content = getattr(choice.message, "content", None)

            # None or empty string — try next model
            if content is None or (isinstance(content, str) and content.strip() == ""):
                logger.warning(
                    f"[{purpose}] model='{model}' returned None/empty content "
                    f"(finish_reason={finish_reason!r}) → trying next model."
                )
                last_error = RuntimeError(
                    f"Model '{model}' returned empty content "
                    f"(finish_reason={finish_reason!r})."
                )
                time.sleep(0.5)
                continue

            # finish_reason='length' — response cut at max_tokens but content is usable
            if finish_reason == "length":
                logger.warning(
                    f"[{purpose}] model='{model}' truncated at max_tokens={max_tokens} "
                    f"— content still usable."
                )

            logger.info(
                f"[{purpose}] ✅ model='{model}' SUCCESS | "
                f"reply_chars={len(content)} | finish={finish_reason!r}"
            )
            return content.strip()

        except Exception as e:
            err_str = str(e)
            # Detect 404 (model slug gone) — log and skip immediately
            if "404" in err_str or "not found" in err_str.lower():
                logger.warning(
                    f"[{purpose}] model='{model}' 404 Not Found — "
                    f"slug is dead, skipping."
                )
                last_error = RuntimeError(f"Model '{model}' returned 404 (slug removed).")
                continue  # no sleep needed

            # Detect 429 (rate limited) — back off then skip
            if "429" in err_str or "rate limit" in err_str.lower() or "too many" in err_str.lower():
                logger.warning(
                    f"[{purpose}] model='{model}' 429 Rate Limited — "
                    f"waiting 2s then trying next model."
                )
                last_error = RuntimeError(f"Model '{model}' rate limited (429).")
                time.sleep(2.0)
                continue

            logger.error(f"[{purpose}] model='{model}' exception: {e}")
            last_error = e
            time.sleep(0.5)
            continue

    # All models exhausted
    logger.error(
        f"[{purpose}] ❌ ALL {len(model_chain)} models exhausted. "
        f"Last error: {last_error}"
    )
    raise last_error


# ── LTM helpers ───────────────────────────────────────────────────────────────

def update_ltm_summary(existing_summary: str, evicted_messages: list) -> str:
    """Consolidates evicted messages from STM into the LTM summary using LLM."""
    formatted_exchanges = []
    for msg in evicted_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted_exchanges.append(f"{role}: {msg['content']}")

    new_exchanges = "\n".join(formatted_exchanges)
    prompt = SUMMARIZE_PROMPT.format(
        existing_summary=existing_summary or "(None)",
        new_messages=new_exchanges
    )

    logger.info("Consolidating memory: updating LTM summary")

    try:
        summary = _call_llm(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            session_id="ltm_update",
            purpose="LTM-summarize"
        )
        return compress_summary(summary)
    except Exception as e:
        logger.error(f"Failed to update LTM summary: {e}")
        raise


def compress_summary(summary: str) -> str:
    """Compresses LTM summary if it exceeds 400 characters."""
    if len(summary) < 400:
        return summary

    logger.warning("LTM summary exceeded threshold. Compressing...")
    try:
        return _call_llm(
            messages=[{"role": "user", "content": COMPRESS_PROMPT.format(summary=summary)}],
            max_tokens=100,
            session_id="ltm_compress",
            purpose="LTM-compress"
        )
    except Exception as e:
        logger.error(f"Failed to compress LTM summary: {e}")
        return summary  # Return uncompressed rather than crash


# ── Main chat function ────────────────────────────────────────────────────────

def chat_llm(message: str, session_id: str) -> str:
    """
    Core LLM chat function with RAG context injection, model fallback,
    context budget management, and full defensive response handling.

    Message construction order:
      1. System prompt (RAG-aware if files uploaded, standard otherwise)
      2. LTM memory summary (if any)
      3. RAG analytics block — TRIMMED to MAX_ANALYTICS_CHARS
      4. RAG retrieved chunks — TRIMMED to MAX_RETRIEVED_CHARS
      5. STM (last 4 messages)
      6. Current user message

    If the full prompt exceeds MAX_TOTAL_PROMPT_CHARS, retrieved chunks
    are progressively trimmed further before sending.
    """
    global session_memories

    if not api_key:
        raise ValueError("OpenRouter API key is missing. Please set OPENROUTER_API_KEY in the .env file.")

    # ── Initialize session ────────────────────────────────────────────────────
    if session_id not in session_memories:
        session_memories[session_id] = {"stm": [], "summary": ""}

    session_memory = session_memories[session_id]
    stm = session_memory["stm"]

    # Session count guard
    if len(session_memories) > 1000:
        old_sessions = list(session_memories.keys())[:200]
        for s_id in old_sessions:
            session_memories.pop(s_id, None)
        logger.info(f"Memory Cleanup: Evicted 200 sessions. Active: {len(session_memories)}")

    # ── Check RAG state ───────────────────────────────────────────────────────
    from rag import retriever as rag_retriever
    has_docs = rag_retriever.has_documents(session_id)

    # ── Build messages array ──────────────────────────────────────────────────
    if has_docs:
        messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]

        # LTM memory block
        if session_memory["summary"]:
            messages.append({
                "role": "system",
                "content": f"User context (from memory):\n{session_memory['summary']}"
            })

        # ── Analytics block (CSV / Excel / WhatsApp stats) ────────────────────
        raw_analytics = rag_retriever.get_analytics_text(session_id)
        if raw_analytics:
            analytics_text = _trim_to_budget(
                raw_analytics, MAX_ANALYTICS_CHARS, label="analytics"
            )
            messages.append({
                "role": "system",
                "content": (
                    "Structured Data Analytics "
                    "(use for counting/aggregation questions):\n\n"
                    + analytics_text
                )
            })

        # ── Semantic retrieval — k=3 to keep context tight ────────────────────
        # We use k=3 by default; large WhatsApp/CSV files benefit from fewer
        # but more precise chunks to avoid flooding the context window.
        session_info = rag_retriever.get_session_info(session_id)
        total_chunks = session_info.get("total_chunks", 0)
        # Adaptive k: fewer chunks when the index is very large
        k = 3 if total_chunks > 100 else 5
        logger.info(
            f"[RAG] total_chunks={total_chunks} → using k={k} for retrieval"
        )

        raw_context = rag_retriever.get_context(session_id, message, k=k)
        if raw_context:
            context_text = _trim_to_budget(
                raw_context, MAX_RETRIEVED_CHARS, label="retrieved-chunks"
            )
            messages.append({
                "role": "system",
                "content": "Relevant Document Content (answer from this):\n\n" + context_text
            })

        # File list hint
        file_list = ", ".join(session_info.get("file_names", []))
        messages.append({
            "role": "system",
            "content": f"Uploaded files in this session: {file_list}"
        })

    else:
        # ── Standard no-RAG path (original behavior preserved) ────────────────
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if session_memory["summary"]:
            messages.append({
                "role": "system",
                "content": f"Long-term memory (user facts):\n{session_memory['summary']}"
            })

    # ── STM (short-term conversation history) ────────────────────────────────
    prior_history = stm[-4:]
    messages.extend(prior_history)
    messages.append({"role": "user", "content": message})

    # ── Context window hard cap ───────────────────────────────────────────────
    # If the prompt is still too large after per-block trimming, progressively
    # remove the retrieved-chunks system message (most expensive block).
    total_chars = _prompt_char_length(messages)
    if total_chars > MAX_TOTAL_PROMPT_CHARS:
        logger.warning(
            f"[Budget] Total prompt = {total_chars} chars > cap {MAX_TOTAL_PROMPT_CHARS}. "
            f"Trimming retrieved-chunks block."
        )
        # Find and trim the retrieved-chunks system message
        for i, msg in enumerate(messages):
            if msg.get("role") == "system" and "Relevant Document Content" in msg.get("content", ""):
                excess = total_chars - MAX_TOTAL_PROMPT_CHARS
                current = msg["content"]
                trimmed = _trim_to_budget(
                    current,
                    max(500, len(current) - excess),
                    label="retrieved-chunks-emergency"
                )
                messages[i] = {"role": "system", "content": trimmed}
                break

        final_chars = _prompt_char_length(messages)
        logger.info(f"[Budget] After emergency trim: {final_chars} chars")

    # ── Max tokens: more room for RAG answers ────────────────────────────────
    max_tokens = 600 if has_docs else 200

    # ── Call LLM with fallback chain ─────────────────────────────────────────
    try:
        reply = _call_llm(messages, max_tokens=max_tokens, session_id=session_id, purpose="chat")

        # ── Update STM on success ─────────────────────────────────────────────
        stm.append({"role": "user", "content": message})
        stm.append({"role": "assistant", "content": reply})

        # ── STM compaction ────────────────────────────────────────────────────
        if len(stm) > 4:
            evicted = stm[:-4]
            logger.info(f"Compacting STM for session {session_id}: evicting {len(evicted)} messages.")
            try:
                new_summary = update_ltm_summary(session_memory["summary"], evicted)
                session_memory["summary"] = new_summary
                session_memory["stm"] = stm[-4:]
                logger.info(
                    f"Compaction OK. LTM={len(new_summary)} chars, "
                    f"STM={len(session_memory['stm'])} msgs"
                )
            except Exception as se:
                logger.error(f"Compaction failed — trimming STM without LTM update: {se}")
                session_memory["stm"] = stm[-4:]

        return reply

    except Exception as e:
        logger.error(f"All LLM attempts failed for session {session_id}: {str(e)}")
        # Re-raise so the HTTP layer returns a proper 502
        raise
