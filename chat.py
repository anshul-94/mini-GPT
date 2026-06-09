import os
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

# Optimized system prompt
SYSTEM_PROMPT = """You are Xai, a friendly AI assistant. Strict rules:
1. Language: Reply ONLY in the user's language (English, Hindi, Hinglish). Never mix.
2. Tone: Friendly, natural, no cultural assumptions.
3. Style: Very short (1-2 sentences)."""

# Robust prompt template to merge evicted turns into LTM summary
SUMMARIZE_PROMPT = """You are a conversation memory manager. Consolidate these new exchanges into the running summary.
Current Summary:
{existing_summary}

New exchanges:
{new_messages}

Output a short, updated summary (max 3 bullets) listing key details (like user name, project stack, goals, and facts). Reply ONLY with the updated summary, nothing else."""

# Concise prompt template to compress large summary
COMPRESS_PROMPT = """Compress this summary to under 20 words, retaining key user facts:
{summary}
Reply ONLY with the compressed version."""

# Thread-safe in-memory session memories storage
# Schema: { session_id (str): { "stm": list, "summary": str } }
session_memories = {}

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
    
    # Default to a working free model
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
    logger.info(f"Consolidating memory: updating LTM summary via model '{model}'")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
            timeout=15.0
        )
        if response.choices and len(response.choices) > 0:
            summary = response.choices[0].message.content
            if summary:
                return compress_summary(summary.strip())
    except Exception as e:
        logger.error(f"Failed to update LTM summary: {e}")
        
    return existing_summary

def compress_summary(summary: str) -> str:
    """Compresses LTM summary further if it exceeds a critical token size threshold."""
    # Compress summary if it gets longer than 400 characters
    if len(summary) < 400:
        return summary
        
    # Default to a working free model
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
    logger.warning("LTM summary exceeded character threshold. Compressing summary...")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": COMPRESS_PROMPT.format(summary=summary)}],
            temperature=0.3,
            max_tokens=100,
            timeout=15.0
        )
        if response.choices and len(response.choices) > 0:
            compressed = response.choices[0].message.content
            if compressed:
                return compressed.strip()
    except Exception as e:
        logger.error(f"Failed to compress LTM summary: {e}")
        
    return summary

def chat_llm(message: str, session_id: str) -> str:
    global session_memories
    
    if not api_key:
        raise ValueError("OpenRouter API key is missing. Please set OPENROUTER_API_KEY in the .env file.")

    # Initialize session memory structure if empty
    if session_id not in session_memories:
        session_memories[session_id] = {
            "stm": [],
            "summary": ""
        }
        
    session_memory = session_memories[session_id]
    stm = session_memory["stm"]
    
    # Append the user's message to Short-Term Memory (STM)
    stm.append({"role": "user", "content": message})
    
    # STM Compaction Logic: Keep only the most recent 6 exchanges (12 messages) directly.
    # When user sends the 7th message, len(stm) becomes 13 (6 turns + 1 new user message).
    # We evict the oldest turn (first 2 messages: user + assistant) and merge them into LTM summary.
    if len(stm) > 12:
        logger.info(f"Compacting memory for session {session_id}: evicting oldest turn.")
        evicted = stm[:2]
        session_memory["stm"] = stm[2:]
        stm = session_memory["stm"]
        # Consolidate evicted messages into LTM summary
        session_memory["summary"] = update_ltm_summary(session_memory["summary"], evicted)
    
    # Manage active memory limits to prevent leaks
    if len(session_memories) > 1000:
        # Evict oldest 200 sessions if size exceeds 1000
        old_sessions = list(session_memories.keys())[:200]
        for s_id in old_sessions:
            session_memories.pop(s_id, None)
        logger.info(f"Memory Cleanup: Evicted 200 oldest sessions. Active sessions: {len(session_memories)}")

    # Construct messages array for OpenRouter API call
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    if session_memory["summary"]:
        messages.append({
            "role": "system",
            "content": f"Context summary of past conversation:\n{session_memory['summary']}"
        })
        
    messages.extend(stm)
    
    # Get model from environment variable, default to a working free model
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
    
    logger.info(f"Sending request for session {session_id} using model '{model}' (STM: {len(stm)} messages, LTM: {len(session_memory['summary'])} chars)")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            timeout=30.0,
            max_tokens=200,
            extra_headers={
                "HTTP-Referer": "https://github.com/anshul-94/mini-GPT",
                "X-Title": "Mini GPT Chatbot"
            }
        )
        
        if not response.choices or len(response.choices) == 0:
            raise RuntimeError("Received empty response choices from OpenRouter API.")
            
        reply = response.choices[0].message.content
        if reply is None:
            raise RuntimeError("Received empty message content from OpenRouter API.")
            
        reply = reply.strip()
        
        # Add assistant reply to STM
        stm.append({"role": "assistant", "content": reply})
        
        return reply
    except Exception as e:
        logger.error(f"Error querying OpenRouter API for session {session_id}: {str(e)}")
        # Remove user message from STM if the call failed, so we don't pollute subsequent requests
        if stm and stm[-1]["role"] == "user":
            stm.pop()
        raise e
