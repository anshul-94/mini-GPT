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

# Optimized system prompt enforcing strict language replication
SYSTEM_PROMPT = """You are Xai, a friendly AI assistant. Strict language rules:
1. Detect the dominant language of the last user message.
2. Reply ONLY in that exact same language:
   - English input -> Reply in English.
   - Hindi input -> Reply in Hindi (Devanagari script).
   - Hinglish input (Hindi written in Roman letters) -> Reply in Hinglish.
3. NEVER switch languages randomly or mix them.
4. Keep replies friendly, natural, and very short (1-2 sentences)."""

# Summarize prompt template organizing LTM summary by categories
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

# Compress prompt template
COMPRESS_PROMPT = """Compress this summary, retaining Name, Preferences, Projects, and Goals:
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
        raise e
        
    raise RuntimeError("Empty or invalid response from LLM for LTM summarization.")

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
            "content": f"User facts summary:\n{session_memory['summary']}"
        })
        
    # OPTIMIZATION: Only send Last 4 Messages of history + Current User Message (last 5 messages from STM)
    messages.extend(stm[-5:])
    
    # Get model from environment variable, default to a working free model
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
    
    logger.info(f"Sending request for session {session_id} using model '{model}' (STM payload: {len(messages)} msgs, LTM summary size: {len(session_memory['summary'])} chars)")
    
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
        
        # Post-turn Compaction Logic:
        # Keep only the last 4 messages (2 turns of history) in STM.
        # Evict older messages and merge them into the LTM summary.
        if len(stm) > 4:
            evicted = stm[:-4]
            logger.info(f"Compacting memory for session {session_id}: evicting {len(evicted)} messages.")
            try:
                new_summary = update_ltm_summary(session_memory["summary"], evicted)
                session_memory["summary"] = new_summary
                session_memory["stm"] = stm[-4:]
                logger.info(f"Compaction successful. New LTM summary size: {len(new_summary)} chars. STM size: {len(session_memory['stm'])}")
            except Exception as se:
                logger.error(f"Compaction postponed due to summarization failure: {se}")
        
        return reply
    except Exception as e:
        logger.error(f"Error querying OpenRouter API for session {session_id}: {str(e)}")
        # Remove user message from STM if the call failed, so we don't pollute subsequent requests
        if stm and stm[-1]["role"] == "user":
            stm.pop()
        raise e
