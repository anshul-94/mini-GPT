import os
import logging
import time
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from chat import chat_llm

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot.main")

app = FastAPI(title="AI Chatbot Backend", version="1.0.0")

# Absolute path configuration for HTML files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")


# CORS middleware configuration to permit local files and cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str = Field(
        ..., 
        min_length=1, 
        max_length=4000, 
        description="User message content, between 1 and 4000 characters"
    )
    session_id: str = Field(
        ..., 
        min_length=1, 
        max_length=100, 
        description="Unique session identifier for isolation"
    )

# Simple rate limiter tracking last request timestamp per session_id
session_rate_limit = {}

def serve_html_file(file_path: str, name: str):
    if not os.path.exists(file_path):
        logger.error(f"{name} not found at expected path: {file_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"{name} not found"
        )
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.exception(f"Failed to read {name}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading {name}"
        )

@app.get("/", response_class=HTMLResponse)
async def serve_chatbot():
    return serve_html_file(INDEX_PATH, "index.html")


@app.post("/chat")
async def chat_endpoint(data: ChatRequest):
    # Apply rate limit (max 1 message per second per session_id)
    now = time.time()
    last_req = session_rate_limit.get(data.session_id, 0.0)
    if now - last_req < 1.0:
        logger.warning(f"Rate limit triggered for session {data.session_id}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait a second before sending another message."
        )
    session_rate_limit[data.session_id] = now
    # Intercept raw ping checks
    if data.message.strip().lower() == "ping":
        return {"reply": "pong"}

    try:
        reply = chat_llm(data.message.strip(), data.session_id)
        return {"reply": reply}
    except ValueError as ve:
        # Handle configuration issues or malformed payloads
        logger.error(f"Configuration or input validation error for session {data.session_id}: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        # Handle API key failures, timeout, rate limiting from provider
        logger.exception(f"Exception handling chat request for session {data.session_id}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI Assistant is temporarily unavailable. Please try again shortly."
        )

# Diagnostic health check endpoint
@app.get("/health")
async def health_check():
    api_key_status = "configured" if os.getenv("OPENROUTER_API_KEY") else "missing"
    return {
        "status": "healthy",
        "openrouter_key": api_key_status,
        "model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")
    }

if __name__ == "__main__":
    import uvicorn
    # Use reload=True and run as module string to allow dynamic code updates
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
