import os
import shutil
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from chat import chat_llm

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot.main")

app = FastAPI(
    title="Xai Assistant",
    description="Multi-Modal RAG AI Assistant. Upload PDFs, CSVs, Excel, WhatsApp chats, images and ask questions.",
    version="2.0.0"
)

# Absolute path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

# Ensure uploads directory exists
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Supported file extensions (whitelist)
ALLOWED_EXTENSIONS = {
    ".pdf", ".csv", ".xlsx", ".xls",
    ".txt", ".docx",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"
}

# Max file size: 50 MB
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ───────────────────────────────────────────────────────────

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


# ── Rate Limiters ─────────────────────────────────────────────────────────────
session_rate_limit: dict = {}          # chat: 1 msg/sec
upload_rate_limit: dict = {}           # upload: 1 upload/3 sec
upload_count: dict = {}                # total uploads per session (max 20)


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def get_file_type_label(ext: str) -> str:
    """Return a human-readable file type label."""
    mapping = {
        ".pdf": "PDF",
        ".csv": "CSV",
        ".xlsx": "Excel",
        ".xls": "Excel",
        ".txt": "Text/WhatsApp",
        ".docx": "Word Document",
        ".png": "Image (PNG)",
        ".jpg": "Image (JPG)",
        ".jpeg": "Image (JPEG)",
        ".gif": "Image (GIF)",
        ".bmp": "Image (BMP)",
        ".webp": "Image (WebP)",
        ".tiff": "Image (TIFF)"
    }
    return mapping.get(ext.lower(), "File")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_chatbot():
    return serve_html_file(INDEX_PATH, "index.html")


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(..., min_length=1, max_length=100)
):
    """
    Upload a file and build a RAG index for the session.
    Supports: PDF, CSV, Excel, DOCX, TXT, WhatsApp .txt, Images (OCR).
    """
    # ── Rate limiting ────────────────────────────────────────────────────────
    now = time.time()
    last_upload = upload_rate_limit.get(session_id, 0.0)
    if now - last_upload < 3.0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Upload rate limit: please wait 3 seconds between uploads."
        )
    upload_rate_limit[session_id] = now

    # Max uploads per session
    total_uploads = upload_count.get(session_id, 0)
    if total_uploads >= 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 20 files per session. Start a new chat to reset."
        )

    # ── Validate file ────────────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # ── Read and size-check ──────────────────────────────────────────────────
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content) // 1024 // 1024} MB). Maximum allowed: 50 MB."
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Save file to disk ────────────────────────────────────────────────────
    session_upload_dir = os.path.join(UPLOADS_DIR, session_id)
    os.makedirs(session_upload_dir, exist_ok=True)
    safe_name = f"{int(now)}_{file.filename}"
    file_path = os.path.join(session_upload_dir, safe_name)

    with open(file_path, "wb") as f_out:
        f_out.write(content)

    logger.info(f"File saved: {file_path} ({len(content)} bytes) for session {session_id}")

    # ── Load + Extract ───────────────────────────────────────────────────────
    try:
        from rag.loader import load_file, analytics_to_text
        from rag.whatsapp import analytics_to_text as wa_analytics_to_text
        from rag import retriever as rag_retriever
        from chat import generate_file_intelligence

        docs, analytics, raw_messages = load_file(file_path)

        if not docs:
            raise HTTPException(status_code=422, detail="Could not extract any text from the file.")

        # Determine file type and build analytics text for injection
        file_type_label = get_file_type_label(file_ext)

        # Generate intelligence if no tabular/WhatsApp analytics exist
        if not analytics and file_ext in [".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"]:
            analytics = generate_file_intelligence(docs, file_type_label, file.filename)

        # WhatsApp analytics use dedicated formatter
        if file_ext == ".txt" and "total_messages" in analytics:
            analytics_text = wa_analytics_to_text(analytics)
            file_type_label = "WhatsApp Chat"
        elif analytics:
            analytics_text = analytics_to_text(analytics, file_type=file_type_label)
        else:
            analytics_text = ""

        # ── Embed + Index ────────────────────────────────────────────────────
        total_chunks = rag_retriever.add_documents(
            session_id=session_id,
            docs=docs,
            file_name=file.filename,
            file_type=file_type_label,
            analytics=analytics,
            analytics_text=analytics_text,
            raw_messages=raw_messages if raw_messages else None
        )

        upload_count[session_id] = total_uploads + 1

        # Build analytics summary for frontend display
        analytics_summary = _build_frontend_analytics(analytics, file_ext)

        logger.info(
            f"Upload complete: session={session_id}, file={file.filename}, "
            f"type={file_type_label}, chunks={total_chunks}"
        )

        return {
            "status": "success",
            "file_name": file.filename,
            "file_type": file_type_label,
            "chunks_indexed": total_chunks,
            "pages_or_rows": len(docs),
            "analytics": analytics_summary,
            "message": f"✅ '{file.filename}' uploaded and indexed successfully! You can now ask questions about it."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Processing failed for {file.filename}: {e}")
        # Clean up saved file on failure
        try:
            os.remove(file_path)
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file: {str(e)}"
        )


@app.get("/session-files/{session_id}")
async def get_session_files(session_id: str):
    """Return the list of uploaded files for a session."""
    from rag import retriever as rag_retriever
    info = rag_retriever.get_session_info(session_id)
    return {
        "session_id": session_id,
        "files": [
            {"name": name, "type": ftype}
            for name, ftype in info.get("files", [])
        ],
        "total_chunks": info.get("total_chunks", 0),
        "has_documents": info.get("has_documents", False)
    }


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
        logger.error(f"Configuration or input validation error for session {data.session_id}: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.exception(f"Exception handling chat request for session {data.session_id}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI Assistant is temporarily unavailable. Please try again shortly."
        )


# Diagnostic health check endpoint
@app.get("/health")
async def health_check():
    api_key_status = "configured" if os.getenv("OPENROUTER_API_KEY") else "missing"
    from rag import retriever as rag_retriever
    # Count active RAG sessions
    active_rag = sum(
        1 for _ in [None]  # placeholder — retriever module tracks internally
    )
    return {
        "status": "healthy",
        "openrouter_key": api_key_status,
        "model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct"),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "huggingface"),
        "rag_enabled": True,
        "uploads_dir": UPLOADS_DIR
    }


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _build_frontend_analytics(analytics: dict, file_ext: str) -> dict:
    """
    Build a simplified analytics dict for the frontend display panel.
    """
    if not analytics:
        return {}

    result = {}

    # WhatsApp analytics
    if "total_messages" in analytics:
        result["type"] = "whatsapp"
        result["total_messages"] = analytics.get("total_messages", 0)
        result["participants"] = analytics.get("participants", [])
        result["most_active"] = analytics.get("most_active_sender", ["N/A", 0])
        result["keyword_counts"] = analytics.get("keyword_counts", {})
        result["top_emojis"] = analytics.get("top_emojis", [])[:5]
        result["top_words"] = analytics.get("top_words", [])[:5]
        result["most_active_day"] = analytics.get("most_active_day", ["N/A", 0])

    # CSV / flat DataFrame analytics
    elif "rows" in analytics:
        result["type"] = "tabular"
        result["rows"] = analytics.get("rows", 0)
        result["columns"] = analytics.get("columns", [])
        result["numeric_stats"] = analytics.get("numeric_stats", {})
        result["top_values"] = {
            col: list(vals.items())[:3]
            for col, vals in analytics.get("top_values", {}).items()
        }

    # Excel multi-sheet analytics
    elif "sheets" in analytics:
        result["type"] = "excel"
        result["total_rows"] = analytics.get("total_rows", 0)
        result["sheet_names"] = analytics.get("sheet_names", [])
        # Summarize first sheet only for display
        first_sheet = list(analytics.get("sheets", {}).values())
        if first_sheet:
            sheet = first_sheet[0]
            result["columns"] = sheet.get("columns", [])
            result["numeric_stats"] = sheet.get("numeric_stats", {})

    # Document intelligence
    elif analytics.get("type") == "document":
        result["type"] = "document"
        result["summary"] = analytics.get("summary", "")
        result["topics"] = analytics.get("topics", [])
        result["entities"] = analytics.get("entities", [])

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
