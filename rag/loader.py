"""
rag/loader.py — Multi-Format Document Loader

Supports: PDF, CSV, Excel (.xlsx/.xls), DOCX, TXT, WhatsApp .txt, Images (OCR).
Returns a list of LangChain Document objects + an optional analytics dict.
"""

import os
import io
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from langchain_core.documents import Document

logger = logging.getLogger("rag.loader")

# ── Supported extensions ───────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".pdf", ".csv", ".xlsx", ".xls",
    ".txt", ".docx",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"
}


def load_pdf(file_path: str) -> List[Document]:
    """Load PDF using LangChain's PyPDFLoader."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logger.info(f"PDF loaded: {len(docs)} pages from {file_path}")
        return docs
    except Exception as e:
        logger.error(f"PDF load failed for {file_path}: {e}")
        raise


def load_csv(file_path: str) -> Tuple[List[Document], Dict[str, Any]]:
    """Load CSV and generate pandas analytics."""
    try:
        import pandas as pd
        from langchain_community.document_loaders.csv_loader import CSVLoader

        df = pd.read_csv(file_path)
        analytics = _dataframe_analytics(df, source=os.path.basename(file_path))

        loader = CSVLoader(file_path=file_path, encoding="utf-8")
        try:
            docs = loader.load()
        except Exception:
            # Fallback: manually create docs from dataframe
            docs = _dataframe_to_docs(df, source=file_path)

        logger.info(f"CSV loaded: {len(docs)} rows, {len(df.columns)} columns")
        return docs, analytics
    except Exception as e:
        logger.error(f"CSV load failed for {file_path}: {e}")
        raise


def load_excel(file_path: str) -> Tuple[List[Document], Dict[str, Any]]:
    """Load Excel file using pandas, generate analytics per sheet."""
    try:
        import pandas as pd

        excel_file = pd.ExcelFile(file_path)
        all_docs = []
        combined_analytics: Dict[str, Any] = {"sheets": {}, "total_rows": 0}

        for sheet_name in excel_file.sheet_names:
            df = excel_file.parse(sheet_name)
            sheet_analytics = _dataframe_analytics(df, source=f"{os.path.basename(file_path)} — Sheet: {sheet_name}")
            combined_analytics["sheets"][sheet_name] = sheet_analytics
            combined_analytics["total_rows"] += len(df)

            docs = _dataframe_to_docs(df, source=file_path, metadata_extra={"sheet": sheet_name})
            all_docs.extend(docs)

        combined_analytics["sheet_names"] = excel_file.sheet_names
        logger.info(f"Excel loaded: {len(all_docs)} rows across {len(excel_file.sheet_names)} sheets")
        return all_docs, combined_analytics
    except Exception as e:
        logger.error(f"Excel load failed for {file_path}: {e}")
        raise


def load_docx(file_path: str) -> List[Document]:
    """Load DOCX using docx2txt."""
    try:
        import docx2txt
        text = docx2txt.process(file_path)
        docs = [Document(
            page_content=text,
            metadata={"source": file_path, "file_type": "docx"}
        )]
        logger.info(f"DOCX loaded: {len(text)} chars from {file_path}")
        return docs
    except Exception as e:
        logger.error(f"DOCX load failed for {file_path}: {e}")
        raise


def load_text(file_path: str) -> List[Document]:
    """Load plain text file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        docs = [Document(
            page_content=text,
            metadata={"source": file_path, "file_type": "txt"}
        )]
        logger.info(f"TXT loaded: {len(text)} chars from {file_path}")
        return docs
    except Exception as e:
        logger.error(f"TXT load failed for {file_path}: {e}")
        raise


def load_image_ocr(file_path: str) -> List[Document]:
    """
    Load image and extract text via Tesseract OCR.
    Falls back to a description if tesseract is not installed.
    """
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)

        if not text.strip():
            text = f"[Image file: {os.path.basename(file_path)}] — No readable text was detected by OCR."
        else:
            logger.info(f"OCR extracted {len(text)} chars from {file_path}")

        docs = [Document(
            page_content=text,
            metadata={"source": file_path, "file_type": "image", "ocr": True}
        )]
        return docs
    except ImportError:
        logger.warning("pytesseract or Pillow not installed. Returning placeholder.")
        docs = [Document(
            page_content=f"[Image: {os.path.basename(file_path)}] — OCR unavailable (install pytesseract and tesseract-ocr).",
            metadata={"source": file_path, "file_type": "image", "ocr": False}
        )]
        return docs
    except Exception as e:
        logger.error(f"Image OCR failed for {file_path}: {e}")
        raise


def load_whatsapp(file_path: str) -> Tuple[List[Document], Dict[str, Any], List[Dict]]:
    """Load WhatsApp chat file: parse messages, compute analytics, return docs + raw messages."""
    from rag.whatsapp import parse_messages, compute_analytics, analytics_to_text, messages_to_documents, is_whatsapp_file

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    if not is_whatsapp_file(text):
        logger.warning(f"{file_path} does not appear to be a WhatsApp export. Loading as plain text.")
        plain_docs = load_text(file_path)
        return plain_docs, {}, []

    messages = parse_messages(text)
    analytics = compute_analytics(messages)
    raw_docs = messages_to_documents(messages)

    docs = [
        Document(page_content=content, metadata=meta)
        for content, meta in raw_docs
    ]

    logger.info(f"WhatsApp loaded: {len(messages)} messages → {len(docs)} doc chunks")
    return docs, analytics, messages  # <— raw messages returned as 3rd element


# ── Main entry point ───────────────────────────────────────────────────────────

def load_file(file_path: str) -> Tuple[List[Document], Dict[str, Any], List[Dict]]:
    """
    Detect file type and load it.
    Returns: (list of Documents, analytics dict, raw_messages list)
    raw_messages is only populated for WhatsApp files; empty list otherwise.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: '{ext}'. Supported: {SUPPORTED_EXTENSIONS}")

    analytics: Dict[str, Any] = {}
    raw_messages: List[Dict] = []

    # WhatsApp detection: .txt files checked first
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            preview = f.read(3000)
        from rag.whatsapp import is_whatsapp_file
        if is_whatsapp_file(preview):
            docs, analytics, raw_messages = load_whatsapp(file_path)
            return docs, analytics, raw_messages
        else:
            docs = load_text(file_path)
            return docs, analytics, raw_messages

    elif ext == ".pdf":
        docs = load_pdf(file_path)
        return docs, analytics, raw_messages

    elif ext == ".csv":
        docs, analytics = load_csv(file_path)
        return docs, analytics, raw_messages

    elif ext in (".xlsx", ".xls"):
        docs, analytics = load_excel(file_path)
        return docs, analytics, raw_messages

    elif ext == ".docx":
        docs = load_docx(file_path)
        return docs, analytics, raw_messages

    elif ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"):
        docs = load_image_ocr(file_path)
        return docs, analytics, raw_messages

    else:
        raise ValueError(f"Unsupported file type: '{ext}'")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dataframe_analytics(df, source: str = "") -> Dict[str, Any]:
    """
    Generate analytics from a pandas DataFrame.
    Includes: shape, column names, dtypes, numeric stats, top-N values per column.
    """
    import pandas as pd

    analytics: Dict[str, Any] = {
        "source": source,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_stats": {},
        "top_values": {},
        "null_counts": df.isnull().sum().to_dict()
    }

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                analytics["numeric_stats"][col] = {
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None,
                    "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                    "sum": float(df[col].sum()) if not df[col].isnull().all() else None,
                    "count": int(df[col].count())
                }
            except Exception:
                pass
        else:
            try:
                top = df[col].value_counts().head(5).to_dict()
                analytics["top_values"][col] = {str(k): int(v) for k, v in top.items()}
            except Exception:
                pass

    return analytics


def _dataframe_to_docs(df, source: str = "", metadata_extra: Dict = None) -> List[Document]:
    """Convert a DataFrame to a list of LangChain Documents (one per row batch)."""
    docs = []
    metadata_extra = metadata_extra or {}
    batch_size = 50

    col_names = list(df.columns)

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        rows_text = []
        for _, row in batch.iterrows():
            row_str = " | ".join([f"{col}: {val}" for col, val in zip(col_names, row.values)])
            rows_text.append(row_str)

        content = f"Columns: {', '.join(col_names)}\n" + "\n".join(rows_text)
        metadata = {"source": source, "row_start": i, "row_end": i + len(batch), **metadata_extra}
        docs.append(Document(page_content=content, metadata=metadata))

    return docs


def analytics_to_text(analytics: Dict[str, Any], file_type: str = "structured") -> str:
    """
    Convert DataFrame analytics to a structured text block for LLM injection.
    Handles both flat (CSV) and nested (Excel multi-sheet) analytics.
    """
    if not analytics:
        return ""

    lines = [f"=== File Analytics ==="]

    # Handle Excel multi-sheet
    if "sheets" in analytics:
        lines.append(f"Total Rows Across All Sheets: {analytics.get('total_rows', 0)}")
        lines.append(f"Sheets: {', '.join(str(s) for s in analytics.get('sheet_names', []))}")
        for sheet_name, sheet_analytics in analytics["sheets"].items():
            lines.append(f"\n--- Sheet: {sheet_name} ---")
            lines.extend(_flat_analytics_lines(sheet_analytics))
    elif analytics.get("type") == "document":
        lines.append(f"Executive Summary: {analytics.get('summary', '')}")
        lines.append(f"Key Topics: {', '.join(analytics.get('topics', []))}")
        lines.append(f"Important Entities: {', '.join(analytics.get('entities', []))}")
    else:
        lines.extend(_flat_analytics_lines(analytics))

    return "\n".join(lines)


def _flat_analytics_lines(a: Dict[str, Any]) -> List[str]:
    """Generate text lines for flat (single DataFrame) analytics."""
    lines = []
    lines.append(f"Rows: {a.get('rows', 'N/A')}")
    lines.append(f"Columns: {', '.join(str(c) for c in a.get('columns', []))}")

    if a.get("numeric_stats"):
        lines.append("\nNumeric Column Statistics:")
        for col, stats in a["numeric_stats"].items():
            parts = []
            if stats.get("min") is not None: parts.append(f"min={stats['min']:.2f}")
            if stats.get("max") is not None: parts.append(f"max={stats['max']:.2f}")
            if stats.get("mean") is not None: parts.append(f"avg={stats['mean']:.2f}")
            if stats.get("sum") is not None: parts.append(f"sum={stats['sum']:.2f}")
            lines.append(f"  {col}: {', '.join(parts)}")

    if a.get("top_values"):
        lines.append("\nTop Values Per Column (up to 5 each):")
        for col, top in a["top_values"].items():
            top_str = ", ".join([f"'{k}': {v}" for k, v in list(top.items())[:5]])
            lines.append(f"  {col}: {top_str}")

    return lines
