"""
rag/whatsapp.py — WhatsApp Chat Parser & Analytics Engine

Parses exported WhatsApp chat files (DD/MM/YY, HH:MM - Sender: Message)
and produces both LangChain Documents for vector search and a rich
analytics summary for direct LLM injection.
"""

import re
import logging
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any

logger = logging.getLogger("rag.whatsapp")

# ── Regex to match WhatsApp message lines ─────────────────────────────────────
# Handles both 12h and 24h formats, and iOS/Android export variants
_MSG_PATTERN = re.compile(
    r"^(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\s+[-–]\s+([^:]+?):\s+(.+)$",
    re.MULTILINE
)

# Common emoji unicode ranges
_EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030]+",
    flags=re.UNICODE
)

# Keywords to track
TRACKED_KEYWORDS = ["sorry", "thank you", "thanks", "please", "love", "miss", "okay", "ok", "yes", "no"]


def is_whatsapp_file(text: str) -> bool:
    """Returns True if the text looks like a WhatsApp export."""
    return bool(_MSG_PATTERN.search(text[:3000]))


def parse_messages(text: str) -> List[Dict[str, str]]:
    """
    Parse WhatsApp export text into a list of message dicts.
    Each dict: { date, time, sender, message }
    Multi-line messages are joined with \\n.
    """
    messages = []
    lines = text.splitlines()
    current = None

    for line in lines:
        match = _MSG_PATTERN.match(line)
        if match:
            if current:
                messages.append(current)
            date, time_, sender, message = match.groups()
            current = {
                "date": date.strip(),
                "time": time_.strip(),
                "sender": sender.strip(),
                "message": message.strip()
            }
        else:
            # Continuation of previous message
            if current and line.strip():
                current["message"] += "\n" + line.strip()

    if current:
        messages.append(current)

    return messages


def compute_analytics(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Compute rich analytics from parsed messages.
    Returns a dict with all stats.
    """
    if not messages:
        return {}

    total_messages = len(messages)
    sender_counts: Counter = Counter()
    word_freq: Counter = Counter()
    emoji_freq: Counter = Counter()
    keyword_counts: Dict[str, int] = {kw: 0 for kw in TRACKED_KEYWORDS}
    daily_activity: Counter = Counter()
    monthly_activity: Counter = Counter()
    media_messages = 0
    longest_message = {"sender": "", "length": 0, "message": ""}

    for msg in messages:
        sender = msg["sender"]
        text = msg["message"]
        date = msg["date"]

        # Skip system messages and media omitted
        if sender in ("<Media omitted>", "‎<Media omitted>") or text in ("<Media omitted>", "‎<Media omitted>"):
            media_messages += 1
            continue

        sender_counts[sender] += 1

        # Word frequency
        words = re.findall(r"\b\w+\b", text.lower())
        word_freq.update(words)

        # Emoji frequency
        emojis = _EMOJI_PATTERN.findall(text)
        emoji_freq.update(emojis)

        # Keyword counts
        text_lower = text.lower()
        for kw in TRACKED_KEYWORDS:
            keyword_counts[kw] += text_lower.count(kw)

        # Daily activity (normalize date format)
        daily_activity[date] += 1

        # Monthly activity
        parts = re.split(r"[\/\-]", date)
        if len(parts) >= 2:
            month_key = f"{parts[1]}/{parts[2] if len(parts) > 2 else '??'}"
            monthly_activity[month_key] += 1

        # Longest message
        if len(text) > longest_message["length"]:
            longest_message = {"sender": sender, "length": len(text), "message": text[:200] + "..." if len(text) > 200 else text}

    # Top words (excluding common stop words)
    stopwords = {"the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
                 "of", "and", "or", "but", "i", "you", "we", "he", "she", "they",
                 "my", "your", "his", "her", "our", "this", "that", "was", "are",
                 "be", "been", "have", "has", "had", "will", "would", "can", "could",
                 "with", "from", "by", "about", "up", "out", "so", "if", "do", "did",
                 "not", "what", "how", "when", "where", "who", "which", "there",
                 "their", "then", "them", "me", "him", "us", "its", "as", "into",
                 "just", "like", "get", "got", "yes", "ok", "okay", "im", "dont",
                 "it's", "i'm", "don't", "was", "were", "all", "no", "yeah"}
    top_words = [(w, c) for w, c in word_freq.most_common(30) if w not in stopwords and len(w) > 2][:15]
    top_emojis = emoji_freq.most_common(10)
    top_senders = sender_counts.most_common()
    most_active_day = daily_activity.most_common(1)[0] if daily_activity else ("N/A", 0)
    most_active_month = monthly_activity.most_common(1)[0] if monthly_activity else ("N/A", 0)

    return {
        "total_messages": total_messages,
        "total_participants": len(sender_counts),
        "participants": list(sender_counts.keys()),
        "message_counts_per_sender": dict(sender_counts.most_common()),
        "most_active_sender": top_senders[0] if top_senders else ("N/A", 0),
        "top_words": top_words,
        "top_emojis": top_emojis,
        "keyword_counts": keyword_counts,
        "most_active_day": most_active_day,
        "most_active_month": most_active_month,
        "media_messages": media_messages,
        "longest_message": longest_message,
        "daily_activity": dict(daily_activity.most_common(10)),
        "monthly_activity": dict(monthly_activity.most_common())
    }


def analytics_to_text(analytics: Dict[str, Any]) -> str:
    """
    Convert analytics dict to a structured text block for LLM injection.
    """
    if not analytics:
        return ""

    lines = ["=== WhatsApp Chat Analytics ==="]
    lines.append(f"Total Messages: {analytics.get('total_messages', 0)}")
    lines.append(f"Total Participants: {analytics.get('total_participants', 0)}")
    lines.append(f"Participants: {', '.join(analytics.get('participants', []))}")
    lines.append("")

    lines.append("--- Messages Per Sender ---")
    for sender, count in analytics.get("message_counts_per_sender", {}).items():
        lines.append(f"  {sender}: {count} messages")

    most_active = analytics.get("most_active_sender", ("N/A", 0))
    lines.append(f"\nMost Active Sender: {most_active[0]} ({most_active[1]} messages)")

    lines.append("\n--- Keyword Counts ---")
    for kw, count in analytics.get("keyword_counts", {}).items():
        if count > 0:
            lines.append(f"  '{kw}': {count} times")

    lines.append("\n--- Top Words ---")
    for word, count in analytics.get("top_words", [])[:10]:
        lines.append(f"  '{word}': {count}")

    if analytics.get("top_emojis"):
        lines.append("\n--- Top Emojis ---")
        for emoji, count in analytics.get("top_emojis", []):
            lines.append(f"  {emoji}: {count}")

    lines.append(f"\n--- Activity ---")
    lines.append(f"Most Active Day: {analytics.get('most_active_day', ('N/A', 0))[0]} ({analytics.get('most_active_day', ('N/A', 0))[1]} messages)")
    lines.append(f"Most Active Month: {analytics.get('most_active_month', ('N/A', 0))[0]} ({analytics.get('most_active_month', ('N/A', 0))[1]} messages)")
    lines.append(f"Media Messages: {analytics.get('media_messages', 0)}")

    return "\n".join(lines)


def messages_to_documents(messages: List[Dict[str, str]]) -> List[Tuple[str, dict]]:
    """
    Convert parsed messages to (text, metadata) tuples for LangChain Document creation.
    Groups messages into chunks of 50 for better retrieval granularity.
    """
    docs = []
    chunk_size = 50
    for i in range(0, len(messages), chunk_size):
        chunk = messages[i:i + chunk_size]
        lines = []
        for msg in chunk:
            lines.append(f"[{msg['date']} {msg['time']}] {msg['sender']}: {msg['message']}")
        text = "\n".join(lines)
        metadata = {
            "source": "whatsapp_chat",
            "chunk_index": i // chunk_size,
            "messages_in_chunk": len(chunk),
            "start_date": chunk[0]["date"],
            "end_date": chunk[-1]["date"]
        }
        docs.append((text, metadata))
    return docs
