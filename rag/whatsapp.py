"""
rag/whatsapp.py — WhatsApp Chat Parser, Analytics Engine & Smart Retriever

Three responsibilities:
  1. Parse raw WhatsApp export text → list of message dicts
  2. Compute rich analytics (counts, emojis, keywords, activity)
  3. Route analytical vs. contextual questions to the right engine
     - Analytical questions  → query_analytics()  (no vector search needed)
     - Entity/topic questions → entity_search()    (full-text scan + context expansion)
     - General questions      → FAISS vector search (existing pipeline)
"""

import re
import logging
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger("rag.whatsapp")

# ── Regex to match WhatsApp message lines ─────────────────────────────────────
# Handles both 12h and 24h formats, and iOS/Android export variants
_MSG_PATTERN = re.compile(
    r"^\[?(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]?\s*(?:[-–]\s+)?([^:]+?):\s+(.+)$",
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
TRACKED_KEYWORDS = [
    "sorry", "thank you", "thanks", "please", "love", "miss",
    "okay", "ok", "yes", "no", "hi", "hello", "bye", "good morning",
    "good night", "happy", "sad", "angry", "haha", "lol"
]

# ── Intent classification patterns ────────────────────────────────────────────
# These patterns detect "analytical" questions that should be answered directly
# from computed stats rather than vector search.
_ANALYTICS_PATTERNS = [
    # Message counts
    r"\b(who|which\s+person|which\s+sender)\b.*(most|maximum|max|highest|top|most\s+active|sent\s+most)",
    r"\b(most\s+active|most\s+messages|most\s+texts|highest\s+messages)\b",
    r"\bhow\s+many\s+messages\b",
    r"\btotal\s+(messages|texts|chats)\b",
    r"\bmessage\s+count\b",
    # Emojis
    r"\btop\s*\d*\s*emoji",
    r"\bmost\s+used\s+emoji",
    r"\bfrequent\s+emoji",
    r"\bemoji\s+(count|stat|analytic|use)",
    # Keywords
    r"\b(who|how\s+many\s+times?|how\s+often).*(sorry|thank|please|love|miss|ok|okay)",
    r"\bsorry\s+(count|times?|most|who)\b",
    r"\bkeyword\b",
    # Time-based
    r"\bmost\s+active\s+(day|hour|time|month|week)\b",
    r"\b(busiest|peak)\s+(day|hour|time)\b",
    r"\bwhen\s+(was|were|is)\s+(the\s+)?(most|busiest|peak|highest)\b",
    r"\bhow\s+active\b",
    # Statistics
    r"\b(stat|statistic|analytic|summary|overview|report)\b",
    r"\bparticipant",
    r"\bwho\s+(is|are|was|were)\s+in\s+(this|the)\s+(chat|group|conversation)\b",
    r"\bwho\s+(chat|text|messag)",
    r"\bhow\s+long\s+(is|was|has)\s+(this|the)\s+chat\b",
    r"\bmedia\s+(count|messages?|files?)\b",
]
_ANALYTICS_RE = re.compile("|".join(_ANALYTICS_PATTERNS), re.IGNORECASE)


def classify_whatsapp_query(query: str) -> str:
    """
    Classify a user query into one of three routing categories:

    Returns:
      "analytics"  — answered directly from pre-computed stats
      "entity"     — entity/person/topic search (context expansion retrieval)
      "vector"     — standard FAISS vector search
    """
    if _ANALYTICS_RE.search(query):
        return "analytics"

    # Entity queries: contain a capitalized name or topic keyword
    # Heuristic: any word 4+ chars that starts with uppercase → likely a name
    words = query.split()
    for word in words:
        clean = re.sub(r"[^a-zA-Z]", "", word)
        if len(clean) >= 4 and clean[0].isupper() and clean.lower() not in {
            "what", "when", "where", "which", "this", "that", "they", "them",
            "their", "have", "does", "tell", "show", "give", "find", "about",
            "with", "from", "between", "discuss", "explain", "describe"
        }:
            return "entity"

    # Keywords that suggest topic/person search
    entity_keywords = r"\b(discuss|discussion|talk|mention|said|about|priyansu|chetu|friendship|relationship|fight|argument|plan|trip)\b"
    if re.search(entity_keywords, query, re.IGNORECASE):
        return "entity"

    return "vector"


# ── Core parser ────────────────────────────────────────────────────────────────

def is_whatsapp_file(text: str) -> bool:
    """Returns True if the text looks like a WhatsApp export."""
    return bool(_MSG_PATTERN.search(text[:3000]))


def parse_messages(text: str) -> List[Dict[str, Any]]:
    """
    Parse WhatsApp export text into a list of message dicts.
    Each dict: { date, time, hour, sender, message }
    Multi-line messages are joined with \\n.
    """
    messages = []
    lines = text.splitlines()
    current: Optional[Dict] = None

    for line in lines:
        match = _MSG_PATTERN.match(line)
        if match:
            if current:
                messages.append(current)
            date, time_, sender, message = match.groups()
            # Extract hour for hourly activity
            hour_match = re.match(r"(\d{1,2}):", time_.strip())
            hour = int(hour_match.group(1)) if hour_match else -1
            # Normalize PM hours
            if "PM" in time_.upper() and hour != 12:
                hour = (hour + 12) % 24
            elif "AM" in time_.upper() and hour == 12:
                hour = 0

            current = {
                "date": date.strip(),
                "time": time_.strip(),
                "hour": hour,
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


# ── Analytics engine ───────────────────────────────────────────────────────────

def compute_analytics(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute rich analytics from parsed messages.
    Returns a dict with all stats including per-sender keyword breakdown.
    """
    if not messages:
        return {}

    total_messages = len(messages)
    sender_counts: Counter = Counter()
    word_freq: Counter = Counter()
    emoji_freq: Counter = Counter()
    keyword_counts: Dict[str, int] = {kw: 0 for kw in TRACKED_KEYWORDS}
    sender_keyword_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {kw: 0 for kw in TRACKED_KEYWORDS})
    sender_emoji_counts: Dict[str, Counter] = defaultdict(Counter)
    daily_activity: Counter = Counter()
    hourly_activity: Counter = Counter()
    monthly_activity: Counter = Counter()
    media_messages = 0
    longest_message = {"sender": "", "length": 0, "message": ""}

    for msg in messages:
        sender = msg["sender"]
        text = msg["message"]
        date = msg["date"]
        hour = msg.get("hour", -1)

        # Skip system messages and media omitted
        if text in ("<Media omitted>", "‎<Media omitted>", "image omitted", "video omitted", "audio omitted", "sticker omitted"):
            media_messages += 1
            continue

        sender_counts[sender] += 1

        # Word frequency
        words = re.findall(r"\b\w+\b", text.lower())
        word_freq.update(words)

        # Emoji frequency
        emojis = _EMOJI_PATTERN.findall(text)
        emoji_freq.update(emojis)
        sender_emoji_counts[sender].update(emojis)

        # Keyword counts (global + per-sender)
        text_lower = text.lower()
        for kw in TRACKED_KEYWORDS:
            count = text_lower.count(kw)
            keyword_counts[kw] += count
            sender_keyword_counts[sender][kw] += count

        # Daily activity
        daily_activity[date] += 1

        # Hourly activity
        if hour >= 0:
            hourly_activity[hour] += 1

        # Monthly activity
        parts = re.split(r"[\/\-]", date)
        if len(parts) >= 2:
            month_key = f"{parts[1]}/{parts[2] if len(parts) > 2 else '??'}"
            monthly_activity[month_key] += 1

        # Longest message
        if len(text) > longest_message["length"]:
            longest_message = {
                "sender": sender,
                "length": len(text),
                "message": text[:200] + "..." if len(text) > 200 else text
            }

    # Top words (excluding common stop words)
    stopwords = {
        "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
        "of", "and", "or", "but", "i", "you", "we", "he", "she", "they",
        "my", "your", "his", "her", "our", "this", "that", "was", "are",
        "be", "been", "have", "has", "had", "will", "would", "can", "could",
        "with", "from", "by", "about", "up", "out", "so", "if", "do", "did",
        "not", "what", "how", "when", "where", "who", "which", "there",
        "their", "then", "them", "me", "him", "us", "its", "as", "into",
        "just", "like", "get", "got", "yes", "ok", "okay", "im", "dont",
        "it's", "i'm", "don't", "was", "were", "all", "no", "yeah", "na",
        "bhi", "toh", "hai", "haan", "nahi", "kar", "kya", "bhi", "aur",
        "se", "ko", "ne", "tha", "thi", "the", "ke", "ki", "koi", "yaar"
    }
    top_words = [
        (w, c) for w, c in word_freq.most_common(30)
        if w not in stopwords and len(w) > 2
    ][:15]

    top_emojis = emoji_freq.most_common(10)
    top_senders = sender_counts.most_common()
    most_active_day = daily_activity.most_common(1)[0] if daily_activity else ("N/A", 0)
    most_active_month = monthly_activity.most_common(1)[0] if monthly_activity else ("N/A", 0)
    most_active_hour_raw = hourly_activity.most_common(1)[0] if hourly_activity else (-1, 0)
    # Format hour nicely
    h = most_active_hour_raw[0]
    hour_label = f"{h:02d}:00-{(h+1)%24:02d}:00" if h >= 0 else "N/A"
    most_active_hour = (hour_label, most_active_hour_raw[1])

    return {
        "total_messages": total_messages,
        "total_participants": len(sender_counts),
        "participants": list(sender_counts.keys()),
        "message_counts_per_sender": dict(sender_counts.most_common()),
        "most_active_sender": top_senders[0] if top_senders else ("N/A", 0),
        "top_words": top_words,
        "top_emojis": top_emojis,
        "keyword_counts": keyword_counts,
        "sender_keyword_counts": {k: dict(v) for k, v in sender_keyword_counts.items()},
        "sender_emoji_counts": {k: dict(v.most_common(5)) for k, v in sender_emoji_counts.items()},
        "most_active_day": most_active_day,
        "most_active_hour": most_active_hour,
        "most_active_month": most_active_month,
        "media_messages": media_messages,
        "longest_message": longest_message,
        "daily_activity": dict(daily_activity.most_common(10)),
        "monthly_activity": dict(monthly_activity.most_common()),
        "hourly_activity": dict(hourly_activity.most_common(6)),
    }


def query_analytics(analytics: Dict[str, Any], query: str) -> str:
    """
    Directly answer an analytical question from pre-computed analytics.
    Returns a rich, formatted answer string.

    This is the dedicated analytics engine — bypasses vector search entirely.
    """
    if not analytics:
        return ""

    q = query.lower()
    lines = []

    counts = analytics.get("message_counts_per_sender", {})
    participants = analytics.get("participants", [])
    total = analytics.get("total_messages", 0)
    most_active_sender = analytics.get("most_active_sender", ("N/A", 0))
    keyword_counts = analytics.get("keyword_counts", {})
    sender_keyword_counts = analytics.get("sender_keyword_counts", {})
    top_emojis = analytics.get("top_emojis", [])
    sender_emoji_counts = analytics.get("sender_emoji_counts", {})
    top_words = analytics.get("top_words", [])
    most_active_day = analytics.get("most_active_day", ("N/A", 0))
    most_active_hour = analytics.get("most_active_hour", ("N/A", 0))
    most_active_month = analytics.get("most_active_month", ("N/A", 0))
    media_count = analytics.get("media_messages", 0)
    longest = analytics.get("longest_message", {})

    # ── Most active day ───────────────────────────────────────────────────────
    if re.search(r"most\s+active\s+day|busiest\s+day|peak\s+day|most\s+active\s+date|active\s+day", q):
        lines.append(f"📅 **Most Active Day:** {most_active_day[0]} ({most_active_day[1]} messages)")
        lines.append(f"\n**Top 10 Most Active Days:**")
        for date, count in analytics.get("daily_activity", {}).items():
            lines.append(f"  • {date}: {count} messages")

    # ── Most active hour ──────────────────────────────────────────────────────
    elif re.search(r"most\s+active\s+(hour|time)|busiest\s+(hour|time)|peak\s+(hour|time)|active\s+(hour|time)", q):
        lines.append(f"⏰ **Most Active Hour:** {most_active_hour[0]} ({most_active_hour[1]} messages)")
        lines.append(f"\n**Top Active Hours:**")
        for hour_key, count in analytics.get("hourly_activity", {}).items():
            h = int(hour_key) if isinstance(hour_key, (int, str)) and str(hour_key).isdigit() else hour_key
            if isinstance(h, int):
                label = f"{h:02d}:00-{(h+1)%24:02d}:00"
            else:
                label = str(h)
            lines.append(f"  • {label}: {count} messages")

    # ── Sorry count ───────────────────────────────────────────────────────────
    elif re.search(r"\bsorry\b", q):
        sorry_total = keyword_counts.get("sorry", 0)
        lines.append(f"🙏 **'Sorry' Usage:**")
        lines.append(f"  Total: {sorry_total} times across the entire chat")
        lines.append(f"\n  **Per Person:**")
        for sender in participants:
            count = sender_keyword_counts.get(sender, {}).get("sorry", 0)
            if count > 0:
                lines.append(f"    • {sender}: said 'sorry' {count} times")
        if not any(sender_keyword_counts.get(s, {}).get("sorry", 0) > 0 for s in participants):
            lines.append("    • Nobody said 'sorry' in this chat")

    # ── Thank you count ───────────────────────────────────────────────────────
    elif re.search(r"\bthank", q):
        ty_total = keyword_counts.get("thank you", 0) + keyword_counts.get("thanks", 0)
        lines.append(f"🙏 **'Thank you / Thanks' Usage:** {ty_total} times total")
        lines.append(f"\n  **Per Person:**")
        for sender in participants:
            sk = sender_keyword_counts.get(sender, {})
            count = sk.get("thank you", 0) + sk.get("thanks", 0)
            if count > 0:
                lines.append(f"    • {sender}: said thank you/thanks {count} times")

    # ── Emojis ────────────────────────────────────────────────────────────────
    elif re.search(r"emoji", q):
        if top_emojis:
            n = 5
            match = re.search(r"top\s*(\d+)", q)
            if match:
                n = int(match.group(1))
            lines.append(f"😊 **Top {n} Emojis Used:**")
            for i, (emoji, count) in enumerate(top_emojis[:n], 1):
                lines.append(f"  {i}. {emoji} — used {count} times")
            lines.append(f"\n**Per Person:**")
            for sender, emoji_dict in sender_emoji_counts.items():
                if emoji_dict:
                    top_3 = list(emoji_dict.items())[:3]
                    emoji_str = ", ".join(f"{e}({c})" for e, c in top_3)
                    lines.append(f"  • {sender}: {emoji_str}")
        else:
            lines.append("No emojis were found in this chat.")

    # ── Keywords ──────────────────────────────────────────────────────────────
    elif re.search(r"\bkeyword\b", q):
        lines.append(f"🔑 **Keyword Counts Across Chat:**")
        for kw, count in sorted(keyword_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                lines.append(f"  • '{kw}': {count} times")

    # ── Who sent the most messages? ───────────────────────────────────────────
    elif re.search(r"most\s*(active|messages?|texts?|sender|person|user|participant)|who\s*(sent|text|messag|chat)", q):
        lines.append(f"📊 **Message Count per Participant:**")
        for sender, count in counts.items():
            pct = round(count / total * 100, 1) if total else 0
            lines.append(f"  • {sender}: {count} messages ({pct}%)")
        lines.append(f"\n🏆 **Most Active:** {most_active_sender[0]} with {most_active_sender[1]} messages")

    # ── Total messages ────────────────────────────────────────────────────────
    elif re.search(r"total\s*(messages?|texts?|chats?)|how\s+many\s+messages?", q):
        lines.append(f"📨 **Total Messages:** {total}")
        lines.append(f"👥 **Participants:** {', '.join(participants)}")
        lines.append(f"📁 **Media/Files Shared:** {media_count}")
        lines.append(f"\n**Per Person:**")
        for sender, count in counts.items():
            lines.append(f"  • {sender}: {count} messages")

    # ── Participants ──────────────────────────────────────────────────────────
    elif re.search(r"participant|who\s+(is|are|was|were)\s+in|who\s+(chat|text)", q):
        lines.append(f"👥 **Participants in this chat ({len(participants)}):**")
        for sender, count in counts.items():
            lines.append(f"  • {sender}: {count} messages")

    # ── Media count ───────────────────────────────────────────────────────────
    elif re.search(r"media|image|video|photo|file|audio|sticker", q):
        lines.append(f"📎 **Media/Files Shared:** {media_count}")

    # ── General stats / overview ──────────────────────────────────────────────
    else:
        # Full overview
        lines.append(f"📊 **WhatsApp Chat Overview:**")
        lines.append(f"  Total Messages: {total}")
        lines.append(f"  Participants: {', '.join(participants)}")
        lines.append(f"  Media Shared: {media_count}")
        lines.append(f"\n**Messages per Person:**")
        for sender, count in counts.items():
            pct = round(count / total * 100, 1) if total else 0
            lines.append(f"  • {sender}: {count} ({pct}%)")
        lines.append(f"\n**Most Active Day:** {most_active_day[0]} ({most_active_day[1]} messages)")
        lines.append(f"**Most Active Hour:** {most_active_hour[0]} ({most_active_hour[1]} messages)")
        if top_emojis:
            top_3 = ", ".join(f"{e}({c})" for e, c in top_emojis[:3])
            lines.append(f"**Top Emojis:** {top_3}")
        if top_words:
            top_5w = ", ".join(f"'{w}'({c})" for w, c in top_words[:5])
            lines.append(f"**Top Words:** {top_5w}")
        sorry_count = keyword_counts.get("sorry", 0)
        if sorry_count:
            lines.append(f"**'Sorry' count:** {sorry_count} times total")

    return "\n".join(lines)


def entity_search(
    messages: List[Dict[str, Any]],
    query: str,
    context_window: int = 5,
    max_results: int = 8
) -> str:
    """
    Full-text search across all parsed messages for entity/name/topic mentions.
    Returns expanded context: previous + matching + next messages merged together.

    Args:
      messages:       all parsed messages from the WhatsApp export
      query:          user's question
      context_window: number of messages before/after each match to include
      max_results:    max number of match groups to return

    Returns a rich formatted string of conversations ready for LLM injection.
    """
    if not messages:
        return ""

    # Extract search terms from query (words 3+ chars, not stopwords)
    stop = {"who", "what", "when", "where", "why", "how", "the", "and", "for",
            "are", "was", "were", "did", "does", "tell", "show", "about",
            "this", "that", "with", "from", "said", "have", "explain", "discuss"}
    terms = [
        w.lower() for w in re.findall(r"\b\w{3,}\b", query)
        if w.lower() not in stop
    ]

    if not terms:
        return ""

    # Find all matching message indices
    matching_indices = []
    for i, msg in enumerate(messages):
        text = msg["message"].lower()
        sender = msg["sender"].lower()
        # Match if any term appears in message text OR sender name
        if any(term in text or term in sender for term in terms):
            matching_indices.append(i)

    if not matching_indices:
        return f"[No messages found mentioning: {', '.join(terms)}]"

    # Group nearby matches to avoid duplicating context
    groups = []
    current_group = [matching_indices[0]]
    for idx in matching_indices[1:]:
        if idx - current_group[-1] <= context_window * 2:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    groups.append(current_group)

    # Limit to max_results groups
    groups = groups[:max_results]

    total_matches = len(matching_indices)
    sections = [
        f"🔍 Found {total_matches} message(s) mentioning '{', '.join(terms)}' "
        f"(showing {len(groups)} conversation group(s) with ±{context_window} message context):\n"
    ]

    for group_idx, group in enumerate(groups, 1):
        start_idx = max(0, group[0] - context_window)
        end_idx = min(len(messages) - 1, group[-1] + context_window)
        context_msgs = messages[start_idx:end_idx + 1]

        sections.append(f"--- Group {group_idx} of {len(groups)} ---")
        sections.append(f"(Messages {start_idx+1}–{end_idx+1} of {len(messages)})")

        for msg in context_msgs:
            # Highlight matching messages
            text = msg["message"]
            is_match = any(term in text.lower() or term in msg["sender"].lower() for term in terms)
            prefix = "▶ " if is_match else "  "
            sections.append(
                f"{prefix}[{msg['date']} {msg['time']}] {msg['sender']}: {text}"
            )

        sections.append("")  # blank line between groups

    return "\n".join(sections)


# ── Analytics text formatter ───────────────────────────────────────────────────

def analytics_to_text(analytics: Dict[str, Any]) -> str:
    """
    Convert analytics dict to a structured text block for LLM injection.
    Used as the context block injected into every RAG prompt for WhatsApp files.
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

    lines.append("\n--- Keyword Counts (Total) ---")
    for kw, count in analytics.get("keyword_counts", {}).items():
        if count > 0:
            lines.append(f"  '{kw}': {count} times")

    lines.append("\n--- 'Sorry' Per Person ---")
    skc = analytics.get("sender_keyword_counts", {})
    for sender in analytics.get("participants", []):
        sorry = skc.get(sender, {}).get("sorry", 0)
        lines.append(f"  {sender}: said 'sorry' {sorry} times")

    lines.append("\n--- Top Words ---")
    for word, count in analytics.get("top_words", [])[:10]:
        lines.append(f"  '{word}': {count}")

    if analytics.get("top_emojis"):
        lines.append("\n--- Top Emojis ---")
        for emoji, count in analytics.get("top_emojis", []):
            lines.append(f"  {emoji}: {count}")

    lines.append(f"\n--- Activity ---")
    lines.append(
        f"Most Active Day: {analytics.get('most_active_day', ('N/A', 0))[0]} "
        f"({analytics.get('most_active_day', ('N/A', 0))[1]} messages)"
    )
    lines.append(
        f"Most Active Hour: {analytics.get('most_active_hour', ('N/A', 0))[0]} "
        f"({analytics.get('most_active_hour', ('N/A', 0))[1]} messages)"
    )
    lines.append(
        f"Most Active Month: {analytics.get('most_active_month', ('N/A', 0))[0]} "
        f"({analytics.get('most_active_month', ('N/A', 0))[1]} messages)"
    )
    lines.append(f"Media Messages: {analytics.get('media_messages', 0)}")

    return "\n".join(lines)


# ── LangChain document builder ─────────────────────────────────────────────────

def messages_to_documents(messages: List[Dict[str, Any]]) -> List[Tuple[str, dict]]:
    """
    Convert parsed messages to (text, metadata) tuples for LangChain Document creation.
    Groups messages into chunks of 30 (smaller than 50 for better retrieval granularity).
    """
    docs = []
    chunk_size = 30  # reduced from 50 for better semantic granularity
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
            "end_date": chunk[-1]["date"],
            "start_msg_idx": i,
            "end_msg_idx": i + len(chunk) - 1
        }
        docs.append((text, metadata))
    return docs
