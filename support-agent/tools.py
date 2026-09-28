import json
import re
from pathlib import Path
from langchain.tools import tool

DATA_DIR = Path(__file__).parent / "data"


def _load(name: str) -> list[dict]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


@tool
def get_ticket(ticket_id: str) -> dict:
    """Return one support ticket by exact ticket ID."""
    if not re.fullmatch(r"T-\d{4}", ticket_id):
        return {"ok": False, "error": "invalid_ticket_id"}
    ticket = next((x for x in _load("tickets.json") if x["id"] == ticket_id), None)
    return {"ok": bool(ticket), "ticket": ticket}


@tool
def search_kb(query: str) -> dict:
    """Search approved support articles and return up to three matching excerpts."""
    terms = {t.lower() for t in re.findall(r"[a-zA-Z]{3,}", query)}
    scored = []
    for article in _load("kb.json"):
        haystack = f'{article["title"]} {article["text"]}'.lower()
        score = sum(term in haystack for term in terms)
        if score:
            scored.append((score, article))
    matches = [item for _, item in sorted(scored, reverse=True)[:3]]
    return {"ok": True, "matches": matches}


@tool
def propose_escalation(ticket_id: str, reason: str) -> dict:
    """Create a non-executing escalation proposal that still requires approval."""
    if not re.fullmatch(r"T-\d{4}", ticket_id):
        return {"ok": False, "error": "invalid_ticket_id"}
    clean_reason = reason.strip()[:300]
    if len(clean_reason) < 10:
        return {"ok": False, "error": "reason_too_short"}
    return {
        "ok": True,
        "status": "approval_required",
        "proposal": {"ticket_id": ticket_id, "reason": clean_reason},
    }
