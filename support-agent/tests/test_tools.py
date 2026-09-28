from tools import get_ticket, search_kb


def test_known_ticket():
    result = get_ticket.invoke({"ticket_id": "T-1001"})
    assert result["ok"] is True
    assert result["ticket"]["id"] == "T-1001"


def test_invalid_ticket_id():
    result = get_ticket.invoke({"ticket_id": "anything"})
    assert result == {"ok": False, "error": "invalid_ticket_id"}


def test_kb_search_is_bounded():
    result = search_kb.invoke({"query": "duplicate billing charge"})
    assert result["ok"] is True
    assert len(result["matches"]) <= 3
