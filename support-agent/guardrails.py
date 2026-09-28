import re

SENSITIVE_TERMS = {"refund", "credit", "delete account", "change email"}


def validate_ticket_id(ticket_id: str) -> None:
    if not re.fullmatch(r"T-\d{4}", ticket_id):
        raise ValueError("Ticket ID must look like T-1001")


def needs_human_approval(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in SENSITIVE_TERMS)


def grounding_warnings(text: str) -> list[str]:
    warnings = []
    lower = text.lower()
    if "policy" in lower and not re.search(r"KB-[A-Z]+-\d{3}", text):
        warnings.append("Policy mentioned without an article citation")
    if any(x in lower for x in ["has been refunded", "refund completed"]):
        warnings.append("Draft may claim an unconfirmed action")
    return warnings
