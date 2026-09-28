from agent import run_agent
from guardrails import validate_ticket_id, needs_human_approval, grounding_warnings


def main():
    ticket_id = input("Ticket ID [T-1001]: ").strip() or "T-1001"
    validate_ticket_id(ticket_id)
    result = run_agent(ticket_id)
    draft = str(result["messages"][-1].content)

    print("\n--- Agent draft ---")
    print(draft)

    warnings = grounding_warnings(draft)
    if warnings:
        print("\nGuardrail warnings:")
        for item in warnings:
            print(f"- {item}")

    if needs_human_approval(draft):
        decision = input("\nApprove this sensitive draft? [y/N]: ").strip().lower()
        print("Approved" if decision == "y" else "Not approved; escalate for review")
    else:
        print("\nNo sensitive action detected. Draft is ready for normal review.")


if __name__ == "__main__":
    main()
