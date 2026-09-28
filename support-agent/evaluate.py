import json
from pathlib import Path
from agent import run_agent


def evaluate_case(case: dict) -> dict:
    result = run_agent(case["ticket_id"])
    text = str(result["messages"][-1].content).lower()
    required = [x.lower() for x in case["must_contain"]]
    forbidden = [x.lower() for x in case["must_not_contain"]]
    checks = {
        "required_present": all(x in text for x in required),
        "forbidden_absent": all(x not in text for x in forbidden),
    }
    return {
        "ticket_id": case["ticket_id"],
        "passed": all(checks.values()),
        "checks": checks,
        "output": text,
    }


def main():
    cases = json.loads(Path("tests/cases.json").read_text(encoding="utf-8"))
    results = [evaluate_case(case) for case in cases]
    score = sum(r["passed"] for r in results) / len(results)
    print(json.dumps({"score": score, "results": results}, indent=2))
    raise SystemExit(0 if score >= 0.80 else 1)


if __name__ == "__main__":
    main()
