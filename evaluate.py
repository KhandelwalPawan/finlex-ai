from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rag_pipeline import get_rag_chain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight FinLex AI evaluation set.")
    parser.add_argument("--questions", type=Path, default=Path("eval_questions.json"))
    parser.add_argument("--output", type=Path, default=Path("eval_results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    chain = get_rag_chain()
    results = []
    for item in questions:
        response = chain.invoke(item["question"])
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_behavior": item.get("expected_behavior"),
                "answer": response.get("answer"),
                "confidence": response.get("confidence"),
                "citations": response.get("citations", []),
            }
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} evaluation results to {args.output}")


if __name__ == "__main__":
    main()
