#!/usr/bin/env python3
"""Process emails.jsonl → results.jsonl. Usage: python process_dataset.py"""
import json
import sys
from pathlib import Path

from triage import extract


def main() -> None:
    input_path = Path("emails.jsonl")
    output_path = Path("results.jsonl")

    emails = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    print(f"Processing {len(emails)} emails...\n", file=sys.stderr)

    with output_path.open("w") as out:
        for record in emails:
            email_text = f"Subject: {record['subject']}\n\n{record['body']}"
            print(f"[{record['id']}] {record['subject'][:50]}", file=sys.stderr, end="  ")
            try:
                triage = extract(email_text)
                result = {
                    "id": record["id"],
                    "subject": record["subject"],
                    **triage.model_dump(),
                }
                out.write(json.dumps(result) + "\n")
                print(
                    f"intent={triage.intent}  urgency={triage.urgency}  "
                    f"sentiment={triage.sentiment}  human={triage.requires_human}  "
                    f"conf={triage.confidence:.2f}",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                out.write(json.dumps({"id": record["id"], "subject": record["subject"], "error": str(exc)}) + "\n")

    print(f"\nResults written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
