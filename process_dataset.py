#!/usr/bin/env python3
"""
Process a JSONL inbox into structured triage records.

Usage:
    python process_dataset.py                          # emails.jsonl -> results.jsonl
    python process_dataset.py --input in.jsonl --output out.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

from triage import extract


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage a JSONL inbox of emails.")
    parser.add_argument("--input", default="emails.jsonl", help="input JSONL (id, subject, body per line)")
    parser.add_argument("--output", default="results.jsonl", help="output JSONL of triage records")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    emails = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    print(f"Processing {len(emails)} emails from {input_path}...\n", file=sys.stderr)

    with output_path.open("w") as out:
        for record in emails:
            email_text = f"Subject: {record.get('subject', '')}\n\n{record['body']}"
            print(f"[{record['id']}] {record.get('subject', '')[:50]}", file=sys.stderr, end="  ")
            try:
                triage = extract(email_text)
                result = {"id": record["id"], "subject": record.get("subject", ""), **triage.model_dump()}
                out.write(json.dumps(result) + "\n")
                print(
                    f"intent={triage.intent}  urgency={triage.urgency}  "
                    f"sentiment={triage.sentiment}  human={triage.requires_human}  "
                    f"conf={triage.confidence:.2f}",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                out.write(json.dumps({"id": record["id"], "subject": record.get("subject", ""), "error": str(exc)}) + "\n")

    print(f"\nResults written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
