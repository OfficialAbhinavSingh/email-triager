#!/usr/bin/env python3
"""
Eval harness: compare results.jsonl against hand-labeled ground truth.

Usage:
    python run_eval.py                      # reads results.jsonl
    python run_eval.py --results out.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

from eval.labels import LABELS
from eval.metrics import score


def main() -> None:
    parser = argparse.ArgumentParser(description="Score triage results against hand labels.")
    parser.add_argument("--results", default="results.jsonl", help="JSONL produced by process_dataset.py")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        sys.exit(f"{results_path} not found. Run: python process_dataset.py")

    raw = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    errors = [r for r in raw if "error" in r]
    predictions = [r for r in raw if "error" not in r]

    if errors:
        print(f"Warning: {len(errors)} email(s) had extraction errors (excluded from eval):")
        for e in errors:
            print(f"  id={e['id']} subject={e['subject']!r}: {e['error']}")
        print()

    label_map = {l["id"]: l for l in LABELS}
    aligned_preds, aligned_labels = [], []
    for pred in sorted(predictions, key=lambda x: x["id"]):
        if pred["id"] in label_map:
            aligned_preds.append(pred)
            aligned_labels.append(label_map[pred["id"]])

    if not aligned_preds:
        sys.exit("No valid predictions to evaluate.")

    s = score(aligned_preds, aligned_labels)

    W = 60
    print("=" * W)
    print("EMAIL TRIAGE — EVAL RESULTS")
    print("=" * W)
    print(f"\nEmails evaluated: {s['n']} / {len(LABELS)}")
    print(f"\nClassification:")
    print(f"  Intent accuracy:      {s['intent_accuracy']:.0%}  ({round(s['intent_accuracy']*s['n'])}/{s['n']})")
    print(f"  Urgency accuracy:     {s['urgency_accuracy']:.0%}  ({round(s['urgency_accuracy']*s['n'])}/{s['n']})")
    print(f"  Urgency MAE:          {s['urgency_mae']:.2f} levels")
    print(f"  Sentiment accuracy:   {s['sentiment_accuracy']:.0%}  ({round(s['sentiment_accuracy']*s['n'])}/{s['n']})")
    if s["order_id_accuracy"] is not None:
        print(f"  Order ID accuracy:    {s['order_id_accuracy']:.0%}")
    print(f"\nRequires-human (routing safety):")
    print(f"  Precision:            {s['requires_human_precision']:.0%}")
    print(f"  Recall:               {s['requires_human_recall']:.0%}")
    print(f"  F1:                   {s['requires_human_f1']:.0%}")
    print(f"\nConfidence calibration:")
    cwc, cww = s["confidence_when_correct"], s["confidence_when_wrong"]
    print(f"  Avg confidence when intent CORRECT: {cwc if cwc is not None else 'n/a'}")
    print(f"  Avg confidence when intent WRONG:   {cww if cww is not None else 'n/a'}")
    if cwc is not None and cww is not None:
        verdict = "well-calibrated" if cwc > cww else "MISCALIBRATED (overconfident on errors)"
        print(f"  → {verdict}")
    print(f"\nOverall score:          {s['overall_score']:.0%}  (avg of intent/urgency/sentiment acc + F1)")
    print(f"\n{'─' * W}")
    print("Per-email breakdown:")
    print(f"{'─' * W}")
    for row in s["per_email"]:
        print(f"\n  Email {row['id']}:")
        print(f"    intent:         {row['intent']}")
        print(f"    urgency:        {row['urgency']}")
        print(f"    sentiment:      {row['sentiment']}")
        print(f"    requires_human: {row['requires_human']}")
    print(f"\n{'=' * W}")


if __name__ == "__main__":
    main()
