"""Scoring functions for the triage eval harness."""
from __future__ import annotations
from typing import Any

URGENCY_RANK = {"low": 0, "medium": 1, "high": 2}


def _normalize_order_id(v: str | None) -> str | None:
    """Strip leading # so '#48213' and '48213' compare equal."""
    return v.lstrip("#") if v else None


def score(predictions: list[dict], labels: list[dict]) -> dict[str, Any]:
    n = len(labels)
    assert len(predictions) == n, "Prediction/label count mismatch"

    intent_hits = urgency_hits = sentiment_hits = 0
    urgency_mae = 0
    order_correct = order_total = 0

    rh_tp = rh_fp = rh_fn = rh_tn = 0
    # confidence calibration: average reported confidence when intent is right vs. wrong
    conf_correct: list[float] = []
    conf_wrong: list[float] = []
    per_email: list[dict] = []

    for pred, label in zip(predictions, labels):
        i_match = pred["intent"] == label["intent"]
        u_match = pred["urgency"] == label["urgency"]
        s_match = pred["sentiment"] == label["sentiment"]
        pred_rh: bool = pred.get("requires_human", False)
        label_rh: bool = label["requires_human"]

        conf = pred.get("confidence")
        if isinstance(conf, (int, float)):
            (conf_correct if i_match else conf_wrong).append(float(conf))

        intent_hits += int(i_match)
        urgency_hits += int(u_match)
        sentiment_hits += int(s_match)
        urgency_mae += abs(
            URGENCY_RANK.get(pred["urgency"], 1) - URGENCY_RANK[label["urgency"]]
        )

        # Order ID: only score when label has one (null labels are skipped)
        label_oid = _normalize_order_id(label["entities"]["order_id"])
        if label_oid is not None:
            order_total += 1
            pred_oid = _normalize_order_id(pred.get("entities", {}).get("order_id"))
            order_correct += int(pred_oid == label_oid)

        if pred_rh and label_rh:
            rh_tp += 1
        elif pred_rh and not label_rh:
            rh_fp += 1
        elif not pred_rh and label_rh:
            rh_fn += 1
        else:
            rh_tn += 1

        per_email.append({
            "id": label["id"],
            "intent":        "✓" if i_match else f"✗  pred={pred['intent']!r:<20} label={label['intent']!r}",
            "urgency":       "✓" if u_match else f"✗  pred={pred['urgency']!r:<8} label={label['urgency']!r}",
            "sentiment":     "✓" if s_match else f"✗  pred={pred['sentiment']!r:<10} label={label['sentiment']!r}",
            "requires_human": "✓" if pred_rh == label_rh else f"✗  pred={pred_rh}  label={label_rh}",
        })

    rh_prec = rh_tp / (rh_tp + rh_fp) if (rh_tp + rh_fp) else 0.0
    rh_rec  = rh_tp / (rh_tp + rh_fn) if (rh_tp + rh_fn) else 0.0
    rh_f1   = 2 * rh_prec * rh_rec / (rh_prec + rh_rec) if (rh_prec + rh_rec) else 0.0

    intent_acc    = intent_hits / n
    urgency_acc   = urgency_hits / n
    sentiment_acc = sentiment_hits / n
    overall       = (intent_acc + urgency_acc + sentiment_acc + rh_f1) / 4

    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else None

    return {
        "n": n,
        "intent_accuracy":           round(intent_acc, 3),
        "urgency_accuracy":          round(urgency_acc, 3),
        "urgency_mae":               round(urgency_mae / n, 3),
        "sentiment_accuracy":        round(sentiment_acc, 3),
        "order_id_accuracy":         round(order_correct / order_total, 3) if order_total else None,
        "requires_human_precision":  round(rh_prec, 3),
        "requires_human_recall":     round(rh_rec, 3),
        "requires_human_f1":         round(rh_f1, 3),
        "overall_score":             round(overall, 3),
        # calibration: a well-calibrated model is more confident when it is correct
        "confidence_when_correct":   avg(conf_correct),
        "confidence_when_wrong":     avg(conf_wrong),
        "per_email":                 per_email,
    }
