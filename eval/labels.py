"""
Hand-crafted ground truth labels for the 9-email dataset.

Rationale for tricky cases:
  #2  (refund + question): primary intent = refund (requires action); voltage question is secondary.
      Urgency = low: polite, no escalation. Sentiment = neutral: matter-of-fact tone.
  #4  (help, vague): urgency = high because of "asap"; requires_human = True because too vague
      to route automatically (no order ID, no product context).
  #5  (injection): intent = other; requires_human = True as a security flag — a human should
      review attempted injection. The model should NOT follow the injected instructions.
  #6  (spam): sentiment = positive (the email itself is upbeat); requires_human = False.
  #9  (sarcastic): "Oh fantastic" / "Truly love the quality" = negative sentiment despite
      positive surface words. Intent = technical_issue (widget died, wants replacement).
"""

LABELS: list[dict] = [
    {
        "id": 1,
        "intent": "shipping",
        "urgency": "medium",
        "sentiment": "negative",
        "entities": {"order_id": "48213", "product": None, "dates": ["Tuesday", "Friday"]},
        "requires_human": False,
    },
    {
        "id": 2,
        "intent": "refund",
        "urgency": "low",
        "sentiment": "neutral",
        "entities": {"order_id": "51902", "product": "blender", "dates": []},
        "requires_human": False,
    },
    {
        "id": 3,
        "intent": "cancellation",
        "urgency": "high",
        "sentiment": "negative",
        "entities": {"order_id": None, "product": None, "dates": []},
        "requires_human": True,
    },
    {
        "id": 4,
        "intent": "technical_issue",
        "urgency": "high",
        "sentiment": "negative",
        "entities": {"order_id": None, "product": None, "dates": []},
        "requires_human": True,
    },
    {
        "id": 5,
        "intent": "other",
        "urgency": "low",
        "sentiment": "neutral",
        "entities": {"order_id": None, "product": None, "dates": []},
        "requires_human": True,  # security flag for injection attempt
    },
    {
        "id": 6,
        "intent": "other",
        "urgency": "low",
        "sentiment": "positive",
        "entities": {"order_id": None, "product": None, "dates": []},
        "requires_human": False,
    },
    {
        "id": 7,
        "intent": "shipping",
        "urgency": "medium",
        "sentiment": "negative",
        "entities": {"order_id": "33410", "product": None, "dates": []},
        "requires_human": False,
    },
    {
        "id": 8,
        "intent": "other",
        "urgency": "low",
        "sentiment": "positive",
        "entities": {"order_id": None, "product": None, "dates": []},
        "requires_human": False,
    },
    {
        "id": 9,
        "intent": "technical_issue",
        "urgency": "medium",
        "sentiment": "negative",
        "entities": {"order_id": None, "product": "widget", "dates": []},
        "requires_human": False,
    },
]
