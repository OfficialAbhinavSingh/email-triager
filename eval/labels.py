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
    # ---- Extended set (emails_extra.jsonl) — unseen test data, ids 10-18 ----
    # Labeled strictly by the implemented field semantics:
    #   requires_human = True iff urgency=high OR bank/legal threat OR too vague to route OR injection.
    {
        "id": 10,  # billing dispute, annoyed but no threat → medium, human not required
        "intent": "billing",
        "urgency": "medium",
        "sentiment": "negative",
        "entities": {"order_id": None, "product": None, "dates": []},
        "requires_human": False,
    },
    {
        "id": 11,  # locked out + "deadline today" → high urgency → human (safety net)
        "intent": "account",
        "urgency": "high",
        "sentiment": "negative",
        "entities": {"order_id": None, "product": None, "dates": []},
        "requires_human": True,
    },
    {
        "id": 12,  # French shipping delay, polite tone → neutral, medium
        "intent": "shipping",
        "urgency": "medium",
        "sentiment": "neutral",
        "entities": {"order_id": "66902", "product": None, "dates": ["lundi"]},
        "requires_human": False,
    },
    {
        "id": 13,  # genuinely positive + a general warranty question
        "intent": "other",
        "urgency": "low",
        "sentiment": "positive",
        "entities": {"order_id": None, "product": "headphones", "dates": []},
        "requires_human": False,
    },
    {
        "id": 14,  # refund + explicit legal threat → high, human required
        "intent": "refund",
        "urgency": "high",
        "sentiment": "negative",
        "entities": {"order_id": "71150", "product": None, "dates": []},
        "requires_human": True,
    },
    {
        "id": 15,  # gibberish, no order/product context → too vague → human
        "intent": "technical_issue",
        "urgency": "medium",
        "sentiment": "negative",
        "entities": {"order_id": None, "product": None, "dates": []},
        "requires_human": True,
    },
    {
        "id": 16,  # wrong item received — fulfillment/shipping issue
        "intent": "shipping",
        "urgency": "medium",
        "sentiment": "negative",
        "entities": {"order_id": "80021", "product": None, "dates": ["Monday"]},
        "requires_human": False,
    },
    {
        "id": 17,  # calm cancellation, no threat, low urgency → contrast with #3
        "intent": "cancellation",
        "urgency": "low",
        "sentiment": "neutral",
        "entities": {"order_id": None, "product": None, "dates": []},
        "requires_human": False,
    },
    {
        "id": 18,  # B2B solicitation = spam → other, no action
        "intent": "other",
        "urgency": "low",
        "sentiment": "neutral",
        "entities": {"order_id": None, "product": None, "dates": []},
        "requires_human": False,
    },
]
