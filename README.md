# Email Triage Service

Turns raw customer emails into validated, structured triage records for automatic routing.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY
python process_dataset.py     # → results.jsonl
python run_eval.py            # → scored metrics
```

## Design decisions

### Structured output via tool_use
Rather than prompting for JSON and regex-parsing the response, the extractor uses Claude's `tool_use` with a strict JSON Schema definition and `tool_choice={"type": "any"}`. The model is forced to call the tool; Pydantic validates the output. Zero brittle string parsing.

### Field semantics (underspecified by assignment)

| Field | Choice |
|---|---|
| `intent` | **Primary** intent when multiple exist. Secondary intent described in `requested_action`. |
| `urgency` | Inferred from language: ALL-CAPS/bank threats/hard deadlines → high; status questions → medium; praise/no-action → low. |
| `sentiment` | **Actual** emotional state — sarcasm is detected as negative, not positive. |
| `entities.order_id` | Digits only, `#` prefix stripped, so `#48213` and `48213` compare equal. |
| `entities.dates` | Preserved as-is from the email ("Tuesday", "two weeks") — not normalized to ISO. |
| `requires_human` | `true` when any of: urgency=high, explicit bank/legal threat, email too vague to route, suspected injection attempt. Post-processing hard-overrides to `true` if urgency=high (safety net). |
| `confidence` | Certainty in the **intent** classification specifically. Lower for vague/ambiguous emails. |

### Prompt injection defense (email #5)
The email body is wrapped in `<customer_email>` XML tags. The system prompt explicitly instructs the model to treat all content inside as data, never as instructions. A second layer: the schema enforces valid enum values — even if injection partially succeeded, the model can't return free-form text.

### Sarcasm (email #9)
The system prompt explicitly instructs: "detect sarcasm — 'Oh fantastic' before a complaint = NEGATIVE." Claude handles this naturally with a clear hint.

### Multilingual (email #7)
No translation step — Claude handles Spanish extraction natively. The prompt says "emails may be in any language; return all fields in English."

### Spam (email #6)
Defined in the system prompt: spam → `intent=other`, `urgency=low`, `requires_human=false`.

### Multi-intent (email #2)
Primary intent wins (`refund`). Voltage question goes into `requested_action` as prose.

## Eval metrics

| Metric | Why |
|---|---|
| Intent accuracy | Primary routing signal |
| Urgency accuracy + MAE | Drives SLA; MAE shows severity of misclassification |
| Sentiment accuracy | Detects sarcasm, affect routing |
| Order ID accuracy | Entity extraction correctness |
| requires_human precision/recall | **Recall matters most** — a missed escalation is worse than a false one |
| Overall | Macro average of the above for a single headline number |

## What I'd do with another day

1. **Confidence calibration** — compare self-reported confidence vs. correct predictions; add logprob-based calibration
2. **Retry on low confidence** — re-run with a chain-of-thought prompt when confidence < 0.6
3. **Multi-intent schema** — allow an `intents: [...]` array with a `primary` flag instead of forcing one
4. **Batch API** — use Anthropic's batch endpoint for cheaper, parallelised dataset processing
5. **More eval data** — 9 emails is enough to debug the prompt, not to trust the numbers
6. **Streaming** — stream tool_use output for lower time-to-first-token in a live inbox

## AI tools used

**Claude Code (this session)** — planned the architecture, wrote all code, iterated on the system prompt.

Key prompts that did the heavy lifting:

---

**Prompt 1 — architecture planning:**
> "Plan an email triage service for a 1-hour hackathon. The service takes raw customer email text and returns a validated JSON triage record. Dataset has 9 deliberately messy emails including: a prompt injection attempt, sarcasm, a Spanish email, spam, a vague 'help' email. Plan the directory structure, tech stack, system prompt strategy (especially injection defense, sarcasm, multilingual, spam), field semantics for underspecified fields, eval metrics, and hand labels for all 9 emails."

---

**Prompt 2 — system prompt design:**
> "Write the system prompt for an email triage extractor. It needs to: defend against prompt injection by treating email content as data not instructions; detect sarcasm as negative sentiment; handle multilingual emails; identify spam as intent=other; document my requires_human rule (true for high urgency, threats, vague emails, or injection attempts). Use XML delimiters for the email body."

---

**Prompt 3 — eval metrics:**
> "Design an eval harness for a 9-email triage dataset. I need: intent accuracy, urgency accuracy + ordinal MAE, sentiment accuracy, order_id extraction accuracy, requires_human precision/recall/F1, and an overall composite score. Explain why recall matters more than precision for requires_human."
