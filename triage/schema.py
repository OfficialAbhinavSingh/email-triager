from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class Entities(BaseModel):
    order_id: Optional[str] = None
    product: Optional[str] = None
    dates: list[str] = Field(default_factory=list)


class TriageRecord(BaseModel):
    intent: Literal["refund", "shipping", "technical_issue", "billing", "cancellation", "account", "other"]
    urgency: Literal["low", "medium", "high"]
    sentiment: Literal["positive", "neutral", "negative"]
    entities: Entities
    requested_action: str
    requires_human: bool
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def enforce_human_on_high_urgency(self) -> "TriageRecord":
        # Safety net: high urgency always escalates regardless of LLM output
        if self.urgency == "high":
            self.requires_human = True
        return self
