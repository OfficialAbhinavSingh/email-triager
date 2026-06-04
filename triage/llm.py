"""
Pluggable LLM backend.

The recruiter said: "Any LLM provider/model. If you don't have keys, stub the
LLM call behind an interface and we'll wire ours in." This module is that
interface. The triage logic (prompt + schema + validation) lives in
extractor.py and is provider-agnostic; only this file knows about Anthropic.

To wire in a different provider, implement the `LLMBackend` protocol and pass
an instance to `extract(email_text, backend=YourBackend())`.
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dotenv import load_dotenv

# Load .env from project root if present (no-op if absent).
# override=True so the project's .env is authoritative — otherwise a stale
# ANTHROPIC_API_KEY exported in the shell (e.g. an sk-ant-oat01 OAuth token,
# which the Messages API rejects with 401) would silently shadow the real key.
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")


@runtime_checkable
class LLMBackend(Protocol):
    """A backend turns (system prompt, user message, tool schema) into the
    tool-call arguments dict matching the tool's input_schema.

    Implement this one method to swap in any provider/model."""

    def extract_tool_call(self, system: str, user: str, tool: dict) -> dict[str, Any]:
        ...


class AnthropicBackend:
    """Default backend: Anthropic Claude with forced tool use.

    The model is read from the LLM_MODEL env var (default claude-sonnet-4-6),
    so nothing is hardcoded — set LLM_MODEL to point at any Claude model.
    """

    def __init__(self, model: str | None = None, max_retries: int = 1, max_tokens: int = 1024):
        import anthropic  # imported lazily so the package imports without the SDK present

        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model or DEFAULT_MODEL
        self.max_retries = max_retries
        self.max_tokens = max_tokens

    def extract_tool_call(self, system: str, user: str, tool: dict) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool["name"]},
                    messages=[{"role": "user", "content": user}],
                )
                for block in response.content:
                    if block.type == "tool_use" and block.name == tool["name"]:
                        return block.input
                raise RuntimeError(f"Model returned no tool call. Content: {response.content}")
            except Exception as exc:  # transient overloads (529), rate limits, etc.
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(2)
        raise last_exc  # type: ignore[misc]


_default_backend: LLMBackend | None = None


def default_backend() -> LLMBackend:
    """Lazily construct the default Anthropic backend (singleton)."""
    global _default_backend
    if _default_backend is None:
        _default_backend = AnthropicBackend()
    return _default_backend
