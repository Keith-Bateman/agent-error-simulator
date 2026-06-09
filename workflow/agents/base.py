"""
BaseAgent — async Ollama /api/chat client with structured-output conventions.

All agents follow the same turn-based protocol:
  1. Build a messages list (system + alternating user/assistant turns).
  2. Send to Ollama via POST /api/chat.
  3. Receive the assistant content string.
  4. Try to parse as JSON; fall back to raw string on failure.

Tool calling is handled at the coordinator level via a request/response
handshake embedded in the conversation:
  - Agent requests a tool by returning JSON with an "action" key.
  - Coordinator executes the tool, appends a user turn with the result.
  - Agent provides its final answer in the next assistant turn.
"""

import json
import logging
from typing import Dict, List, Optional

import httpx

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 90


class OllamaResponse:
    """Parsed result of a single /api/chat call."""

    def __init__(self, content, parsed=None, model="", prompt_tokens=0, completion_tokens=0, injected=False):
        # type: (str, Optional[dict], str, int, int, bool) -> None
        self.content = content
        self.parsed = parsed
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.injected = injected


class BaseAgent:
    """
    Async Ollama chat agent.

    Parameters
    ──────────
    name          : display name used in logs
    model         : Ollama model tag (e.g. "mistral")
    base_url      : http://host:port  (or proxy URL when CTE enabled)
    timeout       : per-request timeout in seconds
    extra_options : additional Ollama options merged into every request
                    (e.g. {"num_ctx": 2048} to cap the context window)
    """

    def __init__(self, name, model, base_url, timeout=DEFAULT_TIMEOUT, extra_options=None):
        # type: (str, str, str, int, Optional[Dict]) -> None
        self.name = name
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._extra_options = extra_options or {}
        self._messages = []  # type: List[Dict]

    def reset(self):
        self._messages = []

    def push_system(self, content):
        # type: (str) -> None
        self._messages = [{"role": "system", "content": content}]

    def push_user(self, content):
        # type: (str) -> None
        self._messages.append({"role": "user", "content": content})

    def push_assistant(self, content):
        # type: (str) -> None
        self._messages.append({"role": "assistant", "content": content})

    async def chat(self):
        # type: () -> OllamaResponse
        """
        Send the current message list to Ollama and return a parsed response.
        Does NOT mutate self._messages — callers must push_assistant() to continue.
        """
        options = {"temperature": 0.1, "num_predict": 1024}
        options.update(self._extra_options)
        payload = {
            "model": self.model,
            "messages": self._messages,
            "stream": False,
            "options": options,
        }

        log.debug("[%s] POST %s/api/chat  messages=%d",
                  self.name, self.base_url, len(self._messages))

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                "{}/api/chat".format(self.base_url),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data.get("message", {}).get("content", "").strip()
        parsed = _try_parse_json(content)

        tokens_prompt = data.get("prompt_eval_count", 0)
        tokens_comp = data.get("eval_count", 0)

        log.info(
            "[%s] response  tokens=%d/%d  json=%s",
            self.name, tokens_prompt, tokens_comp, parsed is not None,
        )
        log.debug("[%s] raw content: %s", self.name, content[:300])

        return OllamaResponse(
            content=content,
            parsed=parsed,
            model=data.get("model", self.model),
            prompt_tokens=tokens_prompt,
            completion_tokens=tokens_comp,
        )

    def synthetic_response(self, content):
        # type: (str) -> OllamaResponse
        """Wrap a pre-crafted string as if it came from Ollama."""
        return OllamaResponse(
            content=content,
            parsed=_try_parse_json(content),
            model=self.model,
            injected=True,
        )


def _try_parse_json(text):
    # type: (str) -> Optional[dict]
    """
    Try to extract JSON from the text.  Handles:
      - bare JSON objects / arrays
      - JSON fenced in ```json ... ``` blocks
    Returns None if no valid JSON found.
    """
    if not text:
        return None
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fenced code block
    for fence in ("```json", "```"):
        if fence in text:
            start = text.find(fence) + len(fence)
            end = text.find("```", start)
            if end != -1:
                snippet = text[start:end].strip()
                try:
                    return json.loads(snippet)
                except (json.JSONDecodeError, ValueError):
                    pass

    return None
