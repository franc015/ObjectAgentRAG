"""Clients de modèles : API Anthropic (Messages) et serveurs compatibles OpenAI (modèles locaux).

Implémentés avec la bibliothèque standard (urllib) : aucune dépendance. Le format interne des
messages est neutre ; chaque client le convertit vers son format réseau.
"""

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    pass


@dataclass
class Turn:
    text: str = ""
    tool_calls: list = field(default_factory=list)   # [{"id","name","args"}]
    usage: dict = field(default_factory=lambda: {"input": 0, "output": 0})
    stop: str = ""


def _post(url: str, headers: dict, payload: dict, timeout: int = 120, retries: int = 4) -> dict:
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            e.close()
            if e.code in (429, 500, 502, 503, 529) and attempt < retries:
                try:
                    delay = float(e.headers.get("retry-after") or 2 ** attempt)
                except ValueError:
                    delay = 2 ** attempt
                time.sleep(min(delay, 30))
                continue
            raise LLMError(f"HTTP {e.code} : {body[:500]}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise LLMError(f"connexion impossible : {e}") from None
    raise LLMError("échec après plusieurs tentatives")


class LLM(ABC):
    provider = "abstract"
    model = ""

    @abstractmethod
    def turn(self, system: str, messages: list, tools: list) -> Turn: ...


class AnthropicLLM(LLM):
    provider = "anthropic"

    def __init__(self, model, api_key, base_url="", max_tokens=2048, timeout=120):
        if not api_key:
            raise LLMError("clé d'API absente : définissez la variable d'environnement indiquée "
                           "dans [llm].api_key_env")
        self.model, self.key, self.max_tokens, self.timeout = model, api_key, max_tokens, timeout
        self.base = (base_url or "https://api.anthropic.com").rstrip("/")

    @staticmethod
    def wire(messages: list) -> list:
        out = []
        for m in messages:
            r = m["role"]
            if r == "user":
                out.append({"role": "user", "content": m["content"]})
            elif r == "assistant":
                blocks = []
                if m.get("text"):
                    blocks.append({"type": "text", "text": m["text"]})
                for c in m.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(vide)"}]})
            elif r == "tool_results":
                blocks = [{"type": "tool_result", "tool_use_id": x["id"], "content": x["content"]}
                          for x in m["results"]]
                if m.get("note"):
                    blocks.append({"type": "text", "text": m["note"]})
                out.append({"role": "user", "content": blocks})
        return out

    def turn(self, system, messages, tools):
        payload = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                   "messages": self.wire(messages)}
        if tools:
            payload["tools"] = [{"name": t["name"], "description": t["description"],
                                 "input_schema": t["parameters"]} for t in tools]
        resp = _post(f"{self.base}/v1/messages",
                     {"x-api-key": self.key, "anthropic-version": "2023-06-01",
                      "content-type": "application/json"}, payload, self.timeout)
        blocks = resp.get("content", [])
        u = resp.get("usage", {})
        return Turn(
            text="".join(b.get("text", "") for b in blocks if b.get("type") == "text"),
            tool_calls=[{"id": b["id"], "name": b["name"], "args": b.get("input") or {}}
                        for b in blocks if b.get("type") == "tool_use"],
            usage={"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0)},
            stop=resp.get("stop_reason", ""))


class OpenAICompatLLM(LLM):
    """Serveur local ou tiers exposant /chat/completions (Ollama, vLLM, llama.cpp, etc.)."""

    provider = "openai_compat"

    def __init__(self, model, api_key="", base_url="http://localhost:11434/v1", max_tokens=2048, timeout=120):
        self.model, self.key, self.max_tokens, self.timeout = model, api_key, max_tokens, timeout
        self.base = base_url.rstrip("/")

    @staticmethod
    def wire(system: str, messages: list) -> list:
        out = [{"role": "system", "content": system}]
        for m in messages:
            r = m["role"]
            if r == "user":
                out.append({"role": "user", "content": m["content"]})
            elif r == "assistant":
                msg = {"role": "assistant", "content": m.get("text") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                          "function": {"name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}}
                                         for c in m["tool_calls"]]
                out.append(msg)
            elif r == "tool_results":
                for x in m["results"]:
                    out.append({"role": "tool", "tool_call_id": x["id"], "content": x["content"]})
                if m.get("note"):
                    out.append({"role": "user", "content": m["note"]})
        return out

    def turn(self, system, messages, tools):
        payload = {"model": self.model, "max_tokens": self.max_tokens, "messages": self.wire(system, messages)}
        if tools:
            payload["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in tools]
        headers = {"content-type": "application/json"}
        if self.key:
            headers["authorization"] = f"Bearer {self.key}"
        resp = _post(f"{self.base}/chat/completions", headers, payload, self.timeout)
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        calls = []
        for c in msg.get("tool_calls") or []:
            raw = c.get("function", {}).get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                args = {"__parse_error": raw}
            calls.append({"id": c.get("id") or f"call_{len(calls)}", "name": c["function"]["name"], "args": args})
        u = resp.get("usage", {})
        return Turn(text=msg.get("content") or "", tool_calls=calls,
                    usage={"input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0)},
                    stop=choice.get("finish_reason", ""))


def make_llm(cfg: dict, provider=None, model=None) -> LLM:
    c = cfg["llm"]
    provider, model = provider or c["provider"], model or c["model"]
    key = os.environ.get(c.get("api_key_env", ""), "")
    if provider == "anthropic":
        return AnthropicLLM(model, key, c.get("base_url", ""), c["max_tokens"], c["timeout"])
    if provider == "openai_compat":
        return OpenAICompatLLM(model, key, c.get("base_url") or "http://localhost:11434/v1",
                               c["max_tokens"], c["timeout"])
    raise LLMError(f"fournisseur inconnu : {provider}")


def make_dates_llm(cfg: dict) -> LLM:
    """Modèle léger dédié à la détection des dates (section [dates] de la configuration)."""
    d = cfg["dates"]
    sub = dict(cfg)
    sub["llm"] = {**cfg["llm"], "provider": d["provider"], "model": d["model"],
                  "base_url": d.get("base_url", ""), "max_tokens": 200}
    return make_llm(sub)
