from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import clickless.config  # noqa: F401


DEFAULT_QWEN_BASE_URL = "https://openai.rc.asu.edu/v1"
DEFAULT_QWEN_MODEL = "qwen38-27b"
DEFAULT_QWEN_TIMEOUT = 120.0


class QwenError(RuntimeError):
    pass


class QwenUnavailable(QwenError):
    pass


class QwenResponseError(QwenError):
    pass


@dataclass(frozen=True)
class QwenClient:
    base_url: str = DEFAULT_QWEN_BASE_URL
    model: str = DEFAULT_QWEN_MODEL
    api_key: str | None = None
    timeout: float = DEFAULT_QWEN_TIMEOUT

    @classmethod
    def from_env(cls) -> "QwenClient":
        timeout_value = os.getenv(
            "CLICKLESS_QWEN_TIMEOUT",
            str(int(DEFAULT_QWEN_TIMEOUT)),
        )
        try:
            timeout = float(timeout_value)
        except ValueError as exc:
            raise QwenError("CLICKLESS_QWEN_TIMEOUT must be a number") from exc
        if timeout <= 0:
            raise QwenError("CLICKLESS_QWEN_TIMEOUT must be greater than zero")
        base_url = os.getenv(
            "CLICKLESS_QWEN_BASE_URL",
            DEFAULT_QWEN_BASE_URL,
        ).rstrip("/")
        api_key = os.getenv("CLICKLESS_QWEN_API_KEY") or None
        if "openai.rc.asu.edu" in base_url and not api_key:
            raise QwenError("Set CLICKLESS_QWEN_API_KEY in the repo-root .env file")
        return cls(
            base_url=base_url,
            model=os.getenv("CLICKLESS_QWEN_MODEL", DEFAULT_QWEN_MODEL),
            api_key=api_key,
            timeout=timeout,
        )

    def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        if not system.strip() or not user.strip():
            raise ValueError("System and user messages are required")
        if max_tokens < 1:
            raise ValueError("max_tokens must be greater than zero")

        system_prompt = system
        if "qwen3" in self.model.lower() and "/no_think" not in system_prompt:
            system_prompt = f"{system_prompt}\n/no_think"
        effective_max_tokens = (
            max(max_tokens, 900)
            if "openai.rc.asu.edu" in self.base_url
            else max_tokens
        )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": effective_max_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        if "qwen3" in self.model.lower():
            body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            payload = self._request("POST", "chat/completions", body)
        except QwenUnavailable as exc:
            if not re.search(r"\b400\b", str(exc)):
                raise
            fallback_body = {k: v for k, v in body.items() if k != "response_format"}
            try:
                payload = self._request("POST", "chat/completions", fallback_body)
            except QwenUnavailable as retry_exc:
                if (
                    "chat_template_kwargs" not in fallback_body
                    or not re.search(r"\b400\b", str(retry_exc))
                ):
                    raise
                plain_body = {
                    k: v for k, v in fallback_body.items() if k != "chat_template_kwargs"
                }
                payload = self._request("POST", "chat/completions", plain_body)

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise QwenResponseError("Qwen returned an invalid chat response") from exc
        if not isinstance(content, str) or not content.strip():
            raise QwenResponseError("Qwen returned empty text content")
        return _extract_json_dict(content)

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "configured": bool(self.base_url and self.model),
            "available": False,
            "model": self.model,
            "base_url": self.base_url,
        }
        try:
            payload = self._request("GET", "models")
            model_ids = [
                item.get("id")
                for item in payload.get("data", [])
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            ]
            result["available"] = True
            result["model_available"] = self.model in model_ids if model_ids else None
            result["model_count"] = len(model_ids)
        except QwenError as exc:
            result["error"] = str(exc)
        return result

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=body,
            method=method,
            headers=self._headers(),
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            raise QwenUnavailable(f"Qwen server returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError) as exc:
            raise QwenUnavailable("Qwen server is unavailable") from exc
        except json.JSONDecodeError as exc:
            raise QwenResponseError("Qwen server returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise QwenResponseError("Qwen server response must be an object")
        return result

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _extract_json_dict(content: str) -> dict[str, Any]:
    """Strip <think> blocks and markdown fences, then extract a JSON dict object."""
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.S | re.I).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:[a-zA-Z0-9_-]+)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    else:
        cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", cleaned, flags=re.I).strip()

    if not cleaned:
        raise QwenResponseError("Qwen did not return valid JSON")

    try:
        direct = json.loads(cleaned)
    except json.JSONDecodeError:
        direct = None
    if isinstance(direct, dict):
        return direct

    decoder = json.JSONDecoder()
    valid_dicts: list[dict[str, Any]] = []
    idx = 0
    while True:
        pos = cleaned.find("{", idx)
        if pos < 0:
            break
        try:
            obj, end = decoder.raw_decode(cleaned, pos)
        except json.JSONDecodeError:
            idx = pos + 1
            continue
        if isinstance(obj, dict):
            valid_dicts.append(obj)
        idx = max(end, pos + 1)

    if valid_dicts:
        return valid_dicts[-1]

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        try:
            sliced = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise QwenResponseError("Qwen did not return valid JSON") from exc
        if not isinstance(sliced, dict):
            raise QwenResponseError("Qwen JSON response must be an object")
        return sliced

    if direct is not None:
        raise QwenResponseError("Qwen JSON response must be an object")
    raise QwenResponseError("Qwen did not return valid JSON")
