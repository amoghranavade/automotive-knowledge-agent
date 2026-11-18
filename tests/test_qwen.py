import json

import pytest

from clickless.llm import QwenClient, QwenResponseError


def test_structured_request_disables_reasoning(monkeypatch):
    captured = {}

    def fake_request(_client, method, path, payload):
        captured.update(payload)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps({"ok": True})},
                }
            ]
        }

    client = QwenClient(api_key="test")
    monkeypatch.setattr(QwenClient, "_request", fake_request)

    assert client.complete_json("Return JSON.", "Confirm.") == {"ok": True}
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}


def test_truncated_response_has_clear_error(monkeypatch):
    client = QwenClient(api_key="test")
    monkeypatch.setattr(
        QwenClient,
        "_request",
        lambda *_args, **_kwargs: {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": None},
                }
            ]
        },
    )

    with pytest.raises(QwenResponseError, match="output limit"):
        client.complete_json("Return JSON.", "Confirm.")
