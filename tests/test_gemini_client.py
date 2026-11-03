from types import SimpleNamespace

import pytest

from core import gemini_client


def test_generate_answer_uses_low_latency_model(monkeypatch):
    calls = {}

    def generate_content(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(text="answer")

    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    monkeypatch.setattr(gemini_client, "_get_client", lambda: client)

    assert gemini_client.generate_answer("question", ["context"]) == "answer"
    assert calls["model"] == "gemini-3.5-flash-lite"
    assert calls["config"].max_output_tokens == 1024
    assert calls["config"].thinking_config.thinking_level.value == "LOW"
    assert "complete sentences" in calls["contents"]


def test_extract_text_rejects_truncated_answer():
    response = SimpleNamespace(
        text="The answer is cut off",
        candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
    )

    with pytest.raises(ValueError, match="truncated the answer"):
        gemini_client._extract_text(response)
