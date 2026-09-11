from unittest.mock import MagicMock

import pytest
from ollama._types import ChatResponse, ListResponse, Message

from rag.config import Settings
from rag.models.llm import OllamaClient


def test_generate_returns_the_message_content() -> None:
    client = MagicMock()
    client.chat.return_value = {"message": {"content": "hello world"}}
    llm = OllamaClient(Settings(), client=client)
    assert llm.generate("hi") == "hello world"


def test_generate_sends_the_configured_model_and_zero_temperature() -> None:
    client = MagicMock()
    client.chat.return_value = {"message": {"content": "ok"}}
    settings = Settings()
    llm = OllamaClient(settings, client=client)
    llm.generate("hi")

    kwargs = client.chat.call_args.kwargs
    assert kwargs["model"] == settings.models.generator
    assert kwargs["options"]["temperature"] == 0.0


def test_generate_prepends_the_system_message_when_given() -> None:
    client = MagicMock()
    client.chat.return_value = {"message": {"content": "ok"}}
    llm = OllamaClient(Settings(), client=client)
    llm.generate("hi", system="be terse")

    messages = client.chat.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "be terse"}
    assert messages[1] == {"role": "user", "content": "hi"}


def test_is_ready_is_false_when_the_model_is_not_pulled() -> None:
    client = MagicMock()
    client.list.return_value = {"models": [{"model": "some-other-model"}]}
    llm = OllamaClient(Settings(), client=client)
    assert llm.is_ready() is False


def test_is_ready_is_true_when_the_configured_model_is_present() -> None:
    settings = Settings()
    client = MagicMock()
    client.list.return_value = {"models": [{"model": settings.models.generator}]}
    llm = OllamaClient(settings, client=client)
    assert llm.is_ready() is True


def test_is_ready_is_false_when_ollama_is_unreachable() -> None:
    client = MagicMock()
    client.list.side_effect = ConnectionError("refused")
    llm = OllamaClient(Settings(), client=client)
    assert llm.is_ready() is False


def test_handles_real_ollama_response_types_not_just_dicts() -> None:
    """Dict mocks would hide an API mismatch: the real client returns pydantic models.

    Constructed locally, so this stays a fast unit test with no network.
    """
    settings = Settings()
    client = MagicMock()
    client.list.return_value = ListResponse(
        models=[ListResponse.Model(model=settings.models.generator)]
    )
    client.chat.return_value = ChatResponse(
        model=settings.models.generator,
        message=Message(role="assistant", content="grounded answer [1]"),
    )
    llm = OllamaClient(settings, client=client)

    assert llm.is_ready() is True
    assert llm.generate("hi") == "grounded answer [1]"


def test_client_is_constructed_with_the_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a timeout a hung CPU generation blocks the request forever."""
    captured: dict[str, object] = {}

    def fake_client(host: str | None = None, **kwargs: object) -> MagicMock:
        captured.update({"host": host, **kwargs})
        return MagicMock()

    monkeypatch.setattr("rag.models.llm.ollama.Client", fake_client)
    settings = Settings()
    OllamaClient(settings)

    assert captured["host"] == settings.ollama.host
    assert captured["timeout"] == settings.ollama.timeout_s
