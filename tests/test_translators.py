import os
import types
from unittest.mock import MagicMock, patch

import pytest

from web_agent_parser.parsers import GoogleTranslateTranslator, TransformersTranslator


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_transformers_translator_joins_chunks() -> None:
    fake_pipe = MagicMock()
    fake_pipe.side_effect = lambda chunk, max_length: [
        {"translation_text": f"EN:{chunk}"}
    ]
    fake_transformers = types.SimpleNamespace(
        pipeline=lambda *args, **kwargs: fake_pipe
    )

    with patch.dict("sys.modules", {"transformers": fake_transformers}):
        with patch("web_agent_parser.parsers._chunk_text", return_value=["uno", "dos"]):
            translator = TransformersTranslator(model_name="dummy")
            result = translator.translate("ignored", "es")

    assert result == "EN:uno EN:dos"
    assert fake_pipe.call_count == 2


def test_google_translate_translator_joins_chunks() -> None:
    translator = GoogleTranslateTranslator(api_key="test-key")
    responses = [
        _FakeResponse(b'{"data":{"translations":[{"translatedText":"Hello"}]}}'),
        _FakeResponse(b'{"data":{"translations":[{"translatedText":"World"}]}}'),
    ]

    with patch("web_agent_parser.parsers._chunk_text", return_value=["hola", "mundo"]):
        with patch("urllib.request.urlopen", side_effect=responses) as urlopen:
            result = translator.translate("ignored", "es")

    assert result == "Hello World"
    assert urlopen.call_count == 2


def test_google_translate_requires_api_key() -> None:
    with patch("web_agent_parser.parsers._load_dotenv_if_available"):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError):
                GoogleTranslateTranslator()
