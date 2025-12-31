import unittest
from unittest.mock import patch

from web_agent_parser.parsers import translate_non_english_html


class _FakeTranslator:
    def __init__(self, replacement: str) -> None:
        self._replacement = replacement

    def translate(self, text: str, source_lang: str) -> str:
        return self._replacement


class TranslateParserTests(unittest.TestCase):
    def test_translates_non_english_text(self) -> None:
        print("Translating non-English text should replace content and emit a hit.")
        html = "<p>Hola mundo.</p>"
        translator = _FakeTranslator("Hello world.")

        def _fake_detect(text: str):
            return ("es", 0.99) if "Hola" in text else ("en", 0.99)

        with patch("web_agent_parser.parsers._detect_language", side_effect=_fake_detect):
            translated, hits = translate_non_english_html(html, translator=translator, min_chars=1, min_alpha=1)

        self.assertEqual(translated, "<p>Hello world.</p>")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].language, "es")

    def test_english_text_is_untouched(self) -> None:
        print("English text should remain unchanged and emit no hits.")
        html = "<p>Hello world.</p>"
        translator = _FakeTranslator("<b>Hi</b>")

        with patch("web_agent_parser.parsers._detect_language", return_value=("en", 0.99)):
            translated, hits = translate_non_english_html(html, translator=translator)

        self.assertEqual(translated, html)
        self.assertEqual(hits, [])

    def test_translated_text_is_escaped(self) -> None:
        print("Translated text is HTML-escaped to avoid injection.")
        html = "<p>Hola mundo.</p>"
        translator = _FakeTranslator("<b>Hi</b>")

        with patch("web_agent_parser.parsers._detect_language", return_value=("es", 0.99)):
            translated, hits = translate_non_english_html(html, translator=translator, min_chars=1, min_alpha=1)

        self.assertEqual(translated, "<p>&lt;b&gt;Hi&lt;/b&gt;</p>")
        self.assertEqual(len(hits), 1)


if __name__ == "__main__":
    unittest.main()
