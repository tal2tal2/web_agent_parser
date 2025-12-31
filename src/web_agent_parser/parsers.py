from __future__ import annotations

from dataclasses import dataclass
from html import escape, unescape
from html.parser import HTMLParser
from typing import Iterable, List, Optional, Protocol, Tuple

import json
import os
import re
import urllib.parse
import urllib.request

try:
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0
    _LANGDETECT_AVAILABLE = True
except Exception:
    _LANGDETECT_AVAILABLE = False

try:
    from dotenv import load_dotenv

    _DOTENV_AVAILABLE = True
except Exception:
    _DOTENV_AVAILABLE = False


class Translator(Protocol):
    def translate(self, text: str, source_lang: str) -> str:
        ...


@dataclass(frozen=True)
class TranslationHit:
    index: int
    tag_path: str
    language: str
    confidence: float
    original: str
    translated: str


class TransformersTranslator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-mul-en", *, device: Optional[int] = None) -> None:
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("transformers is required for TransformersTranslator") from exc
        self._pipe = pipeline("translation", model=model_name, device=device)

    def translate(self, text: str, source_lang: str) -> str:
        chunks = _chunk_text(text)
        translated: List[str] = []
        for chunk in chunks:
            out = self._pipe(chunk, max_length=512)
            if not out:
                translated.append(chunk)
            else:
                translated.append(out[0].get("translation_text", chunk))
        return " ".join(translated)


class GoogleTranslateTranslator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        endpoint: str = "https://translation.googleapis.com/language/translate/v2",
        target_lang: str = "en",
        timeout: int = 10,
        max_chars: int = 4500,
    ) -> None:
        if api_key is None:
            _load_dotenv_if_available()
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY for GoogleTranslateTranslator")
        self._api_key = api_key
        self._endpoint = endpoint
        self._target_lang = target_lang
        self._timeout = timeout
        self._max_chars = max_chars

    def translate(self, text: str, source_lang: str) -> str:
        chunks = _chunk_text(text, max_chars=self._max_chars)
        translated: List[str] = []
        for chunk in chunks:
            payload = {
                "q": chunk,
                "target": self._target_lang,
                "format": "text",
            }
            if source_lang:
                payload["source"] = source_lang
            data = urllib.parse.urlencode(payload).encode("utf-8")
            req = urllib.request.Request(f"{self._endpoint}?key={self._api_key}", data=data, method="POST")
            req.add_header("Content-Type", "application/x-www-form-urlencoded; charset=utf-8")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            translations = parsed.get("data", {}).get("translations", [])
            if not translations:
                translated.append(chunk)
            else:
                translated.append(translations[0].get("translatedText", chunk))
        return " ".join(translated)


def _chunk_text(text: str, *, max_chars: int = 400) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    buffer: List[str] = []
    size = 0
    for sentence in sentences:
        if not sentence:
            continue
        if size + len(sentence) + 1 > max_chars and buffer:
            chunks.append(" ".join(buffer))
            buffer = [sentence]
            size = len(sentence)
        else:
            buffer.append(sentence)
            size += len(sentence) + 1
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks


def _load_dotenv_if_available() -> None:
    if _DOTENV_AVAILABLE:
        load_dotenv(override=False)


def _looks_like_text(text: str, *, min_chars: int, min_alpha: int, min_alpha_ratio: float) -> bool:
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    alpha = sum(1 for ch in stripped if ch.isalpha())
    if alpha < min_alpha:
        return False
    return (alpha / max(len(stripped), 1)) >= min_alpha_ratio


def _detect_language(text: str) -> Tuple[str, float]:
    if not _LANGDETECT_AVAILABLE:
        return "en", 0.0
    try:
        langs = detect_langs(text)
    except Exception:
        return "en", 0.0
    if not langs:
        return "en", 0.0
    top = langs[0]
    return top.lang, float(top.prob)


def _safe_text(text: str) -> str:
    return escape(unescape(text), quote=False)


class _TranslatingHTMLParser(HTMLParser):
    def __init__(
        self,
        translator: Translator,
        *,
        min_chars: int,
        min_alpha: int,
        min_alpha_ratio: float,
        confidence_threshold: float,
    ) -> None:
        super().__init__(convert_charrefs=False)
        self._translator = translator
        self._min_chars = min_chars
        self._min_alpha = min_alpha
        self._min_alpha_ratio = min_alpha_ratio
        self._confidence_threshold = confidence_threshold
        self._out: List[str] = []
        self._stack: List[str] = []
        self._hits: List[TranslationHit] = []
        self._index = 0

    @property
    def hits(self) -> List[TranslationHit]:
        return self._hits

    def get_html(self) -> str:
        return "".join(self._out)

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self._stack.append(tag)
        self._out.append("<" + tag + _format_attrs(attrs) + ">")

    def handle_endtag(self, tag: str) -> None:
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()
        self._out.append(f"</{tag}>")

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self._out.append("<" + tag + _format_attrs(attrs) + " />")

    def handle_data(self, data: str) -> None:
        if not _looks_like_text(
            data,
            min_chars=self._min_chars,
            min_alpha=self._min_alpha,
            min_alpha_ratio=self._min_alpha_ratio,
        ):
            self._out.append(data)
            return
        lang, conf = _detect_language(data)
        if lang != "en" and conf >= self._confidence_threshold:
            translated = self._translator.translate(data, lang)
            safe_translated = _safe_text(translated)
            self._hits.append(
                TranslationHit(
                    index=self._index,
                    tag_path="/".join(self._stack),
                    language=lang,
                    confidence=conf,
                    original=data,
                    translated=translated,
                )
            )
            self._out.append(safe_translated)
        else:
            self._out.append(data)
        self._index += 1

    def handle_entityref(self, name: str) -> None:
        self._out.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._out.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self._out.append(f"<!--{data}-->")

    def handle_decl(self, decl: str) -> None:
        self._out.append(f"<!{decl}>")

    def handle_pi(self, data: str) -> None:
        self._out.append(f"<?{data}>")


def _format_attrs(attrs: Iterable[Tuple[str, Optional[str]]]) -> str:
    parts: List[str] = []
    for key, value in attrs:
        if value is None:
            parts.append(f" {key}")
        else:
            parts.append(f' {key}="{escape(value, quote=True)}"')
    return "".join(parts)


def translate_non_english_html(
    html: str,
    *,
    translator: Optional[Translator] = None,
    translation_model: str = "Helsinki-NLP/opus-mt-mul-en",
    min_chars: int = 20,
    min_alpha: int = 5,
    min_alpha_ratio: float = 0.35,
    confidence_threshold: float = 0.70,
) -> Tuple[str, List[TranslationHit]]:
    """
    Translate non-English natural language segments inside HTML into English.

    Returns translated HTML plus a list of translation hits for auditing.
    """
    if translator is None:
        translator = TransformersTranslator(model_name=translation_model)
    parser = _TranslatingHTMLParser(
        translator,
        min_chars=min_chars,
        min_alpha=min_alpha,
        min_alpha_ratio=min_alpha_ratio,
        confidence_threshold=confidence_threshold,
    )
    parser.feed(html)
    parser.close()
    return parser.get_html(), parser.hits
