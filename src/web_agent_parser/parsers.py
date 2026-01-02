from __future__ import annotations

import base64
import binascii
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html import escape, unescape
from html.parser import HTMLParser
from typing import Iterable, List, Optional, Protocol, Tuple

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
    def translate(self, text: str, source_lang: str) -> str: ...


@dataclass(frozen=True)
class TranslationHit:
    index: int
    tag_path: str
    language: str
    confidence: float
    original: str
    translated: str


@dataclass(frozen=True)
class ObfuscationHit:
    index: int
    tag_path: str
    encoding: str
    original: str
    decoded: str


class TransformersTranslator:
    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-mul-en",
        *,
        device: Optional[int] = None,
    ) -> None:
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "transformers is required for TransformersTranslator"
            ) from exc
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
            req = urllib.request.Request(
                f"{self._endpoint}?key={self._api_key}", data=data, method="POST"
            )
            req.add_header(
                "Content-Type", "application/x-www-form-urlencoded; charset=utf-8"
            )
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


def _looks_like_text(
    text: str, *, min_chars: int, min_alpha: int, min_alpha_ratio: float
) -> bool:
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


_BASE64_RE = re.compile(r"[A-Za-z0-9+/=]{8,}")
_HEX_RE = re.compile(r"(?:0x)?[0-9a-fA-F]{8,}")


def _is_readable_text(
    text: str,
    *,
    min_chars: int,
    min_alpha: int,
    min_alpha_ratio: float,
    min_printable_ratio: float,
) -> bool:
    if not _looks_like_text(
        text, min_chars=min_chars, min_alpha=min_alpha, min_alpha_ratio=min_alpha_ratio
    ):
        return False
    printable = sum(1 for ch in text if ch.isprintable())
    return (printable / max(len(text), 1)) >= min_printable_ratio


def _decode_base64_text(value: str) -> Optional[str]:
    cleaned = value.strip()
    pad = (-len(cleaned)) % 4
    if pad:
        cleaned += "=" * pad
    try:
        decoded = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError):
        return None
    return _decode_bytes_to_text(decoded)


def _decode_hex_text(value: str) -> Optional[str]:
    cleaned = value.strip()
    if cleaned.startswith(("0x", "0X")):
        cleaned = cleaned[2:]
    if len(cleaned) % 2 != 0:
        return None
    try:
        decoded = bytes.fromhex(cleaned)
    except ValueError:
        return None
    return _decode_bytes_to_text(decoded)


def _decode_bytes_to_text(decoded: bytes) -> Optional[str]:
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _candidate_matches(
    text: str, *, min_encoded_len: int
) -> List[Tuple[int, int, str]]:
    matches: List[Tuple[int, int, str]] = []
    for match in _BASE64_RE.finditer(text):
        value = match.group(0)
        if len(value) < min_encoded_len:
            continue
        if not any(ch in value for ch in "+/=") and all(
            ch in "0123456789abcdefABCDEF" for ch in value
        ):
            continue
        matches.append((match.start(), match.end(), "base64"))
    for match in _HEX_RE.finditer(text):
        value = match.group(0)
        if len(value) < min_encoded_len:
            continue
        matches.append((match.start(), match.end(), "hex"))
    matches.sort(key=lambda item: item[0])
    return matches


def _redact_obfuscated_segments(
    text: str,
    *,
    min_encoded_len: int,
    min_decoded_chars: int,
    min_decoded_alpha: int,
    min_decoded_alpha_ratio: float,
    min_printable_ratio: float,
) -> Tuple[str, List[Tuple[str, str, str, int, int]]]:
    matches = _candidate_matches(text, min_encoded_len=min_encoded_len)
    if not matches:
        return text, []

    redactions: List[Tuple[int, int, str, str]] = []
    for start, end, encoding in matches:
        segment = text[start:end]
        decoded = (
            _decode_base64_text(segment)
            if encoding == "base64"
            else _decode_hex_text(segment)
        )
        if not decoded:
            continue
        if not _is_readable_text(
            decoded,
            min_chars=min_decoded_chars,
            min_alpha=min_decoded_alpha,
            min_alpha_ratio=min_decoded_alpha_ratio,
            min_printable_ratio=min_printable_ratio,
        ):
            continue
        redactions.append((start, end, encoding, decoded))

    if not redactions:
        return text, []

    cleaned_parts: List[str] = []
    hits: List[Tuple[str, str, str, int, int]] = []
    last_end = 0
    for start, end, encoding, decoded in redactions:
        if start < last_end:
            continue
        cleaned_parts.append(text[last_end:start])
        last_end = end
        hits.append((encoding, text[start:end], decoded, start, end))
    cleaned_parts.append(text[last_end:])
    return "".join(cleaned_parts), hits


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

    def handle_startendtag(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
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


class _ObfuscationHTMLParser(HTMLParser):
    def __init__(
        self,
        *,
        min_encoded_len: int,
        min_decoded_chars: int,
        min_decoded_alpha: int,
        min_decoded_alpha_ratio: float,
        min_printable_ratio: float,
    ) -> None:
        super().__init__(convert_charrefs=False)
        self._min_encoded_len = min_encoded_len
        self._min_decoded_chars = min_decoded_chars
        self._min_decoded_alpha = min_decoded_alpha
        self._min_decoded_alpha_ratio = min_decoded_alpha_ratio
        self._min_printable_ratio = min_printable_ratio
        self._out: List[str] = []
        self._stack: List[str] = []
        self._hits: List[ObfuscationHit] = []
        self._index = 0

    @property
    def hits(self) -> List[ObfuscationHit]:
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

    def handle_startendtag(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        self._out.append("<" + tag + _format_attrs(attrs) + " />")

    def handle_data(self, data: str) -> None:
        cleaned, hits = _redact_obfuscated_segments(
            data,
            min_encoded_len=self._min_encoded_len,
            min_decoded_chars=self._min_decoded_chars,
            min_decoded_alpha=self._min_decoded_alpha,
            min_decoded_alpha_ratio=self._min_decoded_alpha_ratio,
            min_printable_ratio=self._min_printable_ratio,
        )
        if hits:
            tag_path = "/".join(self._stack)
            for encoding, original, decoded, _, _ in hits:
                self._hits.append(
                    ObfuscationHit(
                        index=self._index,
                        tag_path=tag_path,
                        encoding=encoding,
                        original=original,
                        decoded=decoded,
                    )
                )
        self._out.append(cleaned)
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


def remove_obfuscated_text_html(
    html: str,
    *,
    min_encoded_len: int = 32,
    min_decoded_chars: int = 16,
    min_decoded_alpha: int = 6,
    min_decoded_alpha_ratio: float = 0.25,
    min_printable_ratio: float = 0.85,
) -> Tuple[str, List[ObfuscationHit]]:
    """
    Remove suspected obfuscated text (base64/hex) if it decodes to readable text.

    Strategy:
    - detect long base64/hex-like spans in text nodes
    - decode candidates and check if the decoded bytes look like text
    - remove confirmed obfuscated segments from the HTML
    """
    parser = _ObfuscationHTMLParser(
        min_encoded_len=min_encoded_len,
        min_decoded_chars=min_decoded_chars,
        min_decoded_alpha=min_decoded_alpha,
        min_decoded_alpha_ratio=min_decoded_alpha_ratio,
        min_printable_ratio=min_printable_ratio,
    )
    parser.feed(html)
    parser.close()
    return parser.get_html(), parser.hits
