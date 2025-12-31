"""
web_agent_parser

Parser-level guardrails for agentic web browsing: parse HTML into structured,
annotated content before any LLM reasoning.
"""

from .parsers import (
    GoogleTranslateTranslator,
    TranslationHit,
    TransformersTranslator,
    translate_non_english_html,
)

__all__ = [
    "__version__",
    "GoogleTranslateTranslator",
    "TranslationHit",
    "TransformersTranslator",
    "translate_non_english_html",
]
__version__ = "0.1.0"
