"""
web_agent_parser

Parser-level guardrails for agentic web browsing: parse HTML into structured,
annotated content before any LLM reasoning.
"""

from .parsers import (
    GoogleTranslateTranslator,
    ObfuscationHit,
    TransformersTranslator,
    TranslationHit,
    remove_obfuscated_text_html,
    translate_non_english_html,
)
from .pipeline import (
    FunctionParserStep,
    ParserPipeline,
    ParserResult,
    PipelineConfigError,
    PipelineResult,
    build_default_pipeline,
    build_pipeline,
    load_pipeline_config,
)

__all__ = [
    "__version__",
    "GoogleTranslateTranslator",
    "ObfuscationHit",
    "PipelineResult",
    "PipelineConfigError",
    "FunctionParserStep",
    "ParserPipeline",
    "ParserResult",
    "TranslationHit",
    "TransformersTranslator",
    "build_pipeline",
    "build_default_pipeline",
    "load_pipeline_config",
    "remove_obfuscated_text_html",
    "translate_non_english_html",
]
__version__ = "0.1.0"
