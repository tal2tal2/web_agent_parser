"""
Composable parser pipeline that runs HTML parsers sequentially.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from .parsers import remove_obfuscated_text_html, translate_non_english_html

try:
    import tomllib
except Exception:  # pragma: no cover - optional TOML support
    tomllib = None
    try:
        import tomli as tomllib  # type: ignore[assignment]
    except Exception:  # pragma: no cover - optional TOML support
        tomllib = None


class ParserStep(Protocol):
    name: str

    def run(self, html: str) -> Tuple[str, Sequence[Any]]: ...


@dataclass(frozen=True)
class FunctionParserStep:
    name: str
    parser: Callable[..., Tuple[str, Sequence[Any]]]
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def run(self, html: str) -> Tuple[str, Sequence[Any]]:
        return self.parser(html, **self.kwargs)


@dataclass(frozen=True)
class ParserResult:
    name: str
    html: str
    hits: Sequence[Any]


@dataclass(frozen=True)
class PipelineResult:
    html: str
    results: List[ParserResult]


class ParserPipeline:
    def __init__(self, steps: Optional[Iterable[ParserStep]] = None) -> None:
        self._steps = list(steps or [])

    @property
    def steps(self) -> List[ParserStep]:
        return list(self._steps)

    def add_step(self, step: ParserStep) -> None:
        self._steps.append(step)

    def run(self, html: str) -> PipelineResult:
        current = html
        results: List[ParserResult] = []
        for step in self._steps:
            current, hits = step.run(current)
            results.append(ParserResult(name=step.name, html=current, hits=list(hits)))
        return PipelineResult(html=current, results=results)


class PipelineConfigError(ValueError):
    pass


DEFAULT_PARSER_REGISTRY: Mapping[str, Callable[..., Tuple[str, Sequence[Any]]]] = {
    "remove_obfuscated_text_html": remove_obfuscated_text_html,
    "translate_non_english_html": translate_non_english_html,
}

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "pipeline.default.json"
)


def load_pipeline_config(path: Union[Path, str]) -> Mapping[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise PipelineConfigError(f"Config file not found: {config_path}")
    suffix = config_path.suffix.lower()
    raw = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(raw)
    elif suffix == ".toml":
        if tomllib is None:
            raise PipelineConfigError(
                "TOML parsing requires Python 3.11+ or the tomli package."
            )
        data = tomllib.loads(raw)
    else:
        raise PipelineConfigError(
            f"Unsupported config extension '{suffix}'. Use .json or .toml."
        )
    if not isinstance(data, Mapping):
        raise PipelineConfigError("Pipeline config must be a JSON/TOML object.")
    return data


def build_pipeline(
    *,
    config_path: Optional[Union[Path, str]] = None,
    config: Optional[Mapping[str, Any]] = None,
    registry: Optional[Mapping[str, Callable[..., Tuple[str, Sequence[Any]]]]] = None,
) -> ParserPipeline:
    if config is None:
        config_path = config_path or DEFAULT_CONFIG_PATH
        config = load_pipeline_config(config_path)
    return _build_pipeline_from_config(config, registry=registry)


def build_default_pipeline(
    *,
    config_path: Optional[Union[Path, str]] = None,
    registry: Optional[Mapping[str, Callable[..., Tuple[str, Sequence[Any]]]]] = None,
) -> ParserPipeline:
    return build_pipeline(config_path=config_path, registry=registry)


def _build_pipeline_from_config(
    config: Mapping[str, Any],
    *,
    registry: Optional[Mapping[str, Callable[..., Tuple[str, Sequence[Any]]]]] = None,
) -> ParserPipeline:
    parser_registry = dict(DEFAULT_PARSER_REGISTRY)
    if registry:
        parser_registry.update(registry)

    steps_config = config.get("steps")
    if steps_config is None:
        raise PipelineConfigError("Pipeline config missing 'steps' list.")
    if not isinstance(steps_config, list):
        raise PipelineConfigError("'steps' must be a list of step objects.")

    steps: List[ParserStep] = []
    for index, raw_step in enumerate(steps_config):
        if not isinstance(raw_step, Mapping):
            raise PipelineConfigError(
                f"Step {index} must be an object with a 'parser' field."
            )
        if raw_step.get("enabled", True) is False:
            continue
        parser_name = raw_step.get("parser")
        if not parser_name or not isinstance(parser_name, str):
            raise PipelineConfigError(f"Step {index} is missing a valid 'parser' name.")
        parser = parser_registry.get(parser_name)
        if parser is None:
            raise PipelineConfigError(f"Unknown parser '{parser_name}'.")
        name = raw_step.get("name") or parser_name
        kwargs = raw_step.get("kwargs") or {}
        if not isinstance(kwargs, Mapping):
            raise PipelineConfigError(
                f"Step {index} 'kwargs' must be an object if provided."
            )
        steps.append(FunctionParserStep(name=name, parser=parser, kwargs=dict(kwargs)))
    return ParserPipeline(steps)
