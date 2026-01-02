# Pipeline Configuration

This project loads parser pipelines from a JSON or TOML configuration file. Each
step references a parser by name and can pass keyword arguments to override the
parser defaults.

Default config:

- `src/web_agent_parser/configs/pipeline.default.json`

Sample config:

- `src/web_agent_parser/configs/pipeline.sample.json`

## Config schema

Top-level fields:

- `steps` (required): ordered list of pipeline steps.

Each step supports:

- `parser` (required): parser registry name.
- `name` (optional): human-readable step label (defaults to `parser`).
- `enabled` (optional): `true`/`false` toggle (defaults to `true`).
- `kwargs` (optional): keyword args passed into the parser; omitted keys use the
  parser's default values.

## Example JSON

```json
{
  "steps": [
    { "parser": "remove_obfuscated_text_html" },
    {
      "parser": "translate_non_english_html",
      "kwargs": {
        "min_chars": 30,
        "confidence_threshold": 0.85
      }
    }
  ]
}
```

## Example TOML

```toml
[[steps]]
parser = "remove_obfuscated_text_html"

[[steps]]
parser = "translate_non_english_html"
enabled = true
kwargs = { min_chars = 30, confidence_threshold = 0.85 }
```

## Using the config

```python
from web_agent_parser.pipeline import build_pipeline

pipeline = build_pipeline(config_path="path/to/pipeline.json")
result = pipeline.run(html)
```

## Adding custom parsers

Register a new parser when building the pipeline, and reference it by name in
your config.

```python
from web_agent_parser.pipeline import build_pipeline

pipeline = build_pipeline(
    config_path="path/to/pipeline.json",
    registry={"my_parser": my_parser},
)
```

## TOML support

TOML parsing uses `tomllib` on Python 3.11+ or falls back to the `tomli` package
when available. If neither is installed, use JSON configs.
