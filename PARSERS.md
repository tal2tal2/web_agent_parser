# Parsers Implemented

- `translate_non_english_html(...)` — HTML parser/rewriter that detects non-English text and substitutes English translations, returning hits. `src/web_agent_parser/parsers.py`
- `TransformersTranslator` — offline translation via a Hugging Face model. `src/web_agent_parser/parsers.py`
- `GoogleTranslateTranslator` — Google Translate API-backed translator (reads `GOOGLE_API_KEY`). `src/web_agent_parser/parsers.py`
- `remove_obfuscated_text_html(...)` — strips base64/hex-like obfuscated text that decodes to readable text. `src/web_agent_parser/parsers.py`
