"""
Minimal BrowseSafe runner.

Goal: you can run this on a server with just:
  python browsesafe_infer.py

Defaults:
- model: perplexity-ai/browsesafe (31B)
- dataset: perplexity-ai/browsesafe-bench, split=test
- HTML field: content

Model reference: https://huggingface.co/perplexity-ai/browsesafe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _build_prompt(tokenizer, html: str) -> str:
    """Prefer chat template; fall back to raw HTML."""
    try:
        apply = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply):
            messages = [{"role": "user", "content": html}]
            return apply(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    return html


def _chunk_by_tokens(tokenizer, text: str, *, max_tokens: int) -> List[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [text]
    chunks: List[str] = []
    for start in range(0, len(ids), max_tokens):
        chunks.append(tokenizer.decode(ids[start : start + max_tokens]))
    return chunks


def _predict_one(tokenizer, model, html: str, *, max_new_tokens: int) -> str:
    prompt = _build_prompt(tokenizer, html)
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, pad_token_id=pad_token_id)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return content.split()[0].lower() if content else ""


def predict_document(
    tokenizer,
    model,
    html: str,
    *,
    max_input_tokens: int,
    max_new_tokens: int,
) -> Tuple[str, List[str]]:
    """
    Conservative OR aggregation: if any chunk is "yes", document is "yes".
    """
    chunks = _chunk_by_tokens(tokenizer, html, max_tokens=max_input_tokens)
    preds: List[str] = []
    for ch in chunks:
        preds.append(_predict_one(tokenizer, model, ch, max_new_tokens=max_new_tokens))
        if preds[-1] == "yes":
            return "yes", preds
    final = "no" if any(p == "no" for p in preds) else (preds[-1] if preds else "")
    return final, preds


def main() -> None:
    parser = argparse.ArgumentParser(description="BrowseSafe dataset -> inference runner")
    parser.add_argument("--model", default="perplexity-ai/browsesafe", help="HF model id")
    parser.add_argument("--dataset", default="perplexity-ai/browsesafe-bench", help="HF dataset name")
    parser.add_argument("--split", default="test", help="HF dataset split")
    parser.add_argument("--html-field", default="content", help="Dataset field containing HTML (BrowseSafe-Bench uses `content`)")
    parser.add_argument("--label-field", default="label", help="Dataset field containing ground-truth label")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to run (default: all)")
    parser.add_argument("--streaming", action="store_true", default=True, help="Stream dataset (avoids full download)")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", help="Disable streaming")
    parser.add_argument("--out", default="browsesafe_predictions.jsonl", help="Output JSONL path")
    parser.add_argument("--show-first", type=int, default=1, help="Print TEXT/LABEL/OUTPUT for first N samples")
    parser.add_argument("--show-chars", type=int, default=300, help="Chars to print from TEXT")
    parser.add_argument("--max-input-tokens", type=int, default=12000, help="Chunk size in tokenizer tokens")
    parser.add_argument("--max-new-tokens", type=int, default=4, help='Generation budget (BrowseSafe outputs one token: "yes"/"no")')
    args = parser.parse_args()


    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Minimal server-friendly loading: shard across available GPUs if possible.
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").to("cpu")
    model.eval()

    print(f"Loading dataset: {args.dataset} split={args.split} (streaming={args.streaming})")
    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        it: Iterable[Dict[str, Any]] = ds
        for i, row in enumerate(tqdm(it, desc="browsesafe")):
            if args.limit is not None and i >= args.limit:
                break

            if args.html_field not in row:
                raise KeyError(f"Missing html field {args.html_field!r}. Available keys: {list(row.keys())}")

            html = row[args.html_field]
            if not isinstance(html, str):
                continue

            label = row.get(args.label_field, None)
            pred, chunk_preds = predict_document(
                tokenizer,
                model,
                html,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
            )

            if i < args.show_first:
                snippet = html[: args.show_chars].replace("\n", " ").replace("\r", " ")
                print("\nTEXT:", snippet + ("..." if len(html) > args.show_chars else ""))
                print("LABEL:", repr(label))
                print("MODEL_OUTPUT:", repr(pred))

            rec = {
                "i": i,
                "label": label,
                "prediction": pred,
                "chunk_predictions": chunk_preds,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. Wrote {written} predictions -> {out_path}")


if __name__ == "__main__":
    main()
