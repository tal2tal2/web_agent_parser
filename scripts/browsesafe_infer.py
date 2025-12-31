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
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset, load_from_disk
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ParserFn = Callable[[str], str]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def identity(x: str) -> str:
    return x


def _parser_name(fn: ParserFn) -> str:
    return getattr(fn, "__name__", "parser")


# Optional hook:
# - Leave as None to use identity (default).
# - Or set to your function: def my_parser(text: str) -> str: ...
PARSER_FN: Optional[ParserFn] = None


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


def run(
    *,
    model_id: str = "perplexity-ai/browsesafe",
    dataset: str = "perplexity-ai/browsesafe-bench",
    split: str = "test",
    html_field: str = "content",
    label_field: str = "label",
    dataset_dir: Optional[str] = None,
    redownload: bool = False,
    limit: Optional[int] = None,
    streaming: bool = False,
    out: str = "browsesafe_predictions.jsonl",
    show_first: int = 1,
    show_chars: int = 300,
    max_input_tokens: int = 12000,
    max_new_tokens: int = 4,
    parser_fn: Optional[ParserFn] = None,
) -> None:
    if dataset_dir is None:
        safe_name = str(dataset).replace("/", "__")
        dataset_dir = str(Path("data") / "raw" / safe_name / str(split))

    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Minimal server-friendly loading: shard across available GPUs if possible.
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto").to("cpu")
    model.eval()

    # Parser (user-supplied hook)
    parser_fn = parser_fn or PARSER_FN or identity
    parser_name = _parser_name(parser_fn)
    print(f"Parser: {parser_name}")

    # Dataset
    ds_dir = Path(dataset_dir)
    if streaming:
        print(f"Loading dataset (streaming): {dataset} split={split}")
        ds = load_dataset(dataset, split=split, streaming=True)
        ordered_iter: Optional[Iterable[Dict[str, Any]]] = ds
        n_rows: Optional[int] = None
    else:
        if ds_dir.exists() and not redownload:
            print(f"Loading dataset from disk: {ds_dir}")
            ds = load_from_disk(str(ds_dir))
        else:
            print(f"Downloading dataset: {dataset} split={split}")
            ds = load_dataset(dataset, split=split, streaming=False)
            ds_dir.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving dataset to disk: {ds_dir}")
            ds.save_to_disk(str(ds_dir))
        # Ordered-by-index iteration for reproducibility.
        n_rows = len(ds)
        ordered_iter = None

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    label_counts: Dict[str, int] = {}
    correct = 0
    total_with_label = 0
    with out_path.open("w", encoding="utf-8") as f:
        if ordered_iter is not None:
            iterator = enumerate(tqdm(ordered_iter, desc="browsesafe"))
        else:
            assert n_rows is not None
            iterator = ((i, ds[i]) for i in tqdm(range(n_rows), desc="browsesafe"))

        for i, row in iterator:
            if limit is not None and i >= limit:
                break

            if html_field not in row:
                raise KeyError(f"Missing html field {html_field!r}. Available keys: {list(row.keys())}")

            html = row[html_field]
            if not isinstance(html, str):
                continue

            label = row.get(label_field, None)
            if isinstance(label, str):
                label_counts[label] = label_counts.get(label, 0) + 1
                total_with_label += 1

            raw_hash = _sha256(html)

            parsed = parser_fn(html)
            if not isinstance(parsed, str):
                raise TypeError(f"Parser {parser_name!r} returned {type(parsed).__name__}, expected str")
            parsed_hash = _sha256(parsed)

            pred, chunk_preds = predict_document(
                tokenizer,
                model,
                parsed,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
            )

            if isinstance(label, str) and pred == label:
                correct += 1

            if i < show_first:
                snippet = html[:show_chars].replace("\n", " ").replace("\r", " ")
                print("\nTEXT:", snippet + ("..." if len(html) > show_chars else ""))
                print("LABEL:", repr(label))
                print(f"MODEL_OUTPUT[{parser_name}]:", repr(pred))

            rec = {
                "i": i,
                "text_sha256": raw_hash,
                "label": label,
                "parser": parser_name,
                "parsed_text_sha256": parsed_hash,
                "prediction": pred,
                "chunk_predictions": chunk_preds,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. Wrote {written} predictions -> {out_path}")
    if total_with_label:
        acc = correct / total_with_label
        print(f"Accuracy[{parser_name}]: {correct}/{total_with_label} = {acc:.3f}")
        print(f"Label distribution: {label_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BrowseSafe dataset -> inference runner")
    parser.add_argument("--model", default="perplexity-ai/browsesafe", help="HF model id")
    parser.add_argument("--dataset", default="perplexity-ai/browsesafe-bench", help="HF dataset name")
    parser.add_argument("--split", default="test", help="HF dataset split")
    parser.add_argument(
        "--html-field",
        default="content",
        help="Dataset field containing HTML (BrowseSafe-Bench uses `content`)",
    )
    parser.add_argument("--label-field", default="label", help="Dataset field containing ground-truth label")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Local dataset directory (load from disk if present; otherwise download and save here). Default: data/raw/<dataset>/<split>",
    )
    parser.add_argument("--redownload", action="store_true", help="Force re-download dataset even if --dataset-dir exists")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to run (default: all)")
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Stream dataset from HF (disables ordered-by-index iteration; not recommended for comparisons).",
    )
    parser.add_argument("--out", default="browsesafe_predictions.jsonl", help="Output JSONL path")
    parser.add_argument("--show-first", type=int, default=1, help="Print TEXT/LABEL/OUTPUT for first N samples")
    parser.add_argument("--show-chars", type=int, default=300, help="Chars to print from TEXT")
    parser.add_argument("--max-input-tokens", type=int, default=12000, help="Chunk size in tokenizer tokens")
    parser.add_argument("--max-new-tokens", type=int, default=4, help='Generation budget (BrowseSafe outputs one token: "yes"/"no")')
    args = parser.parse_args()

    run(
        model_id=args.model,
        dataset=args.dataset,
        split=args.split,
        html_field=args.html_field,
        label_field=args.label_field,
        dataset_dir=args.dataset_dir,
        redownload=args.redownload,
        limit=args.limit,
        streaming=args.streaming,
        out=args.out,
        show_first=args.show_first,
        show_chars=args.show_chars,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        parser_fn=None,  # use PARSER_FN or identity
    )


if __name__ == "__main__":
    main()
