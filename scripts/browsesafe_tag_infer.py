"""
Rule-based BrowseSafe runner using ONLY pi_html_parser tags (no other pipeline).

- Loads perplexity-ai/browsesafe-bench (or a local cached copy) and iterates split rows.
- For each HTML doc:
  1) parsed = pi_html_parser(html)   # must add lines like: [PI_TAG severity=... ...]
  2) prediction = "yes" if any tag has severity in YES_SEVERITIES (default: {"medium","high"})
     else "no"

Writes JSONL records similar to browsesafe_infer.py:
{
  "i": ...,
  "text_sha256": ...,
  "label": ...,
  "parser": ...,
  "parsed_text_sha256": ...,
  "prediction": ...,
  "chunk_predictions": [...]
}

Usage:
  python browsesafe_tag_infer_pi_only.py --out browsesafe_pi_tags_only.jsonl

Optional:
  python browsesafe_tag_infer_pi_only.py --yes-severities high
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from collections import Counter

# Import your parser (expects: def pi_html_parser(html: str) -> str)
from html_parser import pi_html_parser


# ----------------------------
# Tag extraction / prediction
# ----------------------------
RE_PI_TAG_LINE = re.compile(r"(?m)^\[PI_TAG\b[^\]]*\]\s*$")
RE_SEVERITY = re.compile(r"\bseverity\s*=\s*(low|medium|high)\b", re.IGNORECASE)

DEFAULT_YES_SEVERITIES = {"medium", "high"}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _parser_name() -> str:
    return getattr(pi_html_parser, "__name__", "pi_html_parser")


def _extract_tag_severities(parsed_text: str) -> List[str]:
    """
    Returns severities found on [PI_TAG ...] lines. If a tag lacks severity, assumes "low".
    """
    severities: List[str] = []
    for m in RE_PI_TAG_LINE.finditer(parsed_text):
        line = m.group(0)
        sm = RE_SEVERITY.search(line)
        severities.append(sm.group(1).lower() if sm else "low")
    return severities


def _predict_from_tags(parsed_text: str, yes_severities: set) -> Tuple[str, List[str]]:
    """
    Document prediction is OR over tag severities.
    chunk_predictions here is a simple trace: one 'yes'/'no' per found tag line
    (or ['no'] if no tags).
    """
    severities = _extract_tag_severities(parsed_text)
    if not severities:
        return "no", ["no"]

    tag_preds = ["yes" if s in yes_severities else "no" for s in severities]
    pred = "yes" if any(p == "yes" for p in tag_preds) else "no"
    return pred, tag_preds


# ----------------------------
# Main runner
# ----------------------------
def run(
    *,
    dataset: str = "perplexity-ai/browsesafe-bench",
    split: str = "test",
    html_field: str = "content",
    label_field: str = "label",
    dataset_dir: Optional[str] = None,
    redownload: bool = False,
    limit: Optional[int] = None,
    streaming: bool = False,
    out: str = "browsesafe_pi_tags_only.jsonl",
    show_first: int = 1,
    show_chars: int = 300,
    yes_severities: Optional[List[str]] = None,
) -> None:
    if dataset_dir is None:
        safe_name = str(dataset).replace("/", "__")
        dataset_dir = str(Path("data") / "raw" / safe_name / str(split))

    yes_set = set(s.lower() for s in (yes_severities or list(DEFAULT_YES_SEVERITIES)))

    parser_name = _parser_name()
    print(f"Parser: {parser_name}")
    print(f"Rule: predict YES if any PI_TAG severity in {sorted(yes_set)}")

    # Dataset loading
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
            iterator = enumerate(tqdm(ordered_iter, desc="browsesafe_tag_only"))
        else:
            assert n_rows is not None
            iterator = ((i, ds[i]) for i in tqdm(range(n_rows), desc="browsesafe_tag_only"))

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

            parsed = pi_html_parser(html)
            if not isinstance(parsed, str):
                raise TypeError(f"Parser {parser_name!r} returned {type(parsed).__name__}, expected str")
            parsed_hash = _sha256(parsed)

            pred, tag_preds = _predict_from_tags(parsed, yes_set)

            if isinstance(label, str) and pred == label:
                correct += 1

            if i < show_first:
                sevs = _extract_tag_severities(parsed)
                snippet = html[:show_chars].replace("\n", " ").replace("\r", " ")
                print("\nTEXT:", snippet + ("..." if len(html) > show_chars else ""))
                print("LABEL:", repr(label))
                print(f"TAG_ONLY_OUTPUT[{parser_name}]:", repr(pred))
                print("TAG_SEVERITIES:", _extract_tag_severities(parsed)[:25])
                print("TAG_SEVERITIES_UNIQUE:", sorted(set(sevs)))
                print("TAG_SEVERITIES_COUNT:", Counter(sevs))

            rec = {
                "i": i,
                "text_sha256": raw_hash,
                "label": label,
                "parser": parser_name,
                "parsed_text_sha256": parsed_hash,
                "prediction": pred,
                "chunk_predictions": tag_preds,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. Wrote {written} predictions -> {out_path}")
    if total_with_label:
        acc = correct / total_with_label
        print(f"Accuracy[{parser_name}]: {correct}/{total_with_label} = {acc:.3f}")
        print(f"Label distribution: {label_counts}")


def main() -> None:
    ap = argparse.ArgumentParser(description="BrowseSafe-bench -> PI_TAG-only runner (pi_html_parser only)")
    ap.add_argument("--dataset", default="perplexity-ai/browsesafe-bench", help="HF dataset name")
    ap.add_argument("--split", default="test", help="HF dataset split")
    ap.add_argument("--html-field", default="content", help="Field containing HTML")
    ap.add_argument("--label-field", default="label", help="Field containing ground-truth label")
    ap.add_argument(
        "--dataset-dir",
        default=None,
        help="Local dataset directory (load from disk if present; otherwise download and save here).",
    )
    ap.add_argument("--redownload", action="store_true", help="Force re-download dataset even if --dataset-dir exists")
    ap.add_argument("--limit", type=int, default=None, help="Max samples to run")
    ap.add_argument("--streaming", action="store_true", default=False, help="Stream dataset from HF")
    ap.add_argument("--out", default="browsesafe_pi_tags_only.jsonl", help="Output JSONL path")
    ap.add_argument("--show-first", type=int, default=1, help="Print info for first N samples")
    ap.add_argument("--show-chars", type=int, default=300, help="Chars to print from TEXT")
    ap.add_argument(
        "--yes-severities",
        nargs="*",
        help='Severities that trigger "yes" (default: medium high)',
    )
    args = ap.parse_args()

    run(
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
        yes_severities=args.yes_severities,
    )


if __name__ == "__main__":
    main()
