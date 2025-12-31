"""
Simple confusion matrix for `scripts/browsesafe_infer.py` JSONL outputs.

Primary use: import + call a function.

Example:
  from scripts.confusion_matrix import confusion_matrix_from_jsonl, format_confusion_matrix
  labels, mat, meta = confusion_matrix_from_jsonl("browsesafe_predictions.jsonl")
  print(format_confusion_matrix(labels, mat))
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield dict records from a JSONL file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def confusion_matrix(
    rows: Iterable[dict],
    *,
    label_key: str = "label",
    pred_key: str = "prediction",
    labels: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[List[int]], Counter]:
    """
    Compute confusion matrix counts.

    Returns (labels, matrix, meta) where:
    - labels: ordered label names used for both axes
    - matrix: matrix[true_i][pred_i] = count
    - meta: Counter with a few useful stats (e.g. "skipped_no_label")
    """
    meta: Counter = Counter()
    pairs: List[Tuple[str, str]] = []
    seen: List[str] = []

    for r in rows:
        y = r.get(label_key, None)
        yhat = r.get(pred_key, None)
        if y is None:
            meta["skipped_no_label"] += 1
            continue
        if yhat is None:
            meta["skipped_no_prediction"] += 1
            continue
        y_s = str(y)
        yhat_s = str(yhat)
        pairs.append((y_s, yhat_s))
        seen.append(y_s)
        seen.append(yhat_s)

    if labels is None:
        labels = sorted(set(seen))  # stable ordering
    label_list = list(labels)
    idx: Dict[str, int] = {lab: i for i, lab in enumerate(label_list)}

    n = len(label_list)
    mat: List[List[int]] = [[0 for _ in range(n)] for _ in range(n)]
    for y, yhat in pairs:
        if y not in idx:
            meta["skipped_true_unknown_label"] += 1
            continue
        if yhat not in idx:
            meta["skipped_pred_unknown_label"] += 1
            continue
        mat[idx[y]][idx[yhat]] += 1
        meta["included"] += 1

    return label_list, mat, meta


def confusion_matrix_from_jsonl(
    predictions_jsonl: str | Path,
    *,
    label_key: str = "label",
    pred_key: str = "prediction",
    labels: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[List[int]], Counter]:
    return confusion_matrix(
        iter_jsonl(predictions_jsonl),
        label_key=label_key,
        pred_key=pred_key,
        labels=labels,
    )


def format_confusion_matrix(labels: Sequence[str], matrix: Sequence[Sequence[int]]) -> str:
    """Return a compact, readable text table. Rows=true labels, Cols=pred labels."""
    labels = list(labels)
    widths = [max(len("true\\pred"), *(len(l) for l in labels))]
    for j, lab in enumerate(labels):
        col_max = max(len(lab), *(len(str(matrix[i][j])) for i in range(len(labels))))
        widths.append(col_max)

    def fmt_row(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    header = fmt_row(["true\\pred", *labels])
    lines = [header]
    for i, y in enumerate(labels):
        lines.append(fmt_row([y, *[str(matrix[i][j]) for j in range(len(labels))]]))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute confusion matrix from browsesafe_infer.py JSONL output")
    p.add_argument("predictions_jsonl", help="Path to JSONL output from browsesafe_infer.py")
    p.add_argument("--label-key", default="label")
    p.add_argument("--pred-key", default="prediction")
    args = p.parse_args()

    labels, mat, meta = confusion_matrix_from_jsonl(args.predictions_jsonl, label_key=args.label_key, pred_key=args.pred_key)
    print(format_confusion_matrix(labels, mat))
    print("\nmeta:", dict(meta))


if __name__ == "__main__":
    main()


