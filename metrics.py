from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence

from evaluate import compute_score


def ctc_decode_ids(ids: Iterable[int], id_to_token: Sequence[str], blank_id: int) -> List[str]:
    output = []
    previous = None
    for idx in ids:
        idx = int(idx)
        if idx == blank_id:
            previous = idx
            continue
        if idx == previous:
            continue
        output.append(id_to_token[idx])
        previous = idx
    return output


def decode_logits(logits, id_to_token: Sequence[str], blank_id: int) -> List[str]:
    import numpy as np

    pred_ids = np.argmax(logits, axis=-1)
    return [" ".join(ctc_decode_ids(ids, id_to_token, blank_id)) for ids in pred_ids]


def write_ground_truth_csv(rows: Sequence[dict], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["canonical", "transcript"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "canonical": row["canonical"],
                    "transcript": row["transcript"],
                }
            )


def write_results_csv(predictions: Sequence[str], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["predict"])
        writer.writeheader()
        for prediction in predictions:
            writer.writerow({"predict": prediction})


def compute_mdd_metrics(
    rows: Sequence[dict],
    predictions: Sequence[str],
) -> dict:
    if len(rows) != len(predictions):
        raise ValueError(f"rows/predictions length mismatch: {len(rows)} != {len(predictions)}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        ground_truth_path = tmp / "ground_truth.csv"
        results_path = tmp / "results.csv"
        write_ground_truth_csv(rows, ground_truth_path)
        write_results_csv(predictions, results_path)
        score, f1, der, per = compute_score(str(ground_truth_path), str(results_path))

    return {
        "score": float(score),
        "f1": float(f1),
        "der": float(der),
        "per": float(per),
    }


def build_compute_metrics(
    id_to_token: Sequence[str],
    blank_id: int,
    eval_rows: Sequence[dict],
):
    def compute_metrics(eval_pred):
        logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        predictions = decode_logits(logits, id_to_token, blank_id)
        return compute_mdd_metrics(eval_rows, predictions)

    return compute_metrics
