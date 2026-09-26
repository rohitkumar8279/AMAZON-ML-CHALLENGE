"""Official challenge scorer: macro-average F0.5 over every Source-1 row.

The challenge defines an empty/empty singleton prediction as F0.5 = 1. Missing
prediction rows are treated as empty predictions, so a missing row for a
non-singleton correctly scores zero. Pair-level micro metrics are reported
separately and are never labelled as macro F0.5.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Set


@dataclass(frozen=True)
class Evaluation:
    macro_f05: float
    micro_f05: float
    micro_precision: float
    micro_recall: float
    singleton_accuracy: float
    singleton_count: int
    entity_count: int
    true_positives: int
    false_positives: int
    false_negatives: int


def _read_pairs(path: str, id_column: str, matches_column: str) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or id_column not in reader.fieldnames or matches_column not in reader.fieldnames:
            raise ValueError(
                f"{path!r} must contain TSV columns {id_column!r} and {matches_column!r}"
            )
        for row in reader:
            entity_id = (row.get(id_column) or "").strip()
            if not entity_id:
                raise ValueError(f"{path!r} contains an empty {id_column!r}")
            if entity_id in result:
                raise ValueError(f"{path!r} contains duplicate entity {entity_id!r}")
            raw_matches = row.get(matches_column) or ""
            result[entity_id] = {value.strip() for value in raw_matches.split(",") if value.strip()}
    return result


def _f05(tp: int, predicted: int, actual: int) -> float:
    if predicted == 0 and actual == 0:
        return 1.0
    denominator = predicted + 0.25 * actual
    return 1.25 * tp / denominator if denominator else 0.0


def evaluate_files(pred_file: str, gt_file: str) -> Evaluation:
    gt = _read_pairs(gt_file, "source1_entity_id", "matched_entity_ids")
    pred = _read_pairs(pred_file, "source1_entity_id", "matched_entity_ids")

    # Extra prediction IDs cannot improve the score, but count their links as
    # false positives. This also makes malformed/incomplete submissions visible.
    extra_ids = pred.keys() - gt.keys()
    tp = fp = fn = 0
    entity_scores = []
    singleton_count = singleton_correct = 0

    for entity_id, true_matches in gt.items():
        predicted_matches = pred.get(entity_id, set())
        entity_tp = len(predicted_matches & true_matches)
        entity_fp = len(predicted_matches - true_matches)
        entity_fn = len(true_matches - predicted_matches)
        tp += entity_tp
        fp += entity_fp
        fn += entity_fn
        entity_scores.append(_f05(entity_tp, len(predicted_matches), len(true_matches)))
        if not true_matches:
            singleton_count += 1
            singleton_correct += not predicted_matches

    fp += sum(len(pred[entity_id]) for entity_id in extra_ids)
    micro_precision = tp / (tp + fp) if tp + fp else 1.0
    micro_recall = tp / (tp + fn) if tp + fn else 1.0
    micro_f05 = _f05(tp, tp + fp, tp + fn)
    macro_f05 = sum(entity_scores) / len(entity_scores) if entity_scores else 0.0
    singleton_accuracy = singleton_correct / singleton_count if singleton_count else 1.0

    return Evaluation(
        macro_f05=macro_f05,
        micro_f05=micro_f05,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        singleton_accuracy=singleton_accuracy,
        singleton_count=singleton_count,
        entity_count=len(gt),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def evaluate(pred_file: str, gt_file: str) -> Evaluation:
    result = evaluate_files(pred_file, gt_file)
    print(f"Macro F0.5: {result.macro_f05:.6f} ({result.entity_count:,} entities)")
    print(
        "Micro pair metrics: "
        f"F0.5={result.micro_f05:.6f}, "
        f"P={result.micro_precision:.6f}, R={result.micro_recall:.6f}"
    )
    print(
        f"Singleton accuracy: {result.singleton_accuracy:.6f} "
        f"({result.singleton_count:,} singleton entities)"
    )
    print(
        f"TP={result.true_positives:,} FP={result.false_positives:,} "
        f"FN={result.false_negatives:,}"
    )
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python evaluator.py PREDICTIONS.tsv GROUND_TRUTH.tsv")
    evaluate(sys.argv[1], sys.argv[2])
