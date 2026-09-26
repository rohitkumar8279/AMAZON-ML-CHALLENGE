"""Train the V7 candidate/feature pipeline with a CatBoost GPU classifier.

Candidate indexing and pair-feature generation intentionally stay on CPU.
CatBoost trains on both Kaggle T4 GPUs by default; use --task-type CPU for the
same data pipeline with CPU model fitting. The full Source-1/S2/S3 tables are
used. Only easy negative *pairs* are downsampled by the shared V7 builder.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool

import train_v7 as common
from pipeline_v7 import FEATURE_NAMES


def fit_model(
    x_path: Path,
    y_path: Path,
    rows: int,
    args: argparse.Namespace,
) -> CatBoostClassifier:
    if rows <= 0:
        raise RuntimeError("No training pairs were generated")
    x = common.matrix(x_path, rows)
    y = np.memmap(y_path, dtype=np.uint8, mode="r", shape=(rows,))
    pool = Pool(data=x, label=y, feature_names=list(FEATURE_NAMES))

    params = {
        "iterations": args.rounds,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_seed": args.seed,
        "thread_count": args.threads,
        "task_type": args.task_type,
        "verbose": args.verbose,
        "allow_writing_files": False,
    }
    if args.task_type == "GPU":
        params["devices"] = args.devices
        params["gpu_ram_part"] = args.gpu_ram_part
        params["border_count"] = args.border_count

    model = CatBoostClassifier(**params)
    model.fit(pool)
    del pool, x, y
    gc.collect()
    return model


def predict_memmap(
    model: CatBoostClassifier,
    x_path: Path,
    rows: int,
    work_dir: Path,
    chunk_rows: int,
    prediction_task_type: str,
) -> np.memmap:
    output_path = work_dir / "val_prob_v7_catboost.f64"
    output = np.memmap(output_path, dtype=np.float64, mode="w+", shape=(rows,))
    x = common.matrix(x_path, rows)
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        batch = np.asarray(x[start:end], dtype=np.float32)
        output[start:end] = model.predict_proba(
            batch, task_type=prediction_task_type, thread_count=1, verbose=False
        )[:, 1]
        if end == rows or end % 1_000_000 < chunk_rows:
            print(f"  scored {end:,}/{rows:,} validation pairs", flush=True)
    output.flush()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/v7_gpu"))
    parser.add_argument("--max-candidates-per-source", type=int, default=64)
    parser.add_argument("--max-block-size", type=int, default=200)
    parser.add_argument("--hard-negatives-per-source", type=int, default=8)
    parser.add_argument("--holdout-per-mille", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 2))
    parser.add_argument("--task-type", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--devices", default="0:1", help="CatBoost GPU IDs, e.g. 0:1 for both Kaggle T4s")
    parser.add_argument("--gpu-ram-part", type=float, default=0.75)
    parser.add_argument("--border-count", type=int, default=128)
    parser.add_argument("--prediction-task-type", choices=("GPU", "CPU"), default="CPU")
    parser.add_argument("--prediction-chunk-rows", type=int, default=50_000)
    parser.add_argument("--verbose", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_candidates_per_source < 1:
        raise SystemExit("--max-candidates-per-source must be positive")
    if not 1 <= args.holdout_per_mille <= 100:
        raise SystemExit("--holdout-per-mille must be between 1 and 100")
    if args.max_block_size < 2 or args.hard_negatives_per_source < 1:
        raise SystemExit("Block size must be >=2 and hard-negative count must be >=1")
    if args.depth < 4 or args.depth > 10:
        raise SystemExit("For the Kaggle T4 memory budget, choose CatBoost depth from 4 through 10")

    train_dir = args.train_dir.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    truth_path = train_dir / "train_ground_truth.tsv"
    if not truth_path.exists():
        raise FileNotFoundError(f"Training labels not found: {truth_path}")

    print("Reading all ground-truth groups...", flush=True)
    truth = common.read_truth(truth_path)
    print(f"Loaded {len(truth):,} non-singleton Source-1 groups", flush=True)
    print(
        f"Pair generation: CPU; CatBoost fit: {args.task_type}"
        + (f" on devices {args.devices}" if args.task_type == "GPU" else ""),
        flush=True,
    )

    tables = common.build_pair_tables(
        train_dir,
        work_dir,
        truth,
        args.max_candidates_per_source,
        args.max_block_size,
        args.hard_negatives_per_source,
        args.holdout_per_mille,
        args.seed,
    )
    train_x, train_y, val_x, val_meta, train_rows, val_rows, val_ids, val_true, val_found = tables

    print(f"Fitting development CatBoost model on {train_rows:,} pairs...", flush=True)
    dev_model = fit_model(train_x, train_y, train_rows, args)
    val_prob = predict_memmap(
        dev_model,
        val_x,
        val_rows,
        work_dir,
        args.prediction_chunk_rows,
        args.prediction_task_type,
    )
    threshold2, threshold3, val_macro, threshold_converged = common.threshold_sweep(
        val_meta, val_prob, len(val_ids), val_true
    )
    recall = sum(val_found) / sum(val_true) if sum(val_true) else 1.0
    oracle = common.candidate_oracle(val_found, val_true)
    singleton_count = sum(count == 0 for count in val_true)
    print(
        f"Validation: macro F0.5={val_macro:.6f}; oracle={oracle:.6f}; "
        f"pair candidate recall={recall:.6f}; thresholds S2={threshold2:.4f}, S3={threshold3:.4f}",
        flush=True,
    )

    del dev_model, val_prob
    gc.collect()
    common.append_binary(val_x, train_x)
    meta = np.memmap(val_meta, dtype=common.META_DTYPE, mode="r", shape=(val_rows,))
    with open(train_y, "ab") as handle:
        meta["label"].tofile(handle)
    full_rows = train_rows + val_rows
    print(f"Refitting production model on all {full_rows:,} generated labelled pairs...", flush=True)
    final_model = fit_model(train_x, train_y, full_rows, args)

    model_name = "entity_matcher_v7_catboost.cbm"
    metadata_name = "entity_matcher_v7_catboost.json"
    model_path = work_dir / model_name
    final_model.save_model(str(model_path))
    metadata = {
        "model": model_name,
        "backend": "catboost",
        "task_type": args.task_type,
        "devices": args.devices if args.task_type == "GPU" else None,
        "feature_names": list(FEATURE_NAMES),
        "max_candidates_per_source": args.max_candidates_per_source,
        "max_block_size": args.max_block_size,
        "hard_negatives_per_source": args.hard_negatives_per_source,
        "threshold_source2": threshold2,
        "threshold_source3": threshold3,
        "threshold_coordinate_refinement_converged": threshold_converged,
        "target_exclusivity": "highest_probability_candidate_per_target_id",
        "validation_entities": len(val_ids),
        "validation_singletons": singleton_count,
        "validation_pair_candidate_recall": recall,
        "validation_candidate_oracle_macro_f05": oracle,
        "validation_macro_f05_after_threshold_and_exclusivity": val_macro,
        "rounds": args.rounds,
        "depth": args.depth,
        "seed": args.seed,
        "note": "Blocking and pair features run on CPU. GPU fitting uses the selected CatBoost devices. Validate on the same held-out entities as LightGBM before comparing.",
    }
    (work_dir / metadata_name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = {
        "backend": "catboost",
        "task_type": args.task_type,
        "devices": args.devices if args.task_type == "GPU" else None,
        "macro_f05": val_macro,
        "candidate_oracle_macro_f05": oracle,
        "pair_candidate_recall": recall,
        "validation_entities": len(val_ids),
        "singleton_entities": singleton_count,
        "threshold_source2": threshold2,
        "threshold_source3": threshold3,
        "threshold_coordinate_refinement_converged": threshold_converged,
    }
    report_path = work_dir / "validation_report_v7_catboost.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model: {model_path}", flush=True)
    print(f"Saved validation report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
