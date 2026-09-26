"""Train GPU CatBoost and CPU LightGBM on the same full-data V8 pair set.

All source tables are indexed and every Source-1 entity participates: a stable
1% entity holdout is used to choose the model blend and thresholds, then the
selected model(s) are refit on every generated labelled pair. Only easy
negative *pairs* are sampled by the shared builder; positive pairs are kept.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier, Pool

import train_v8 as common
from pipeline_v8 import FEATURE_NAMES


def _catboost_params(args: argparse.Namespace, iterations: int) -> dict:
    return {
        "iterations": iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_seed": args.seed,
        "thread_count": args.threads,
        "task_type": "GPU",
        "devices": args.devices,
        "gpu_ram_part": args.gpu_ram_part,
        "border_count": args.border_count,
        "metric_period": 25,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.85,
        "allow_writing_files": True,
        "verbose": args.verbose,
    }


def fit_catboost(
    x_path: Path,
    y_path: Path,
    rows: int,
    args: argparse.Namespace,
    iterations: int,
    snapshot_path: Path,
    eval_x: np.ndarray | None = None,
    eval_y: np.ndarray | None = None,
) -> tuple[CatBoostClassifier, int]:
    if rows <= 0:
        raise RuntimeError("No training pairs were generated")
    x = common.matrix(x_path, rows)
    y = np.memmap(y_path, dtype=np.uint8, mode="r", shape=(rows,))
    train_pool = Pool(data=x, label=y, feature_names=list(FEATURE_NAMES))
    eval_pool = None
    if eval_x is not None and eval_y is not None:
        eval_pool = Pool(data=eval_x, label=eval_y, feature_names=list(FEATURE_NAMES))
    params = _catboost_params(args, iterations)
    params["train_dir"] = str(snapshot_path.parent / "catboost_v8_info")
    model = CatBoostClassifier(**params)
    fit_kwargs = {
        "verbose": args.verbose,
        "save_snapshot": True,
        "snapshot_file": str(snapshot_path),
        "snapshot_interval": 300,
    }
    if eval_pool is not None:
        fit_kwargs.update(
            eval_set=eval_pool,
            early_stopping_rounds=args.early_stopping_rounds,
            use_best_model=True,
        )
    model.fit(train_pool, **fit_kwargs)
    best_iteration = model.get_best_iteration()
    selected_iterations = max(1, best_iteration + 1 if best_iteration >= 0 else model.tree_count_)
    del train_pool, eval_pool, x, y
    gc.collect()
    return model, selected_iterations


def catboost_snapshot_path(
    work_dir: Path, phase: str, args: argparse.Namespace, rows: int,
    iterations: int, eval_rows: int = 0,
) -> Path:
    manifest_path = work_dir / "pair_tables_v8_manifest.json"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else "unpublished"
    config = {
        "phase": phase, "rows": rows, "eval_rows": eval_rows,
        "pair_manifest_sha256": manifest_sha,
        "iterations": iterations, "depth": args.depth,
        "learning_rate": args.learning_rate, "l2_leaf_reg": args.l2_leaf_reg,
        "seed": args.seed, "threads": args.threads, "devices": args.devices,
        "gpu_ram_part": args.gpu_ram_part, "border_count": args.border_count,
        "early_stopping_rounds": args.early_stopping_rounds,
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return work_dir / f"catboost_{phase}_{digest}.cbsnapshot"


def _lgb_params(args: argparse.Namespace) -> dict:
    params = common.lgb_params(args.seed, args.threads)
    params.update({"metric": "auc", "learning_rate": 0.04, "max_bin": 127})
    return params


def fit_lightgbm(
    x_path: Path,
    y_path: Path,
    rows: int,
    args: argparse.Namespace,
    rounds: int,
    eval_x: np.ndarray | None = None,
    eval_y: np.ndarray | None = None,
) -> tuple[lgb.Booster, int]:
    if rows <= 0:
        raise RuntimeError("No training pairs were generated")
    x = common.matrix(x_path, rows)
    y = np.memmap(y_path, dtype=np.uint8, mode="r", shape=(rows,))
    train_set = lgb.Dataset(x, label=y, feature_name=list(FEATURE_NAMES), free_raw_data=True)
    valid_sets = None
    callbacks = [lgb.log_evaluation(args.verbose)] if args.verbose else []
    if eval_x is not None and eval_y is not None:
        valid_sets = [lgb.Dataset(eval_x, label=eval_y, reference=train_set, free_raw_data=True)]
        callbacks += [lgb.early_stopping(args.early_stopping_rounds, verbose=True)]
    booster = lgb.train(
        _lgb_params(args), train_set, num_boost_round=rounds,
        valid_sets=valid_sets, valid_names=["entity_holdout"] if valid_sets else None,
        callbacks=callbacks,
    )
    selected_rounds = max(1, booster.best_iteration or rounds)
    del train_set, valid_sets, x, y
    gc.collect()
    return booster, selected_rounds


def predict_catboost_memmap(
    model: CatBoostClassifier, x_path: Path, rows: int, work_dir: Path, chunk_rows: int
) -> np.memmap:
    output = np.memmap(work_dir / "val_prob_v8_catboost.f64", dtype=np.float64, mode="w+", shape=(rows,))
    x = common.matrix(x_path, rows)
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        output[start:end] = model.predict_proba(
            np.asarray(x[start:end], dtype=np.float32), thread_count=2, verbose=False
        )[:, 1]
        if end == rows or end % 1_000_000 < chunk_rows:
            print(f"  CatBoost scored {end:,}/{rows:,} validation pairs", flush=True)
    output.flush()
    del x
    return output


def predict_lgb_memmap(booster: lgb.Booster, x_path: Path, rows: int, work_dir: Path, chunk_rows: int) -> np.memmap:
    output = np.memmap(work_dir / "val_prob_v8_lightgbm.f64", dtype=np.float64, mode="w+", shape=(rows,))
    x = common.matrix(x_path, rows)
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        output[start:end] = booster.predict(x[start:end], num_threads=max(1, min(os.cpu_count() or 2, 4)))
        if end == rows or end % 1_000_000 < chunk_rows:
            print(f"  LightGBM scored {end:,}/{rows:,} validation pairs", flush=True)
    output.flush()
    del x
    return output


def append_holdout_training_sample(
    val_x_path: Path,
    val_meta_path: Path,
    train_x_path: Path,
    train_y_path: Path,
    val_rows: int,
    hard_negatives_per_source: int,
) -> int:
    """Append all validation positives plus the same top-hard-negative quota.

    Validation features include every candidate so the metric/threshold sweep
    sees the real capped candidate distribution. Refitting on all validation
    negatives would shift the sampled training prior, so retain positives and
    the first K nonmatches per source/query just as the development set does.
    """
    meta = np.memmap(val_meta_path, dtype=common.META_DTYPE, mode="r", shape=(val_rows,))
    labels = meta["label"]
    keep = labels == 1
    negative_rows = np.flatnonzero(labels == 0)
    if len(negative_rows):
        # build_pair_tables writes source 2 then source 3; within each source it
        # writes validation queries and their ranked candidates in ascending
        # order. Therefore each source/query key is already a contiguous run.
        negative_keys = (
            meta["source"][negative_rows].astype(np.int64) << 32
        ) | meta["query"][negative_rows].astype(np.int64)
        if np.any(negative_keys[1:] < negative_keys[:-1]):
            raise RuntimeError("Validation metadata is not ordered by source/query as expected")
        starts = np.empty(len(negative_rows), dtype=bool)
        starts[0] = True
        starts[1:] = negative_keys[1:] != negative_keys[:-1]
        row_positions = np.arange(len(negative_rows), dtype=np.int64)
        group_start = np.maximum.accumulate(np.where(starts, row_positions, 0))
        negative_rank = row_positions - group_start
        keep[negative_rows[negative_rank < hard_negatives_per_source]] = True

    selected_rows = np.flatnonzero(keep)
    val_features = common.matrix(val_x_path, val_rows)
    with open(train_x_path, "ab") as x_handle:
        for start in range(0, len(selected_rows), 100_000):
            batch = np.asarray(val_features[selected_rows[start:start + 100_000]], dtype=np.float32)
            batch.tofile(x_handle)
    with open(train_y_path, "ab") as y_handle:
        labels[selected_rows].tofile(y_handle)
    del val_features, meta
    gc.collect()
    return len(selected_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/v8_gpu"))
    parser.add_argument("--max-candidates-per-source", type=int, default=128)
    parser.add_argument("--max-block-size", type=int, default=200)
    parser.add_argument("--max-address-block-size", type=int, default=1200)
    parser.add_argument("--hard-negatives-per-source", type=int, default=8)
    parser.add_argument("--holdout-per-mille", type=int, default=10)
    parser.add_argument("--catboost-rounds", type=int, default=1200)
    parser.add_argument("--lightgbm-rounds", type=int, default=1600)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 2))
    parser.add_argument("--devices", default="0:1")
    parser.add_argument("--gpu-ram-part", type=float, default=0.75)
    parser.add_argument("--border-count", type=int, default=64)
    parser.add_argument("--prediction-chunk-rows", type=int, default=50_000)
    parser.add_argument("--checkpoint-rows", type=int, default=50_000,
                        help="Flush and checkpoint pair generation after this many Source-1 rows")
    parser.add_argument("--rebuild-pair-tables", action="store_true",
                        help="Ignore a complete feature-table cache and rebuild candidate/feature pairs")
    parser.add_argument("--verbose", type=int, default=100)
    return parser.parse_args()


def _pair_cache_config(train_dir: Path, args: argparse.Namespace) -> dict:
    input_files = [
        train_dir / "train_source1.tsv", train_dir / "train_source2.tsv",
        train_dir / "train_source3.tsv", train_dir / "train_ground_truth.tsv",
    ]
    inputs = {}
    for path in input_files:
        size = path.stat().st_size
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            digest.update(handle.read(65_536))
            handle.seek(max(0, size - 65_536))
            digest.update(handle.read(65_536))
        inputs[path.name] = {"size": size, "edge_sha256": digest.hexdigest()}
    code_signatures = {}
    for path in (Path(__file__).with_name("pipeline_v8.py"), Path(__file__).with_name("train_v8.py")):
        code_signatures[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": 8,
        "inputs": inputs,
        "code_sha256": code_signatures,
        "max_candidates_per_source": args.max_candidates_per_source,
        "max_block_size": args.max_block_size,
        "max_address_block_size": args.max_address_block_size,
        "hard_negatives_per_source": args.hard_negatives_per_source,
        "holdout_per_mille": args.holdout_per_mille,
        "seed": args.seed,
        "feature_count": len(FEATURE_NAMES),
    }


def load_pair_table_cache(train_dir: Path, work_dir: Path, args: argparse.Namespace):
    manifest_path = work_dir / "pair_tables_v8_manifest.json"
    if args.rebuild_pair_tables or not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != _pair_cache_config(train_dir, args):
            print("Pair-table cache configuration differs; rebuilding candidate/feature tables.", flush=True)
            return None
        train_rows, val_rows = int(manifest["train_rows"]), int(manifest["val_rows"])
        paths = {
            "train_x": work_dir / "dev_x.f32",
            "train_y": work_dir / "dev_y.u8",
            "val_x": work_dir / "val_x.f32",
            "val_meta": work_dir / "val_meta.bin",
        }
        expected_sizes = {
            "train_x": train_rows * len(FEATURE_NAMES) * np.dtype(np.float32).itemsize,
            "train_y": train_rows * np.dtype(np.uint8).itemsize,
            "val_x": val_rows * len(FEATURE_NAMES) * np.dtype(np.float32).itemsize,
            "val_meta": val_rows * common.META_DTYPE.itemsize,
        }
        if any(not paths[key].is_file() or paths[key].stat().st_size != size
               for key, size in expected_sizes.items()):
            print("Pair-table cache files are incomplete; rebuilding candidate/feature tables.", flush=True)
            return None
        print(f"Reusing complete V8 pair-table cache: {train_rows:,} train / {val_rows:,} validation pairs", flush=True)
        return (
            paths["train_x"], paths["train_y"], paths["val_x"], paths["val_meta"],
            train_rows, val_rows, manifest["val_ids"], manifest["val_true"], manifest["val_found"],
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Pair-table cache could not be validated ({exc}); rebuilding.", flush=True)
        return None


def save_pair_table_cache(train_dir: Path, work_dir: Path, args: argparse.Namespace, tables) -> None:
    train_x, train_y, val_x, val_meta, train_rows, val_rows, val_ids, val_true, val_found = tables
    manifest = {
        "config": _pair_cache_config(train_dir, args),
        "train_rows": train_rows,
        "val_rows": val_rows,
        "val_ids": val_ids,
        "val_true": val_true,
        "val_found": val_found,
    }
    target = work_dir / "pair_tables_v8_manifest.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    temporary.replace(target)


def main() -> None:
    args = parse_args()
    if args.max_candidates_per_source < 1:
        raise SystemExit("--max-candidates-per-source must be positive")
    if not 1 <= args.holdout_per_mille <= 100:
        raise SystemExit("--holdout-per-mille must be between 1 and 100")
    if min(args.max_block_size, args.max_address_block_size) < 2 or args.hard_negatives_per_source < 1:
        raise SystemExit("Posting limits must be >=2 and hard-negative count must be >=1")
    if not 4 <= args.depth <= 10:
        raise SystemExit("For Kaggle T4 memory, choose CatBoost depth from 4 through 10")

    train_dir = args.train_dir.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    truth_path = train_dir / "train_ground_truth.tsv"
    if not truth_path.exists():
        raise FileNotFoundError(f"Training labels not found: {truth_path}")

    print(
        f"Full source files; candidate cap={args.max_candidates_per_source}/source; "
        f"CatBoost GPU {args.devices} + LightGBM CPU comparison",
        flush=True,
    )
    tables = load_pair_table_cache(train_dir, work_dir, args)
    if tables is None:
        print("Reading all ground-truth groups...", flush=True)
        truth = common.read_truth(truth_path)
        print(f"Loaded {len(truth):,} non-singleton Source-1 groups", flush=True)
        tables = common.build_pair_tables(
            train_dir, work_dir, truth,
            args.max_candidates_per_source, args.max_block_size, args.max_address_block_size,
            args.hard_negatives_per_source, args.holdout_per_mille, args.seed,
            checkpoint_rows=args.checkpoint_rows,
            resume=not args.rebuild_pair_tables,
        )
        save_pair_table_cache(train_dir, work_dir, args, tables)
        del truth
        gc.collect()
    else:
        print("Using cached validated pair tables; skipping full candidate/feature generation.", flush=True)
    train_x, train_y, val_x, val_meta, train_rows, val_rows, val_ids, val_true, val_found = tables
    val_x_mem = common.matrix(val_x, val_rows)
    meta = np.memmap(val_meta, dtype=common.META_DTYPE, mode="r", shape=(val_rows,))
    val_labels = meta["label"]

    print(f"Training CatBoost development model on {train_rows:,} sampled pairs...", flush=True)
    cat_dev, cat_iterations = fit_catboost(
        train_x, train_y, train_rows, args, args.catboost_rounds,
        catboost_snapshot_path(work_dir, "development", args, train_rows, args.catboost_rounds, val_rows),
        val_x_mem, val_labels,
    )
    cat_prob = predict_catboost_memmap(
        cat_dev, val_x, val_rows, work_dir, args.prediction_chunk_rows
    )
    del cat_dev
    gc.collect()

    print(f"Training LightGBM CPU comparator on {train_rows:,} sampled pairs...", flush=True)
    lgb_dev, lgb_rounds = fit_lightgbm(
        train_x, train_y, train_rows, args, args.lightgbm_rounds, val_x_mem, val_labels
    )
    lgb_prob = predict_lgb_memmap(lgb_dev, val_x, val_rows, work_dir, args.prediction_chunk_rows)
    del lgb_dev, val_x_mem, meta, val_labels
    gc.collect()

    model_metrics = {}
    blend_weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    for cat_weight in blend_weights:
        if cat_weight == 0.0:
            label = "lightgbm"
            probabilities = lgb_prob
        elif cat_weight == 1.0:
            label = "catboost"
            probabilities = cat_prob
        else:
            label = f"blend_catboost_{cat_weight:.2f}"
            probabilities = cat_weight * cat_prob + (1.0 - cat_weight) * lgb_prob
        t2, t3, score, converged = common.threshold_sweep(
            val_meta, probabilities, len(val_ids), val_true
        )
        model_metrics[label] = {
            "catboost_weight": cat_weight,
            "lightgbm_weight": 1.0 - cat_weight,
            "macro_f05": score,
            "threshold_source2": t2,
            "threshold_source3": t3,
            "threshold_coordinate_refinement_converged": converged,
        }
        print(f"Validation {label}: macro F0.5={score:.6f}; S2={t2:.6f}; S3={t3:.6f}", flush=True)

    selected_name = max(model_metrics, key=lambda name: model_metrics[name]["macro_f05"])
    selected = model_metrics[selected_name]
    cat_weight = float(selected["catboost_weight"])
    threshold2 = float(selected["threshold_source2"])
    threshold3 = float(selected["threshold_source3"])
    val_macro = float(selected["macro_f05"])
    del cat_prob, lgb_prob
    gc.collect()

    recall = sum(val_found) / sum(val_true) if sum(val_true) else 1.0
    oracle = common.candidate_oracle(val_found, val_true)
    singleton_count = sum(count == 0 for count in val_true)
    print(
        f"Selected {selected_name}; candidate-only macro F0.5 ceiling={oracle:.6f}; "
        f"pair recall={recall:.6f}; model macro F0.5={val_macro:.6f}; "
        f"thresholds S2={threshold2:.6f}, S3={threshold3:.6f}",
        flush=True,
    )

    # Keep the holdout's complete candidate set for validation above, but use
    # the same pair-sampling policy when adding it to the full-data refit.
    holdout_train_rows = append_holdout_training_sample(
        val_x, val_meta, train_x, train_y, val_rows, args.hard_negatives_per_source
    )
    full_rows = train_rows + holdout_train_rows
    print(f"Refitting selected model(s) on all {full_rows:,} generated labelled pairs...", flush=True)

    model_files = {}
    if cat_weight > 0:
        cat_final, _ = fit_catboost(
            train_x, train_y, full_rows, args, cat_iterations,
            catboost_snapshot_path(work_dir, "production", args, full_rows, cat_iterations),
        )
        cat_name = "entity_matcher_v8_catboost.cbm"
        cat_final.save_model(str(work_dir / cat_name))
        model_files["catboost"] = cat_name
        del cat_final
        gc.collect()
    if cat_weight < 1:
        lgb_final, _ = fit_lightgbm(train_x, train_y, full_rows, args, lgb_rounds)
        lgb_name = "entity_matcher_v8_lightgbm.txt"
        lgb_final.save_model(str(work_dir / lgb_name))
        model_files["lightgbm"] = lgb_name
        del lgb_final
        gc.collect()

    metadata = {
        "model_files": model_files,
        "pair_table_config": _pair_cache_config(train_dir, args),
        "backend": "catboost_gpu_lightgbm_cpu_blend_selection",
        "catboost_weight": cat_weight,
        "lightgbm_weight": 1.0 - cat_weight,
        "feature_names": list(FEATURE_NAMES),
        "max_candidates_per_source": args.max_candidates_per_source,
        "max_block_size": args.max_block_size,
        "max_address_block_size": args.max_address_block_size,
        "hard_negatives_per_source": args.hard_negatives_per_source,
        "threshold_source2": threshold2,
        "threshold_source3": threshold3,
        "threshold_coordinate_refinement_converged": selected["threshold_coordinate_refinement_converged"],
        "target_exclusivity": "highest_probability_candidate_per_target_id",
        "validation_entities": len(val_ids),
        "validation_singletons": singleton_count,
        "validation_pair_candidate_recall": recall,
        "validation_candidate_oracle_macro_f05": oracle,
        "validation_macro_f05_after_threshold_and_exclusivity": val_macro,
        "model_selection_validation_metrics": model_metrics,
        "catboost_rounds_selected_from_holdout": cat_iterations,
        "lightgbm_rounds_selected_from_holdout": lgb_rounds,
        "training_config": {
            "catboost_rounds_requested": args.catboost_rounds,
            "lightgbm_rounds_requested": args.lightgbm_rounds,
            "early_stopping_rounds": args.early_stopping_rounds,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
            "border_count": args.border_count,
            "metric_period": 25,
            "devices": args.devices,
            "seed": args.seed,
            "threads": args.threads,
            "holdout_per_mille": args.holdout_per_mille,
            "checkpoint_rows": args.checkpoint_rows,
        },
        "training_rows_after_holdout_refit": full_rows,
        "note": (
            "All Source-1 and target source records are indexed. The 1% entity holdout is used "
            "for model/blend/threshold selection then included in the production refit. Only easy "
            "negative pairs are sampled. Candidate-only oracle is a hard ceiling for this candidate set."
        ),
    }
    metadata_path = work_dir / "entity_matcher_v8.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = {
        "selected_model": selected_name,
        "catboost_weight": cat_weight,
        "lightgbm_weight": 1.0 - cat_weight,
        "macro_f05": val_macro,
        "candidate_oracle_macro_f05": oracle,
        "pair_candidate_recall": recall,
        "validation_entities": len(val_ids),
        "singleton_entities": singleton_count,
        "threshold_source2": threshold2,
        "threshold_source3": threshold3,
        "candidate_cap_per_source": args.max_candidates_per_source,
        "candidate_model_metrics": model_metrics,
    }
    report_path = work_dir / "validation_report_v8.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model metadata: {metadata_path}", flush=True)
    print(f"Saved validation report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
