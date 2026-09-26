"""Train a Kaggle-ready LightGBM matcher from the full labelled dataset.

The held-out slice is selected by a stable hash of Source-1 IDs (default 1%).
All remaining Source-1 entities and all ground-truth positives are used for
model fitting; only easy pair negatives are downsampled. The holdout retains
the complete capped candidate set for metric/threshold evaluation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import lightgbm as lgb
import numpy as np

from pipeline_v8 import CandidateIndex, FEATURE_NAMES, make_record, pair_features


META_DTYPE = np.dtype([("query", "<i4"), ("target", "<i8"), ("source", "u1"), ("label", "u1")])


class RawFeatureWriter:
    def __init__(
        self, x_path: Path, y_path: Path | None = None, flush_rows: int = 20_000,
        mode: str = "wb", row_count: int = 0,
    ):
        self.x_handle = open(x_path, mode)
        self.y_handle = open(y_path, mode) if y_path else None
        self.flush_rows = flush_rows
        self.features: List[List[float]] = []
        self.labels: List[int] = []
        self.row_count = row_count

    def add(self, features: Sequence[float], label: int | None = None) -> None:
        self.features.append(features)
        if self.y_handle:
            self.labels.append(int(label or 0))
        if len(self.features) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.features:
            return
        matrix = np.asarray(self.features, dtype=np.float32)
        matrix.tofile(self.x_handle)
        if self.y_handle:
            np.asarray(self.labels, dtype=np.uint8).tofile(self.y_handle)
        self.row_count += len(self.features)
        self.features.clear()
        self.labels.clear()

    def close(self) -> int:
        self.flush()
        self.x_handle.close()
        if self.y_handle:
            self.y_handle.close()
        return self.row_count


class RawMetaWriter:
    def __init__(self, path: Path, flush_rows: int = 50_000, mode: str = "wb", row_count: int = 0):
        self.handle = open(path, mode)
        self.flush_rows = flush_rows
        self.rows: List[Tuple[int, int, int, int]] = []
        self.row_count = row_count

    def add(self, query_index: int, target_key: int, source: int, label: int) -> None:
        self.rows.append((query_index, target_key, source, label))
        if len(self.rows) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        block = np.asarray(self.rows, dtype=META_DTYPE)
        block.tofile(self.handle)
        self.row_count += len(self.rows)
        self.rows.clear()

    def close(self) -> int:
        self.flush()
        self.handle.close()
        return self.row_count


def read_truth(path: Path) -> Dict[str, Set[str]]:
    truth: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            matches = row.get("matched_entity_ids") or ""
            if matches:
                truth[row["source1_entity_id"]] = {value.strip() for value in matches.split(",") if value.strip()}
    return truth


def is_validation(entity_id: str, holdout_per_mille: int, seed: int) -> bool:
    digest = hashlib.blake2b(f"{seed}:{entity_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 1000 < holdout_per_mille


def _generation_fingerprint(
    train_dir: Path, cap: int, block_cap: int, address_block_cap: int,
    negatives_per_source: int, holdout_per_mille: int, seed: int,
) -> str:
    signatures = {}
    for name in (
        "train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"
    ):
        path = train_dir / name
        size = path.stat().st_size
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            digest.update(handle.read(65_536))
            handle.seek(max(0, size - 65_536))
            digest.update(handle.read(65_536))
        signatures[name] = {"size": size, "edge_sha256": digest.hexdigest()}
    code_sha = hashlib.sha256(Path(__file__).with_name("pipeline_v8.py").read_bytes()).hexdigest()
    builder_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = {
        "inputs": signatures,
        "pipeline_sha256": code_sha,
        "builder_sha256": builder_sha,
        "feature_names": list(FEATURE_NAMES),
        "cap": cap,
        "block_cap": block_cap,
        "address_block_cap": address_block_cap,
        "negatives_per_source": negatives_per_source,
        "holdout_per_mille": holdout_per_mille,
        "seed": seed,
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def _write_generation_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def _sync_writer(writer) -> None:
    if isinstance(writer, RawFeatureWriter):
        writer.x_handle.flush()
        os.fsync(writer.x_handle.fileno())
        if writer.y_handle:
            writer.y_handle.flush()
            os.fsync(writer.y_handle.fileno())
    else:
        writer.handle.flush()
        os.fsync(writer.handle.fileno())


def read_rows(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def build_pair_tables(
    train_dir: Path,
    work_dir: Path,
    truth: Dict[str, Set[str]],
    cap: int,
    block_cap: int,
    address_block_cap: int,
    negatives_per_source: int,
    holdout_per_mille: int,
    seed: int,
    checkpoint_rows: int = 50_000,
    resume: bool = True,
):
    if checkpoint_rows < 1:
        raise ValueError("checkpoint_rows must be positive")
    train_x_path = work_dir / "dev_x.f32"
    train_y_path = work_dir / "dev_y.u8"
    val_x_path = work_dir / "val_x.f32"
    val_meta_path = work_dir / "val_meta.bin"
    paths = {
        "train_x": train_x_path, "train_y": train_y_path,
        "val_x": val_x_path, "val_meta": val_meta_path,
    }
    state_path = work_dir / "pair_generation_v8_state.json"
    fingerprint = _generation_fingerprint(
        train_dir, cap, block_cap, address_block_cap, negatives_per_source,
        holdout_per_mille, seed,
    )
    state = None
    if resume and state_path.is_file():
        try:
            candidate_state = json.loads(state_path.read_text(encoding="utf-8"))
            if candidate_state.get("fingerprint") == fingerprint:
                for key, path in paths.items():
                    expected_size = int(candidate_state["file_sizes"][key])
                    if not path.is_file() or path.stat().st_size < expected_size:
                        raise RuntimeError(
                            f"Pair-generation checkpoint is ahead of {path}; "
                            "preserve the work directory and inspect the missing artifact."
                        )
                    with open(path, "r+b") as handle:
                        handle.truncate(expected_size)
                state = candidate_state
                active_source = int(state["active_source"])
                if active_source in state["completed_sources"] and active_source < 3:
                    print(
                        f"Source {active_source} pair generation is complete; resuming with Source {active_source + 1}.",
                        flush=True,
                    )
                else:
                    print(
                        "Resuming pair/feature generation from checkpoint: "
                        f"Source {active_source} after "
                        f"{state['processed_rows'][str(active_source)]:,} Source-1 rows.",
                        flush=True,
                    )
            else:
                print("Pair-generation checkpoint belongs to different data/settings; rebuilding tables.", flush=True)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"Pair-generation checkpoint is invalid ({exc}); rebuilding tables.", flush=True)

    if state is None:
        state = {
            "fingerprint": fingerprint,
            "active_source": 2,
            "completed_sources": [],
            "processed_rows": {"2": 0, "3": 0},
            "row_counts": {"train": 0, "val": 0, "meta": 0},
            "file_sizes": {key: 0 for key in paths},
            "val_ids": [], "val_truth_count": [], "val_found_count": [],
        }
        mode = "wb"
        _write_generation_state(state_path, state)
    else:
        mode = "ab"
    row_counts = state["row_counts"]
    train_writer = RawFeatureWriter(
        train_x_path, train_y_path, mode=mode, row_count=int(row_counts["train"])
    )
    val_writer = RawFeatureWriter(val_x_path, mode=mode, row_count=int(row_counts["val"]))
    meta_writer = RawMetaWriter(
        val_meta_path, mode=mode, row_count=int(row_counts["meta"])
    )
    val_ids: List[str] = list(state["val_ids"])
    val_truth_count: List[int] = list(state["val_truth_count"])
    val_found_count: List[int] = list(state["val_found_count"])
    total_train_queries = 0
    total_val_queries = 0

    for source in (2, 3):
        if source in state["completed_sources"]:
            print(f"Source {source} pair table is complete; skipping query/feature pass.", flush=True)
            continue
        target_path = train_dir / f"train_source{source}.tsv"
        print(f"Indexing source {source}: {target_path}", flush=True)
        index = CandidateIndex(max_posting=block_cap, max_address_posting=address_block_cap)
        started = time.time()
        for count, row in enumerate(read_rows(target_path), start=1):
            index.add(make_record(row["entity_id"], row["business_name"], row["business_address"], row["country"], source))
            if count % 500_000 == 0:
                print(f"  indexed {count:,} records in {time.time() - started:.1f}s", flush=True)
        print(
            f"  indexed {len(index.records):,}; active blocks={index.block_count:,}; "
            f"frequency-pruned blocks={index.pruned_count:,}",
            flush=True,
        )

        val_query_index = 0
        resume_rows = int(state["processed_rows"][str(source)])
        query_started = time.time()
        for count, row in enumerate(read_rows(train_dir / "train_source1.tsv"), start=1):
            if count <= resume_rows:
                if is_validation(row["entity_id"], holdout_per_mille, seed):
                    val_query_index += 1
                continue
            query = make_record(row["entity_id"], row["business_name"], row["business_address"], row["country"], 1)
            true_ids = truth.get(query.entity_id, set())
            held_out = is_validation(query.entity_id, holdout_per_mille, seed)
            candidates = index.query(query, cap=cap)
            if held_out:
                if source == 2:
                    val_ids.append(query.entity_id)
                    val_truth_count.append(len(true_ids))
                    val_found_count.append(0)
                elif val_query_index >= len(val_ids) or val_ids[val_query_index] != query.entity_id:
                    raise RuntimeError("Validation split order changed between source passes")
                for candidate in candidates:
                    target = index.records[candidate.record_index]
                    label = int(target.entity_id in true_ids)
                    val_writer.add(pair_features(query, target, candidate, index))
                    meta_writer.add(
                        val_query_index,
                        (source << 32) | candidate.record_index,
                        source,
                        label,
                    )
                    if label:
                        val_found_count[val_query_index] += 1
                val_query_index += 1
                total_val_queries += int(source == 2)
            else:
                positives: List[Tuple[object, int]] = []
                negatives: List[Tuple[object, int]] = []
                for candidate in candidates:
                    target = index.records[candidate.record_index]
                    label = int(target.entity_id in true_ids)
                    (positives if label else negatives).append((candidate, label))
                # Candidate order is a cheap hard-negative rank. All retrieved
                # positives are retained, while easy negatives are sampled down.
                selected = positives + negatives[:negatives_per_source]
                for candidate, label in selected:
                    target = index.records[candidate.record_index]
                    train_writer.add(pair_features(query, target, candidate, index), label)
                total_train_queries += int(source == 2)

            if count % 100_000 == 0:
                elapsed = max(time.time() - query_started, 1e-9)
                print(
                    f"  processed {count:,} Source-1 rows at {count / elapsed:,.0f} rows/s; "
                    f"train queries={total_train_queries:,}; validation queries={total_val_queries:,}; "
                    f"pass elapsed={elapsed / 60:.1f} min",
                    flush=True,
                )
            if count % checkpoint_rows == 0:
                train_writer.flush()
                val_writer.flush()
                meta_writer.flush()
                for writer in (train_writer, val_writer, meta_writer):
                    _sync_writer(writer)
                state["active_source"] = source
                state["processed_rows"][str(source)] = count
                state["row_counts"] = {
                    "train": train_writer.row_count,
                    "val": val_writer.row_count,
                    "meta": meta_writer.row_count,
                }
                state["file_sizes"] = {key: path.stat().st_size for key, path in paths.items()}
                state["val_ids"] = val_ids
                state["val_truth_count"] = val_truth_count
                state["val_found_count"] = val_found_count
                _write_generation_state(state_path, state)

        # Make the finished source durable before moving on to the next index.
        train_writer.flush()
        val_writer.flush()
        meta_writer.flush()
        for writer in (train_writer, val_writer, meta_writer):
            _sync_writer(writer)
        state["active_source"] = source
        state["processed_rows"][str(source)] = count
        state["completed_sources"] = sorted(set(state["completed_sources"] + [source]))
        state["row_counts"] = {
            "train": train_writer.row_count,
            "val": val_writer.row_count,
            "meta": meta_writer.row_count,
        }
        state["file_sizes"] = {key: path.stat().st_size for key, path in paths.items()}
        state["val_ids"] = val_ids
        state["val_truth_count"] = val_truth_count
        state["val_found_count"] = val_found_count
        _write_generation_state(state_path, state)
        del index
        gc.collect()

    train_rows = train_writer.close()
    val_rows = val_writer.close()
    meta_rows = meta_writer.close()
    if val_rows != meta_rows:
        raise RuntimeError(f"Validation feature/meta row mismatch: {val_rows} != {meta_rows}")
    if not val_ids:
        raise RuntimeError("The deterministic validation split is empty")
    if len(val_ids) != len(set(val_ids)):
        raise RuntimeError("Validation query IDs are not unique")
    print(f"Pair tables: train={train_rows:,}, validation={val_rows:,}, entities={len(val_ids):,}", flush=True)
    # Leave the final checkpoint in place until the training caller atomically
    # publishes the complete pair-table manifest. A restart in that small gap
    # can still reuse these finished files without repeating candidate search.
    return train_x_path, train_y_path, val_x_path, val_meta_path, train_rows, val_rows, val_ids, val_truth_count, val_found_count


def lgb_params(seed: int, threads: int) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.04,
        "num_leaves": 47,
        "max_depth": -1,
        "max_bin": 127,
        "min_data_in_leaf": 120,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "num_threads": threads,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "force_col_wise": True,
    }


def matrix(path: Path, rows: int) -> np.memmap:
    return np.memmap(path, dtype=np.float32, mode="r", shape=(rows, len(FEATURE_NAMES)))


def train_booster(x_path: Path, y_path: Path, rows: int, rounds: int, seed: int, threads: int) -> lgb.Booster:
    if rows == 0:
        raise RuntimeError("No training pairs were generated")
    x = matrix(x_path, rows)
    y = np.memmap(y_path, dtype=np.uint8, mode="r", shape=(rows,))
    dataset = lgb.Dataset(x, label=y, feature_name=list(FEATURE_NAMES), free_raw_data=True)
    return lgb.train(lgb_params(seed, threads), dataset, num_boost_round=rounds)


def predict_memmap(booster: lgb.Booster, x_path: Path, rows: int, work_dir: Path) -> np.memmap:
    output_path = work_dir / "val_prob.f64"
    output = np.memmap(output_path, dtype=np.float64, mode="w+", shape=(rows,))
    x = matrix(x_path, rows)
    chunk = 100_000
    for start in range(0, rows, chunk):
        end = min(rows, start + chunk)
        output[start:end] = booster.predict(x[start:end], num_threads=-1)
        if end % 1_000_000 < chunk:
            print(f"  scored {end:,}/{rows:,} validation pairs", flush=True)
    output.flush()
    return output


def _macro_f05_for_counts(tp: np.ndarray, predicted: np.ndarray, actual: np.ndarray) -> float:
    denominator = predicted + 0.25 * actual
    scores = np.divide(1.25 * tp, denominator, out=np.zeros_like(denominator, dtype=np.float64), where=denominator > 0)
    scores[(predicted == 0) & (actual == 0)] = 1.0
    return float(scores.mean()) if len(scores) else 0.0


def _entity_scores(tp: np.ndarray, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    denominator = predicted + 0.25 * actual
    scores = np.divide(
        1.25 * tp,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )
    scores[(predicted == 0) & (actual == 0)] = 1.0
    return scores


def threshold_sweep(
    val_meta_path: Path,
    probabilities: np.ndarray,
    query_count: int,
    true_count: Sequence[int],
) -> Tuple[float, float, float, bool]:
    """Tune one threshold per target source after global target exclusivity."""
    rows = len(probabilities)
    meta = np.memmap(val_meta_path, dtype=META_DTYPE, mode="r", shape=(rows,))
    query = meta["query"]
    target = meta["target"]
    source = meta["source"]
    label = meta["label"].astype(np.uint8, copy=False)

    # A target row can be assigned to at most one Source-1 entity. Its highest
    # probability edge wins; all other edges to that target are discarded.
    order = np.lexsort((-probabilities, target))
    sorted_targets = target[order]
    winner = np.ones(rows, dtype=bool)
    winner[order[1:]] = sorted_targets[1:] != sorted_targets[:-1]
    actual = np.asarray(true_count, dtype=np.int32)
    thresholds = np.round(np.arange(0.05, 1.0, 0.01), 2)
    by_source = {}

    for source_id in (2, 3):
        eligible = winner & (source == source_id)
        pred_counts = np.zeros((query_count, len(thresholds)), dtype=np.int32)
        tp_counts = np.zeros_like(pred_counts)
        q_values = query[eligible]
        p_values = probabilities[eligible]
        y_values = label[eligible]
        for column, threshold in enumerate(thresholds):
            selected = p_values >= threshold
            if np.any(selected):
                selected_q = q_values[selected]
                pred_counts[:, column] = np.bincount(selected_q, minlength=query_count)
                selected_tp = selected & (y_values == 1)
                if np.any(selected_tp):
                    tp_counts[:, column] = np.bincount(q_values[selected_tp], minlength=query_count)
        by_source[source_id] = (pred_counts, tp_counts)

    pred2, tp2 = by_source[2]
    pred3, tp3 = by_source[3]
    best_score = -1.0
    best_t2 = best_t3 = 0.5
    for i, t2 in enumerate(thresholds):
        for j, t3 in enumerate(thresholds):
            score = _macro_f05_for_counts(tp2[:, i] + tp3[:, j], pred2[:, i] + pred3[:, j], actual)
            if score > best_score:
                best_score, best_t2, best_t3 = score, float(t2), float(t3)

    # Refine the coarse joint grid with exact coordinate sweeps over every
    # distinct model probability for each source. Each sweep updates only the
    # entities affected by a newly admitted edge, so it avoids a huge threshold
    # x entity x candidate tensor.
    def optimize_one(source_id: int, other_id: int, other_threshold: float) -> float:
        base = winner & (source == other_id) & (probabilities >= other_threshold)
        base_pred = np.bincount(query[base], minlength=query_count).astype(np.int32, copy=False)
        base_true = base & (label == 1)
        base_tp = np.bincount(query[base_true], minlength=query_count).astype(np.int32, copy=False)
        base_sum = float(_entity_scores(base_tp, base_pred, actual).sum())

        edges = np.flatnonzero(winner & (source == source_id))
        if not len(edges):
            return float(np.nextafter(1.0, np.inf))
        edge_q = query[edges]
        edge_p = probabilities[edges]
        edge_y = label[edges].astype(np.int32, copy=False)
        by_query = np.lexsort((-edge_p, edge_q))
        edge_q = edge_q[by_query]
        edge_p = edge_p[by_query]
        edge_y = edge_y[by_query]

        row = np.arange(len(edges), dtype=np.int64)
        starts_mask = np.empty(len(edges), dtype=bool)
        starts_mask[0] = True
        starts_mask[1:] = edge_q[1:] != edge_q[:-1]
        group_start = np.maximum.accumulate(np.where(starts_mask, row, 0))
        within_query_rank = row - group_start
        prefix_tp = np.cumsum(edge_y, dtype=np.int64)
        group_tp_before = np.zeros(len(edges), dtype=np.int64)
        starts = np.flatnonzero(starts_mask)
        later_starts = starts[1:]
        group_tp_before[later_starts] = prefix_tp[later_starts - 1]
        group_tp_before = np.maximum.accumulate(group_tp_before)
        prior_tp = prefix_tp - edge_y - group_tp_before

        before_pred = base_pred[edge_q] + within_query_rank
        before_tp = base_tp[edge_q] + prior_tp
        delta = (
            _entity_scores(before_tp + edge_y, before_pred + 1, actual[edge_q])
            - _entity_scores(before_tp, before_pred, actual[edge_q])
        )

        by_probability = np.argsort(-edge_p, kind="mergesort")
        ordered_probability = edge_p[by_probability]
        cumulative_delta = np.cumsum(delta[by_probability], dtype=np.float64)
        group_ends = np.r_[np.flatnonzero(ordered_probability[:-1] != ordered_probability[1:]), len(edges) - 1]
        score_at_end = base_sum + cumulative_delta[group_ends]
        best = base_sum
        best_threshold = float(np.nextafter(float(ordered_probability[0]), np.inf))
        improved = np.flatnonzero(score_at_end > best + 1e-12)
        if len(improved):
            best_position = improved[np.argmax(score_at_end[improved])]
            best = float(score_at_end[best_position])
            best_threshold = float(ordered_probability[group_ends[best_position]])
        return best_threshold

    converged = False
    for _ in range(12):
        next_t2 = optimize_one(2, 3, best_t3)
        next_t3 = optimize_one(3, 2, next_t2)
        if next_t2 == best_t2 and next_t3 == best_t3:
            converged = True
            break
        best_t2, best_t3 = next_t2, next_t3

    final2 = winner & (source == 2) & (probabilities >= best_t2)
    final3 = winner & (source == 3) & (probabilities >= best_t3)
    pred = np.bincount(query[final2], minlength=query_count) + np.bincount(query[final3], minlength=query_count)
    tp = np.bincount(query[final2 & (label == 1)], minlength=query_count) + np.bincount(
        query[final3 & (label == 1)], minlength=query_count
    )
    best_score = float(_entity_scores(tp, pred, actual).mean())
    return best_t2, best_t3, best_score, converged


def candidate_oracle(found_count: Sequence[int], true_count: Sequence[int]) -> float:
    found = np.asarray(found_count, dtype=np.float64)
    actual = np.asarray(true_count, dtype=np.float64)
    scores = np.ones_like(found)
    non_singletons = actual > 0
    scores[non_singletons] = np.divide(
        1.25 * found[non_singletons],
        found[non_singletons] + 0.25 * actual[non_singletons],
        out=np.zeros(np.count_nonzero(non_singletons), dtype=np.float64),
        where=(found[non_singletons] + 0.25 * actual[non_singletons]) > 0,
    )
    return float(scores.mean()) if len(scores) else 0.0


def append_binary(source: Path, target: Path) -> None:
    with open(source, "rb") as src, open(target, "ab") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("/kaggle/input/amazon-ml-challenge-2026/dataset/train"))
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/v8"))
    parser.add_argument("--max-candidates-per-source", type=int, default=128)
    parser.add_argument("--max-block-size", type=int, default=200)
    parser.add_argument("--max-address-block-size", type=int, default=1200)
    parser.add_argument("--hard-negatives-per-source", type=int, default=8)
    parser.add_argument("--holdout-per-mille", type=int, default=10, help="10 means a 1% entity-level holdout")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 2))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_candidates_per_source:
        raise SystemExit("--max-candidates-per-source must be positive")
    if not 1 <= args.holdout_per_mille <= 100:
        raise SystemExit("--holdout-per-mille must be between 1 and 100")
    if args.max_block_size < 2 or args.max_address_block_size < 2 or args.hard_negatives_per_source < 1:
        raise SystemExit("Block size must be >=2 and negative count must be >=1")
    train_dir = args.train_dir.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    truth_path = train_dir / "train_ground_truth.tsv"
    if not truth_path.exists():
        raise FileNotFoundError(f"Training labels not found: {truth_path}")
    print("Reading all ground-truth groups...", flush=True)
    truth = read_truth(truth_path)
    print(f"Loaded {len(truth):,} non-singleton Source-1 groups", flush=True)

    tables = build_pair_tables(
        train_dir,
        work_dir,
        truth,
        args.max_candidates_per_source,
        args.max_block_size,
        args.max_address_block_size,
        args.hard_negatives_per_source,
        args.holdout_per_mille,
        args.seed,
    )
    train_x, train_y, val_x, val_meta, train_rows, val_rows, val_ids, val_true, val_found = tables

    print("Fitting development LightGBM model...", flush=True)
    dev_model = train_booster(train_x, train_y, train_rows, args.rounds, args.seed, args.threads)
    val_prob = predict_memmap(dev_model, val_x, val_rows, work_dir)
    threshold2, threshold3, val_macro, threshold_converged = threshold_sweep(
        val_meta, val_prob, len(val_ids), val_true
    )
    recall = sum(val_found) / sum(val_true) if sum(val_true) else 1.0
    oracle = candidate_oracle(val_found, val_true)
    print(
        f"Validation: macro F0.5={val_macro:.6f}; oracle={oracle:.6f}; "
        f"pair candidate recall={recall:.6f}; thresholds S2={threshold2:.2f}, S3={threshold3:.2f}",
        flush=True,
    )

    # Refit with the held-out entity pairs appended. Thresholds remain those
    # measured on the disjoint holdout; all labelled entities now train the
    # shipped model. Feature files are binary memmaps to avoid Python row lists.
    del dev_model
    gc.collect()
    append_binary(val_x, train_x)
    # Validation labels are stored in the structured metadata file's final byte.
    meta = np.memmap(val_meta, dtype=META_DTYPE, mode="r", shape=(val_rows,))
    with open(train_y, "ab") as handle:
        meta["label"].tofile(handle)
    full_rows = train_rows + val_rows
    print(f"Refitting production model on all {full_rows:,} generated labelled pairs...", flush=True)
    final_model = train_booster(train_x, train_y, full_rows, args.rounds, args.seed, args.threads)

    model_path = work_dir / "entity_matcher_v7.txt"
    final_model.save_model(str(model_path))
    metadata = {
        "model": model_path.name,
        "feature_names": list(FEATURE_NAMES),
        "max_candidates_per_source": args.max_candidates_per_source,
        "max_block_size": args.max_block_size,
        "hard_negatives_per_source": args.hard_negatives_per_source,
        "threshold_source2": threshold2,
        "threshold_source3": threshold3,
        "threshold_coordinate_refinement_converged": threshold_converged,
        "target_exclusivity": "highest_probability_candidate_per_target_id",
        "validation_entities": len(val_ids),
        "validation_pair_candidate_recall": recall,
        "validation_candidate_oracle_macro_f05": oracle,
        "validation_macro_f05_after_threshold_and_exclusivity": val_macro,
        "lightgbm_rounds": args.rounds,
        "seed": args.seed,
        "note": "Candidate cap and thresholds must be revalidated if blocking settings or data version changes.",
    }
    metadata_path = work_dir / "entity_matcher_v7.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report_path = work_dir / "validation_report_v7.json"
    report_path.write_text(
        json.dumps(
            {
                "macro_f05": val_macro,
                "candidate_oracle_macro_f05": oracle,
                "pair_candidate_recall": recall,
                "validation_entities": len(val_ids),
                "threshold_source2": threshold2,
                "threshold_source3": threshold3,
                "threshold_coordinate_refinement_converged": threshold_converged,
                "singleton_entities": sum(count == 0 for count in val_true),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Model: {model_path}", flush=True)
    print(f"Metadata: {metadata_path}", flush=True)
    print(f"Validation report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
