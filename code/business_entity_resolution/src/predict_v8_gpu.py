"""Batched V8 test inference; score the complete configured cap, not 1 row/call."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from catboost import CatBoostClassifier
import lightgbm as lgb

from pipeline_v8 import CandidateIndex, FEATURE_NAMES, make_record, pair_features


def read_rows(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def _file_signature(path: Path) -> dict:
    size = path.stat().st_size
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read(65_536))
        handle.seek(max(0, size - 65_536))
        digest.update(handle.read(65_536))
    return {"size": size, "edge_sha256": digest.hexdigest()}


def _inference_fingerprint(test_dir: Path, model_dir: Path, metadata_path: Path, metadata: dict) -> str:
    signatures = {
        "test": {
            name: _file_signature(test_dir / name)
            for name in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv")
        },
        "models": {
            name: _file_signature(model_dir / filename)
            for name, filename in metadata["model_files"].items()
        },
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "pipeline_sha256": hashlib.sha256(Path(__file__).with_name("pipeline_v8.py").read_bytes()).hexdigest(),
        "predictor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(signatures, sort_keys=True).encode("utf-8")).hexdigest()


def _write_resume_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def add_selected_edges(connection: sqlite3.Connection, edges: List[Tuple[str, str, int, float]]) -> None:
    if not edges:
        return
    connection.executemany(
        """
        INSERT INTO selected_edges(target_id, source1_id, query_index, score) VALUES (?, ?, ?, ?)
        ON CONFLICT(target_id) DO UPDATE SET source1_id=excluded.source1_id,
          query_index=excluded.query_index, score=excluded.score
        WHERE excluded.score > selected_edges.score
        """,
        edges,
    )
    edges.clear()


def score_pair_batch(features, cat_model, lgb_model, cat_weight: float, threads: int) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    probability = None
    if cat_weight > 0:
        cat_probability = cat_model.predict_proba(
            matrix, thread_count=threads, verbose=False
        )[:, 1]
        probability = cat_weight * cat_probability
    if cat_weight < 1:
        lgb_probability = lgb_model.predict(matrix, num_threads=threads)
        contribution = (1.0 - cat_weight) * lgb_probability
        probability = contribution if probability is None else probability + contribution
    return np.asarray(probability, dtype=np.float32)


def generate_source_candidates(
    test_dir: Path,
    source: int,
    cap: int,
    block_cap: int,
    address_block_cap: int,
    threshold: float,
    cat_model,
    lgb_model,
    cat_weight: float,
    connection: sqlite3.Connection,
    candidate_path: Path,
    query_batch_size: int,
    prediction_threads: int,
    resume_state: dict,
    state_path: Path,
) -> int:
    target_path = test_dir / f"test_source{source}.tsv"
    query_path = test_dir / "test_source1.tsv"
    index = CandidateIndex(max_posting=block_cap, max_address_posting=address_block_cap)
    print(f"Indexing test source {source}...", flush=True)
    started = time.time()
    for count, row in enumerate(read_rows(target_path), start=1):
        index.add(make_record(row["entity_id"], row["business_name"], row["business_address"], row["country"], source))
        if count % 500_000 == 0:
            print(f"  indexed {count:,} rows ({time.time() - started:.1f}s)", flush=True)
    print(
        f"  active blocks={index.block_count:,}; pruned blocks={index.pruned_count:,}; "
        f"address block cap={address_block_cap:,}", flush=True,
    )

    edge_batch: List[Tuple[str, str, int, float]] = []
    feature_rows = []
    candidate_ids: List[str] = []
    query_ranges: List[Tuple[int, int, int]] = []
    source_key = str(source)
    resume_rows = int(resume_state["rows_processed"].get(source_key, 0))
    resume_bytes = int(resume_state["candidate_bytes"].get(source_key, 0))
    candidate_count = int(resume_state["candidate_counts"].get(source_key, 0))
    if candidate_path.exists() and candidate_path.stat().st_size < resume_bytes:
        raise RuntimeError(f"Resume candidate file is shorter than its checkpoint: {candidate_path}")
    if resume_rows and not candidate_path.is_file():
        raise RuntimeError(f"Resume checkpoint has {resume_rows:,} rows but no candidate file: {candidate_path}")
    query_started = time.time()

    current_query_ids: List[str] = []
    batch_start = 0
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "r+" if candidate_path.exists() else "w+"
    processed_count = resume_rows
    with open(candidate_path, output_mode, encoding="utf-8", newline="") as output:
        output.seek(0)
        output.truncate(resume_bytes)
        output.seek(resume_bytes)
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")

        def flush_batch(last_processed_row: int):
            nonlocal feature_rows, candidate_ids, query_ranges
            if feature_rows:
                probabilities = score_pair_batch(
                    feature_rows, cat_model, lgb_model, cat_weight, prediction_threads
                )
                for query_index, start, end in query_ranges:
                    for target_id, score in zip(candidate_ids[start:end], probabilities[start:end]):
                        score = float(score)
                        if score >= threshold:
                            edge_batch.append((target_id, current_query_ids[query_index - batch_start], query_index, score))
                add_selected_edges(connection, edge_batch)
                feature_rows = []
                candidate_ids = []
                query_ranges = []
            output.flush()
            connection.commit()
            resume_state["rows_processed"][source_key] = last_processed_row
            resume_state["candidate_bytes"][source_key] = candidate_path.stat().st_size
            resume_state["candidate_counts"][source_key] = candidate_count
            _write_resume_state(state_path, resume_state)

        for count, row in enumerate(read_rows(query_path), start=1):
            if count <= resume_rows:
                continue
            if not current_query_ids:
                batch_start = count - 1
            query = make_record(row["entity_id"], row["business_name"], row["business_address"], row["country"], 1)
            candidates = index.query(query, cap=cap)
            ids = [index.records[c.record_index].entity_id for c in candidates]
            writer.writerow((query.entity_id, ",".join(ids)))
            current_query_ids.append(query.entity_id)
            candidate_count += len(candidates)
            start = len(candidate_ids)
            for candidate in candidates:
                target = index.records[candidate.record_index]
                candidate_ids.append(target.entity_id)
                feature_rows.append(pair_features(query, target, candidate, index))
            if candidates:
                query_ranges.append((count - 1, start, len(candidate_ids)))
            processed_count = count

            if len(current_query_ids) >= query_batch_size:
                flush_batch(processed_count)
                current_query_ids = []
            if count % 100_000 == 0:
                elapsed = max(time.time() - query_started, 1e-9)
                print(
                    f"  built candidate features for {count:,} Source-1 rows at "
                    f"{count / elapsed:,.0f} rows/s; candidates={candidate_count:,}; "
                    f"mean={candidate_count / count:.1f}/query; elapsed={elapsed / 60:.1f} min",
                    flush=True,
                )
        flush_batch(processed_count)
        add_selected_edges(connection, edge_batch)
        connection.commit()
        output.flush()
    resume_state["rows_processed"][source_key] = processed_count
    resume_state["candidate_bytes"][source_key] = candidate_path.stat().st_size
    resume_state["candidate_counts"][source_key] = candidate_count
    resume_state["completed_sources"] = sorted(set(resume_state["completed_sources"] + [source]))
    _write_resume_state(state_path, resume_state)
    return candidate_count


def write_final_outputs(
    test_dir: Path,
    output_dir: Path,
    candidate_s2: Path,
    candidate_s3: Path,
    connection: sqlite3.Connection,
) -> None:
    candidate_out = output_dir / "candidate_pairs.tsv"
    matching_out = output_dir / "matching_results.tsv"
    with open(candidate_s2, "r", encoding="utf-8", newline="") as left, open(
        candidate_s3, "r", encoding="utf-8", newline=""
    ) as right, open(candidate_out, "w", encoding="utf-8", newline="") as output:
        left_reader = csv.reader(left, delimiter="\t")
        right_reader = csv.reader(right, delimiter="\t")
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(("source1_entity_id", "candidate_entity_ids"))
        for row2, row3 in zip(left_reader, right_reader):
            if not row2 or not row3 or row2[0] != row3[0]:
                raise RuntimeError("Source-specific candidate rows are misaligned")
            candidates = [value for value in (row2[1] if len(row2) > 1 else "", row3[1] if len(row3) > 1 else "") if value]
            writer.writerow((row2[0], ",".join(candidates)))
        if next(left_reader, None) is not None or next(right_reader, None) is not None:
            raise RuntimeError("Source-specific candidate files have different row counts")

    connection.execute("CREATE INDEX IF NOT EXISTS selected_edges_source1 ON selected_edges(source1_id)")
    cursor = iter(connection.execute(
        "SELECT query_index, source1_id, target_id FROM selected_edges ORDER BY query_index, target_id"
    ))
    next_edge = next(cursor, None)
    with open(matching_out, "w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(("source1_entity_id", "matched_entity_ids"))
        for query_index, row in enumerate(read_rows(test_dir / "test_source1.tsv")):
            entity_id = row["entity_id"]
            matches: List[str] = []
            while next_edge is not None and next_edge[0] == query_index:
                if next_edge[1] != entity_id:
                    raise RuntimeError(
                        f"Prediction row ID mismatch at row {query_index}: {next_edge[1]!r} != {entity_id!r}"
                    )
                matches.append(next_edge[2])
                next_edge = next(cursor, None)
            if next_edge is not None and next_edge[0] < query_index:
                raise RuntimeError(f"Prediction references stale Source-1 row index {next_edge[0]}")
            writer.writerow((entity_id, ",".join(matches)))
    if next_edge is not None:
        raise RuntimeError(f"Prediction references unknown Source-1 row index {next_edge[0]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/kaggle/working/v8_gpu"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/output_v8"))
    parser.add_argument("--query-batch-size", type=int, default=1024)
    parser.add_argument("--prediction-threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.query_batch_size < 1 or args.prediction_threads < 1:
        raise SystemExit("Batch size and prediction thread count must be positive")
    test_dir = args.test_dir.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = model_dir / "entity_matcher_v8.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_names") != list(FEATURE_NAMES):
        raise RuntimeError("The model feature schema does not match this inference code")
    fingerprint = _inference_fingerprint(test_dir, model_dir, metadata_path, metadata)
    output_manifest_path = output_dir / "inference_v8_manifest.json"
    matching_out = output_dir / "matching_results.tsv"
    candidate_out = output_dir / "candidate_pairs.tsv"
    if output_manifest_path.is_file() and matching_out.is_file() and candidate_out.is_file():
        try:
            previous_output = json.loads(output_manifest_path.read_text(encoding="utf-8"))
            signatures = previous_output.get("output_signatures", {})
            outputs_intact = (
                signatures.get("matching_results.tsv") == _file_signature(matching_out)
                and signatures.get("candidate_pairs.tsv") == _file_signature(candidate_out)
            )
            if previous_output.get("fingerprint") == fingerprint and outputs_intact:
                print("Complete matching V8 inference outputs already exist; reusing them.", flush=True)
                print(f"Wrote {matching_out}", flush=True)
                print(f"Wrote {candidate_out}", flush=True)
                return
        except (OSError, json.JSONDecodeError):
            pass

    resume_dir = output_dir / f".v8_resume_{fingerprint[:16]}"
    resume_dir.mkdir(parents=True, exist_ok=True)
    state_path = resume_dir / "inference_state.json"
    if state_path.is_file():
        try:
            resume_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Inference checkpoint is unreadable: {state_path}; preserve it and inspect the error") from exc
        if resume_state.get("fingerprint") != fingerprint:
            raise RuntimeError("Inference checkpoint fingerprint does not match the current models/test files")
    else:
        resume_state = {
            "fingerprint": fingerprint,
            "completed_sources": [],
            "rows_processed": {"2": 0, "3": 0},
            "candidate_bytes": {"2": 0, "3": 0},
            "candidate_counts": {"2": 0, "3": 0},
        }
        _write_resume_state(state_path, resume_state)

    database_path = resume_dir / "selected_edges.sqlite"
    if not database_path.exists() and (
        resume_state["completed_sources"] or any(resume_state["rows_processed"].values())
    ):
        print("SQLite edge checkpoint is missing; restarting test scoring from the saved model.", flush=True)
        resume_state.update({
            "completed_sources": [],
            "rows_processed": {"2": 0, "3": 0},
            "candidate_bytes": {"2": 0, "3": 0},
            "candidate_counts": {"2": 0, "3": 0},
        })
        _write_resume_state(state_path, resume_state)

    cat_weight = float(metadata["catboost_weight"])
    cat_model = None
    lgb_model = None
    if cat_weight > 0:
        cat_model = CatBoostClassifier()
        cat_model.load_model(str(model_dir / metadata["model_files"]["catboost"]))
    if cat_weight < 1:
        lgb_model = lgb.Booster(model_file=str(model_dir / metadata["model_files"]["lightgbm"]))
    cap = int(metadata["max_candidates_per_source"])
    block_cap = int(metadata["max_block_size"])
    address_block_cap = int(metadata["max_address_block_size"])
    thresholds = {2: float(metadata["threshold_source2"]), 3: float(metadata["threshold_source3"])}

    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS selected_edges (target_id TEXT PRIMARY KEY, source1_id TEXT NOT NULL, query_index INTEGER NOT NULL, score REAL NOT NULL)"
    )
    candidate_files = {source: resume_dir / f"source{source}_candidates.tsv" for source in (2, 3)}
    # Completed sources are already represented in the accumulator. A partial
    # source returns its full checkpointed + newly scored count from the helper.
    total_candidates = sum(
        int(resume_state["candidate_counts"].get(str(source), 0))
        for source in resume_state["completed_sources"]
    )
    for source in (2, 3):
        candidate_path = candidate_files[source]
        if source in resume_state["completed_sources"]:
            if not candidate_path.is_file():
                raise RuntimeError(f"Checkpoint marks Source {source} complete but its candidate file is missing")
            if candidate_path.stat().st_size != int(resume_state["candidate_bytes"][str(source)]):
                raise RuntimeError(f"Completed Source {source} candidate file size differs from checkpoint")
            print(f"Source {source} already complete; skipping its index and scoring pass.", flush=True)
            continue
        total_candidates += generate_source_candidates(
            test_dir, source, cap, block_cap, address_block_cap, thresholds[source],
            cat_model, lgb_model, cat_weight, connection, candidate_path,
            args.query_batch_size, args.prediction_threads, resume_state, state_path,
        )
    write_final_outputs(test_dir, output_dir, candidate_files[2], candidate_files[3], connection)
    connection.close()
    output_manifest = {
        "fingerprint": fingerprint,
        "candidate_count": total_candidates,
        "max_candidates_per_source": cap,
        "threshold_source2": thresholds[2],
        "threshold_source3": thresholds[3],
        "output_signatures": {
            "matching_results.tsv": _file_signature(matching_out),
            "candidate_pairs.tsv": _file_signature(candidate_out),
        },
    }
    manifest_tmp = output_manifest_path.with_suffix(".tmp")
    manifest_tmp.write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")
    manifest_tmp.replace(output_manifest_path)
    resolved_output = output_dir.resolve()
    resolved_resume = resume_dir.resolve()
    if resolved_resume.parent != resolved_output or not resume_dir.name.startswith(".v8_resume_"):
        raise RuntimeError(f"Refusing to remove unexpected inference workspace: {resolved_resume}")
    shutil.rmtree(resolved_resume)
    print(f"Scored {total_candidates:,} capped candidates in batches of {args.query_batch_size:,}", flush=True)
    print(f"Wrote {output_dir / 'matching_results.tsv'}", flush=True)
    print(f"Wrote {output_dir / 'candidate_pairs.tsv'}", flush=True)


if __name__ == "__main__":
    main()
