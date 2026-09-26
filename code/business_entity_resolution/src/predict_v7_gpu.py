"""Generate V7 inference outputs with a trained CatBoost model."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from catboost import CatBoostClassifier

from pipeline_v7 import CandidateIndex, FEATURE_NAMES, make_record, pair_features


def read_rows(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


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


def generate_source_candidates(
    test_dir: Path,
    source: int,
    cap: int,
    block_cap: int,
    threshold: float,
    model: CatBoostClassifier,
    connection: sqlite3.Connection,
    candidate_path: Path,
) -> int:
    target_path = test_dir / f"test_source{source}.tsv"
    query_path = test_dir / "test_source1.tsv"
    index = CandidateIndex(max_posting=block_cap)
    print(f"Indexing test source {source}...", flush=True)
    started = time.time()
    for count, row in enumerate(read_rows(target_path), start=1):
        index.add(make_record(row["entity_id"], row["business_name"], row["business_address"], row["country"], source))
        if count % 500_000 == 0:
            print(f"  indexed {count:,} rows ({time.time() - started:.1f}s)", flush=True)
    print(f"  active blocks={index.block_count:,}; pruned blocks={index.pruned_count:,}", flush=True)

    edge_batch: List[Tuple[str, str, int, float]] = []
    candidate_count = 0
    with open(candidate_path, "w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        for count, row in enumerate(read_rows(query_path), start=1):
            query = make_record(row["entity_id"], row["business_name"], row["business_address"], row["country"], 1)
            candidates = index.query(query, cap=cap)
            target_ids: List[str] = []
            if candidates:
                feature_matrix = np.asarray(
                    [pair_features(query, index.records[c.record_index], c, index) for c in candidates],
                    dtype=np.float32,
                )
                probabilities = model.predict_proba(
                    feature_matrix, task_type="CPU", thread_count=1, verbose=False
                )[:, 1]
                for candidate, probability in zip(candidates, probabilities):
                    target = index.records[candidate.record_index]
                    target_ids.append(target.entity_id)
                    if float(probability) >= threshold:
                        edge_batch.append((target.entity_id, query.entity_id, count - 1, float(probability)))
                        if len(edge_batch) >= 20_000:
                            add_selected_edges(connection, edge_batch)
                candidate_count += len(candidates)
            writer.writerow((query.entity_id, ",".join(target_ids)))
            if count % 100_000 == 0:
                add_selected_edges(connection, edge_batch)
                connection.commit()
                print(f"  scored {count:,} Source-1 rows", flush=True)
    add_selected_edges(connection, edge_batch)
    connection.commit()
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
    parser.add_argument("--model-dir", type=Path, default=Path("/kaggle/working/v7_gpu"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/output_gpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_dir = args.test_dir.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((model_dir / "entity_matcher_v7_catboost.json").read_text(encoding="utf-8"))
    if metadata.get("feature_names") != list(FEATURE_NAMES):
        raise RuntimeError("The model feature schema does not match this inference code")
    model = CatBoostClassifier()
    model.load_model(str(model_dir / metadata["model"]))
    cap = int(metadata["max_candidates_per_source"])
    block_cap = int(metadata["max_block_size"])
    thresholds = {2: float(metadata["threshold_source2"]), 3: float(metadata["threshold_source3"])}

    with tempfile.TemporaryDirectory(prefix="v7_gpu_inference_", dir=output_dir) as temp_name:
        temp_dir = Path(temp_name)
        connection = sqlite3.connect(temp_dir / "selected_edges.sqlite")
        connection.execute(
            "CREATE TABLE selected_edges (target_id TEXT PRIMARY KEY, source1_id TEXT NOT NULL, query_index INTEGER NOT NULL, score REAL NOT NULL)"
        )
        candidate_files = {}
        total_candidates = 0
        for source in (2, 3):
            candidate_path = temp_dir / f"source{source}_candidates.tsv"
            candidate_files[source] = candidate_path
            total_candidates += generate_source_candidates(
                test_dir, source, cap, block_cap, thresholds[source], model, connection, candidate_path
            )
        write_final_outputs(test_dir, output_dir, candidate_files[2], candidate_files[3], connection)
        connection.close()
    print(f"Scored {total_candidates:,} capped candidates", flush=True)
    print(f"Wrote {output_dir / 'matching_results.tsv'}", flush=True)
    print(f"Wrote {output_dir / 'candidate_pairs.tsv'}", flush=True)


if __name__ == "__main__":
    main()
