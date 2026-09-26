# Kaggle hand-off: business entity resolution

## Goal and verified scoring rule

The task links each Source-1 reference record to zero or more records in Sources 2 and 3. The official metric is macro F0.5: calculate F0.5 for each Source-1 ID and average across all Source-1 IDs. An empty prediction for a true singleton scores 1; any non-empty prediction for a singleton scores 0. This is not the same as global pair-level (micro) F0.5.

`evaluator.py` was corrected to implement the entity-level metric, treat missing prediction rows as empty, and report micro metrics separately.

## Project findings

- Dataset sizes: train has 2,206,821 S1, 5,034,616 S2, 5,285,603 S3 rows and 7,638,365 labelled links. Test has 1,732,544 S1, 4,887,273 S2, and 5,082,316 S3 rows.
- The V6.1 ablation measured candidate-oracle macro F0.5 of about 0.978 on one 5,000-entity slice after relaxing prefiltering. It was not a fitted-model or leaderboard score. The unfiltered candidate volume grew to about 7.3M pairs on that slice, so full-scale unrestricted scoring is not practical.
- Older reported V4 scores mix in-sample and pair-level evaluation and should not be compared with the official metric. V4 remains available as the frozen fallback.
- A public solution repository self-reports validation macro F0.5 of 0.976105 (99.24% precision, 94.54% recall) and an ensemble ablation of 0.976653 using LightGBM + HistGradientBoosting. These are author-reported holdout results, not an independently verified leaderboard score. Another public design explores multi-view TF-IDF retrieval plus cross-fitted two-stage boosting, but states its full-data run expects at least 32 vCPU and 128 GB RAM; that makes it a useful architecture reference, not a direct Kaggle T4 recipe. See [the reported baseline and ablations](https://github.com/Akash-bardia/amazon-ml-challenge-2026) and [the larger-host pipeline design](https://github.com/AyanAhmedKhan/amazon-ml-challenge).
- Validation must be based on held-out Source-1 IDs. Never tune for the public leaderboard alone; the private split is what matters.

## Current implementation

V7 has been added without changing the frozen V4 inference script:

- `code/business_entity_resolution/src/pipeline_v7.py`: country-aware frequency-capped blocking; exact, compact, core-token, rare-token, prefix, name/number, address/number and transliteration channels; cheap candidate ranking; default hard cap of 64 candidates per target source; 60 numeric pair features including TF-IDF cosine/Jaccard, weighted overlap, character 5-gram similarity, source and blocking provenance.
- `code/business_entity_resolution/src/train_v7.py`: processes the full training S1/S2/S3 files, uses a deterministic 1% S1 holdout, keeps all retrieved positive pairs and samples up to 8 hard negative pairs per source/query, selects a coarse joint S2/S3 threshold pair then refines each source threshold over the observed probabilities after target exclusivity, and refits LightGBM using all generated labelled pairs.
- `code/business_entity_resolution/src/predict_v7.py`: applies the saved cap and thresholds to test, enforces target-record exclusivity, and writes both submission TSVs. `candidate_pairs.tsv` contains the exact final capped candidate set scored by the model.
- `README_KAGGLE.md` and `code/business_entity_resolution/README.md`: Kaggle commands, input layout, runtime notes, and validation instructions.
- `.gitignore`: excludes challenge inputs, archives, local feature binaries, models, and outputs from Git.

The code is implemented but has not been run on Kaggle or benchmarked on the full dataset. The default cap, candidate recall, runtime, memory use, LightGBM fit, threshold sweep, and final score remain unverified. The requested 0.98–0.99 is a goal, not a guarantee. The decisive first gate is whether the capped candidate-oracle macro F0.5 remains sufficiently high on a representative held-out slice.

## V8 high-recall GPU notebook (new run)

`Kaggle_Entity_Resolution_V8_Best_GPU.ipynb` is the recommended notebook for a new Kaggle session. It embeds its scripts and validator, discovers an attached official dataset under `/kaggle/input/`, and does not need GitHub credentials. It uses all Source-2/3 records for blocking, processes all Source-1 training entities, holds out a deterministic 1% of Source-1 IDs for model/threshold selection, and adds that holdout back for production fitting. Easy negative *pairs* are sampled; positive candidate pairs are retained.

V8 adds address-only and transliterated-address-only blocking provenance because the V6.1 error audit showed name-damaged matches could still share a rare address term. It sets independent posting caps for name-like blocks and address blocks. It compares multi-GPU CatBoost, CPU LightGBM, and probability blends on the same held-out entities; the selected blend and source thresholds are then refit on every training entity using all retrieved positives and the same hard-negative sampling rule. Pair/feature generation writes durable checkpoints every 50,000 Source-1 rows per target source, and completed pair tables are fingerprinted and cached. Rerunning the training cell after an interruption rebuilds the active target index but resumes query processing at the last committed batch. CatBoost writes GPU training snapshots every five minutes. Test inference also checkpoints candidate output and selected edges, batches model calls, and resumes from the last completed query batch. The notebook probes numeric-only CatBoost GPU prediction and uses it when available, otherwise it falls back to CPU; blocking and feature extraction remain CPU work. The notebook avoids retraining if matching model/report artifacts are already present.

Checkpoint files live under `/kaggle/working/v8_best_gpu` and inference checkpoints under `/kaggle/working/output_v8_best`. Resume is available when Kaggle preserves that working directory after a kernel interruption; a completely new session/account should be treated as a clean run unless those notebook outputs were explicitly saved and attached.

Treat the V6.1 `0.978` candidate oracle as a preliminary diagnostic only: the experiment selected its 5,000 validation rows from a 25,000-row prefix before shuffling, and its score was an oracle (perfect filtering of candidates), not a trained model score. V8 reports a stable hash holdout candidate oracle and post-threshold macro F0.5 separately. If the candidate oracle is below 0.99, no classifier can reach 0.99 under that candidate cap; blocking must improve first. The V8 candidate cap (128 per target source), validation score, inference runtime, peak RAM, and leaderboard score have not been measured yet and must not be represented as achieved.

## Kaggle workflow

```bash
pip install -r code/business_entity_resolution/requirements.txt

python code/business_entity_resolution/src/train_v7.py \
  --train-dir /kaggle/input/<dataset-slug>/dataset/train \
  --work-dir /kaggle/working/v7

python code/business_entity_resolution/src/predict_v7.py \
  --test-dir /kaggle/input/<dataset-slug>/dataset/test \
  --model-dir /kaggle/working/v7 \
  --output-dir /kaggle/working/output

python utils/validate_submission.py \
  --matching /kaggle/working/output/matching_results.tsv \
  --candidate /kaggle/working/output/candidate_pairs.tsv \
  --test-dir /kaggle/input/<dataset-slug>/dataset/test
```

Replace `<dataset-slug>` with the mounted Kaggle input folder. Leave enough free working disk for raw float32 pair features; the amount scales with the number of retained rows. GPU is not required for LightGBM or blocking. Kaggle notebook CPU, RAM, disk, and runtime limits must be checked in the actual session before committing to a full run.

## Research and model strategy

LightGBM is the first model because the pairwise task produces numeric similarities and channel indicators at high volume. Character/name transliteration is deterministic and uses only supplied values. Keep pretrained LLM/cross-encoder models as an optional small top-candidate reranking experiment: they add inference cost and licensing/data-provenance questions, and published F1 on unrelated entity-matching datasets does not predict this challenge's macro F0.5.

Useful primary sources:

- LightGBM efficiency and histogram GBDT: <https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree>
- Current LightGBM Python API: <https://lightgbm.readthedocs.io/en/stable/Python-API.html>
- Listwise/comparison strategies for entity matching (COLING 2025): <https://aclanthology.org/2025.coling-main.8/>
- Ditto pairwise pretrained language-model matcher (PVLDB): <https://www.vldb.org/pvldb/vol14/p50-li.pdf>
- Kaggle competition setup and leaderboard guidance: <https://www.kaggle.com/docs/competitions-setup>
