# Kaggle hand-off: business entity resolution

## Goal and verified scoring rule

The task links each Source-1 reference record to zero or more records in Sources 2 and 3. The official metric is macro F0.5: calculate F0.5 for each Source-1 ID and average across all Source-1 IDs. An empty prediction for a true singleton scores 1; any non-empty prediction for a singleton scores 0. This is not the same as global pair-level (micro) F0.5.

`evaluator.py` was corrected to implement the entity-level metric, treat missing prediction rows as empty, and report micro metrics separately.

## Project findings

- Dataset sizes: train has 2,206,821 S1, 5,034,616 S2, 5,285,603 S3 rows and 7,638,365 labelled links. Test has 1,732,544 S1, 4,887,273 S2, and 5,082,316 S3 rows.
- The V6.1 ablation measured candidate-oracle macro F0.5 of about 0.978 on one 5,000-entity slice after relaxing prefiltering. It was not a fitted-model or leaderboard score. The unfiltered candidate volume grew to about 7.3M pairs on that slice, so full-scale unrestricted scoring is not practical.
- Older reported V4 scores mix in-sample and pair-level evaluation and should not be compared with the official metric. V4 remains available as the frozen fallback.
- A public GitHub README reports 0.976 macro F0.5 for this task and describes multi-key blocks, source thresholds, and target exclusivity. The claim is self-reported and unverified; use those methods as ideas to evaluate, not as evidence that the score is achieved.
- Validation must be based on held-out Source-1 IDs. Never tune for the public leaderboard alone; the private split is what matters.

## Current implementation

V7 has been added without changing the frozen V4 inference script:

- `code/business_entity_resolution/src/pipeline_v7.py`: country-aware frequency-capped blocking; exact, compact, core-token, rare-token, prefix, name/number, address/number and transliteration channels; cheap candidate ranking; default hard cap of 64 candidates per target source; 60 numeric pair features including TF-IDF cosine/Jaccard, weighted overlap, character 5-gram similarity, source and blocking provenance.
- `code/business_entity_resolution/src/train_v7.py`: processes the full training S1/S2/S3 files, uses a deterministic 1% S1 holdout, keeps all retrieved positive pairs and samples up to 8 hard negative pairs per source/query, selects a coarse joint S2/S3 threshold pair then refines each source threshold over the observed probabilities after target exclusivity, and refits LightGBM using all generated labelled pairs.
- `code/business_entity_resolution/src/predict_v7.py`: applies the saved cap and thresholds to test, enforces target-record exclusivity, and writes both submission TSVs. `candidate_pairs.tsv` contains the exact final capped candidate set scored by the model.
- `README_KAGGLE.md` and `code/business_entity_resolution/README.md`: Kaggle commands, input layout, runtime notes, and validation instructions.
- `.gitignore`: excludes challenge inputs, archives, local feature binaries, models, and outputs from Git.

The code is implemented but has not been run on Kaggle or benchmarked on the full dataset. The default cap, candidate recall, runtime, memory use, LightGBM fit, threshold sweep, and final score remain unverified. The requested 0.98–0.99 is a goal, not a guarantee. The decisive first gate is whether the capped candidate-oracle macro F0.5 remains sufficiently high on a representative held-out slice.

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
