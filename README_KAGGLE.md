# Kaggle training and inference (V7)

For an importable end-to-end Kaggle notebook, use [Kaggle_Entity_Resolution_V7.ipynb](Kaggle_Entity_Resolution_V7.ipynb). Attach the official challenge data through Kaggle's **Add Input** panel before running it.

V7 is a new, separate pipeline. The V4 submission remains untouched as the fallback. V7 uses the full labelled challenge files: every Source-1 row participates in candidate generation, a stable 1% Source-1 holdout is used for threshold selection, and the production LightGBM model is then refit using the holdout too. Only easy negative *pairs* are downsampled; the training entities and ground-truth groups are not sampled away.

## What V7 does

1. Normalizes names and addresses, and makes transliterated variants for cross-script retrieval.
2. Builds frequency-capped inverted indexes over both target sources. Blocks cover exact and compact names, core name tokens, rare name tokens, prefixes, name plus street number, address tokens plus number, and transliterated text.
3. Ranks each entity's candidates with a cheap token/rarity/provenance score, then keeps at most `--max-candidates-per-source` per target source. The default is 64 per source (at most 128 pairs per Source-1 entity).
4. Computes 60 pair features: TF-IDF-weighted Jaccard and cosine, token overlap, RapidFuzz string similarities, character n-grams, address-number agreement/conflict, source indicators, all blocking provenance bits, and retrieval rank.
5. Trains LightGBM on every training entity's retrieved positives and up to 8 hard negatives per target source. It scores the complete capped candidate lists for the holdout, finds a coarse joint threshold pair and refines each source threshold over observed probabilities against the official entity-level macro F0.5 after target exclusivity.
6. Refits the shipped model from all generated training pairs and writes a model, metadata, and validation report.

The candidate cap is a compute/recall trade-off, not a score guarantee. First inspect `validation_report_v7.json`: if candidate recall or the candidate-oracle macro F0.5 is below the target, a classifier cannot recover those missing links. The V6.1 result of 0.978 was an oracle ceiling on one validation slice, not a LightGBM or leaderboard score. A 0.98–0.99 result remains an experimental goal until a full held-out run supports it.

## Kaggle setup

Add the supplied challenge archive as a Kaggle Dataset, preserving its `dataset/train/` and `dataset/test/` directories. Clone this repository in a Kaggle Notebook or add the repository files as a notebook input. At the defaults, raw feature matrices can approach roughly 11 GB before model bins and runtime overhead; leave at least 15 GB free in `/kaggle/working` and check RAM in the actual Kaggle session. LightGBM and blocking are CPU workloads; a GPU is not required.

Install dependencies:

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

Train on the complete training files (replace `<dataset-slug>` with the Kaggle input directory name):

```bash
python code/business_entity_resolution/src/train_v7.py \
  --train-dir /kaggle/input/<dataset-slug>/dataset/train \
  --work-dir /kaggle/working/v7
```

The defaults are 64 candidates per target source, 200 records per blocking key, 8 retained hard negatives per source, a 1% validation holdout, and 500 boosting rounds. Override them on the command line when comparing settings; for example:

```bash
python code/business_entity_resolution/src/train_v7.py \
  --train-dir /kaggle/input/<dataset-slug>/dataset/train \
  --work-dir /kaggle/working/v7 \
  --max-candidates-per-source 128 \
  --hard-negatives-per-source 8
```

Use the same held-out IDs when comparing blocking settings. A smaller cap is faster but can reduce the oracle ceiling. A larger cap increases feature-table disk and training costs. Keep the holdout at its natural candidate prevalence; it is not negative-downsampled. Do not select settings from the public leaderboard alone.

Run test inference using the fitted model and its saved thresholds:

```bash
python code/business_entity_resolution/src/predict_v7.py \
  --test-dir /kaggle/input/<dataset-slug>/dataset/test \
  --model-dir /kaggle/working/v7 \
  --output-dir /kaggle/working/output
```

This writes `matching_results.tsv` and `candidate_pairs.tsv`. Candidate output is the exact capped set passed to the model. The decoder keeps the highest-scoring reference for each target ID and permits multiple matches per Source-1 entity.

Validate both artifacts with the supplied challenge validator:

```bash
python utils/validate_submission.py \
  --matching /kaggle/working/output/matching_results.tsv \
  --candidate /kaggle/working/output/candidate_pairs.tsv \
  --test-dir /kaggle/input/<dataset-slug>/dataset/test
```

For a labelled validation prediction file, use the corrected root evaluator:

```bash
python evaluator.py predictions.tsv /kaggle/input/<dataset-slug>/dataset/train/train_ground_truth.tsv
```

It reports the official macro F0.5 separately from global pair-level micro metrics. The official score averages the per-entity scores over all Source-1 entities; a correct empty prediction for a singleton scores 1, while an incorrect non-empty prediction scores 0.

## Model choices and next experiments

LightGBM is the primary model because the task has rich numeric pair features, high pair counts, and a precision-heavy thresholded decision. The original LightGBM paper describes histogram-based Gradient Boosting Decision Trees with techniques designed for efficiency and scale; its current Python API accepts array-backed data, which this implementation uses to avoid Python object lists for pair features. [LightGBM paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree), [LightGBM Python API](https://lightgbm.readthedocs.io/en/stable/Python-API.html).

The strongest next model experiment is a capped listwise or comparison reranker over the top few candidates *after* the LightGBM pass, measured only if the 60 numeric features leave systematic errors. Recent entity-matching work reports benefits from comparing/selecting candidates with record interactions, but it studies different datasets and metrics; it does not establish a gain on this challenge. [COLING 2025 ComEM study](https://aclanthology.org/2025.coling-main.8/). Ditto shows that pretrained transformer cross-encoders can improve pair classification on established entity-matching benchmarks, including one company dataset, but its reported F1 is not comparable to this challenge's macro F0.5, and large-scale scoring of tens of millions of pairs makes it a later reranking experiment rather than the first production path. [Ditto paper](https://www.vldb.org/pvldb/vol14/p50-li.pdf).

An online repository claims 0.976 macro F0.5 for this challenge and describes LightGBM, multi-key blocks, source thresholds, and target exclusivity. Its README is self-reported, the visible page shows two commits and no independent score evidence, and the claimed code/results have not been reproduced here; those design ideas are hypotheses, not benchmark facts. [Public repository and claim](https://github.com/Akash-bardia/amazon-ml-challenge-2026).

The public/private leaderboard split is a reason to keep model and threshold selection anchored to a representative local holdout. [Kaggle competition documentation](https://www.kaggle.com/docs/competitions-setup).

## Fair-play and resource notes

- V7 uses only the supplied records and labels. It makes no API calls and adds no external business data.
- The transliteration library is a deterministic text transform, not a source of business facts.
- The model is LightGBM under an open-source license. The shipped inference code is self-contained apart from the pinned requirements.
- Generated model weights, feature binaries, Kaggle inputs, output predictions, and the multi-gigabyte source archive should stay out of Git.
