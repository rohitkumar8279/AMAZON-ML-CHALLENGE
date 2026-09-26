# Business Entity Resolution

## Pipelines

- `src/inference_v4.py` and `AmazonML_PreSubmission_V4_REAL.zip` are the preserved V4 baseline.
- `src/train_v7.py`, `src/predict_v7.py`, and `src/pipeline_v7.py` are the new Kaggle workflow. V7 is separate from V4 while it is being benchmarked.

V7 uses LightGBM on 60 pair features. Candidate generation unions frequency-capped exact, token, address-number, prefix, and transliteration blocks, ranks them cheaply, and caps the model input at 64 candidates per target source by default. Each source has a validation-tuned threshold, and inference assigns each target record to at most one Source-1 entity.

The 0.98–0.99 score goal is not a measured result. The V6.1 0.978 figure is a candidate-only oracle from one validation slice. V7 has to demonstrate candidate recall, the candidate oracle, and held-out macro F0.5 before any score claim is made.

## Kaggle run

Install the pinned runtime:

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

Train against all supplied training files:

```bash
python code/business_entity_resolution/src/train_v7.py \
  --train-dir /kaggle/input/<dataset-slug>/dataset/train \
  --work-dir /kaggle/working/v7
```

Run inference:

```bash
python code/business_entity_resolution/src/predict_v7.py \
  --test-dir /kaggle/input/<dataset-slug>/dataset/test \
  --model-dir /kaggle/working/v7 \
  --output-dir /kaggle/working/output
```

V7 processes every Source-1 entity and every Source-2/Source-3 target record. It downsamples only training negative pairs. The holdout is selected by a stable hash (1% by default), used to optimize the official entity-level macro F0.5 thresholds, and then added back for the final all-data fit. The generated raw feature matrices need free `/kaggle/working` disk space; size depends on the number of retained pairs.

Run the challenge validator before packaging:

```bash
python utils/validate_submission.py \
  --matching /kaggle/working/output/matching_results.tsv \
  --candidate /kaggle/working/output/candidate_pairs.tsv \
  --test-dir /kaggle/input/<dataset-slug>/dataset/test
```

The candidate file contains exactly the pairs passed to LightGBM, before thresholding and target exclusivity. Keep datasets, models, binary features, and generated outputs out of source control; see the repository `.gitignore`.
