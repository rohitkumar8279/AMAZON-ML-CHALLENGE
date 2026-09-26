import sys
import os
import csv
import random
import time
from collections import Counter, defaultdict

# Add pre_submission_v4_real_frozen to path to import its modules
sys.path.append(os.path.abspath('experiments/pre_submission_v4_real_frozen'))
from inference_v4 import (
    BlockIndex, compute_features_v4, fast_prefilter, SGDLogistic,
    build_idf_vocab, norm_name, addr_nums, norm_addr,
    THRESHOLD, SGD_LR, SGD_EPOCHS, N_FEATURES, CHAR4_MAX_BLOCK
)

print("Phase 2: Cardinality of current V4", flush=True)

random.seed(42)

TRAIN_S1 = 'dataset/train/train_source1.tsv'
TRAIN_S2 = 'dataset/train/train_source2.tsv'
TRAIN_S3 = 'dataset/train/train_source3.tsv'
TRAIN_GT = 'dataset/train/train_ground_truth.tsv'

print("Loading ground truth...", flush=True)
gt = {}
with open(TRAIN_GT, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        m = row['matched_entity_ids']
        if m:
            gt[row['source1_entity_id']] = set(m.split(','))

# Read all S1 entities to split
print("Reading S1 entities for 80/20 split...", flush=True)
s1_rows = []
with open(TRAIN_S1, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        s1_rows.append(row)

# Split 80/20
random.shuffle(s1_rows)
split_idx = int(len(s1_rows) * 0.8)
train_s1 = s1_rows[:split_idx]
val_s1 = s1_rows[split_idx:]
print(f"Train S1: {len(train_s1)}, Val S1: {len(val_s1)}", flush=True)

# 1. Build IDF vocab on train S1 only (or all, V4 used first 200k)
# For fairness to V4, we'll build it from train_s1
print("Building IDF vocab on train split...", flush=True)
name_df = Counter()
addr_df = Counter()
for row in train_s1:
    for t in set(norm_name(row['business_name']).split()):
        name_df[t] += 1
    for t in set(norm_addr(row['business_address']).split()):
        addr_df[t] += 1
name_idf = {k: __import__('math').log((len(train_s1) + 1) / (v + 1)) + 1 for k, v in name_df.items()}
addr_idf = {k: __import__('math').log((len(train_s1) + 1) / (v + 1)) + 1 for k, v in addr_df.items()}

# 2. Block and Index S2 and S3 for training
print("Indexing S2 and S3...", flush=True)
s2_idx = BlockIndex()
with open(TRAIN_S2, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        s2_idx.add(row['entity_id'], row['business_name'], row['business_address'], row['country'])

s3_idx = BlockIndex()
with open(TRAIN_S3, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        s3_idx.add(row['entity_id'], row['business_name'], row['business_address'], row['country'])

# 3. Train Model on Training Split
print("Generating training pairs...", flush=True)
X, y = [], []
train_sample = train_s1[:50000] # Cap training like V4 did
for r in train_sample:
    s1_id = r['entity_id']
    nn = norm_name(r['business_name'])
    na = norm_addr(r['business_address'])
    an = addr_nums(r['business_address'])
    country = r['country']
    
    cands = s2_idx.get_candidates(nn, an, country) | s3_idx.get_candidates(nn, an, country)
    true_m = gt.get(s1_id, set())
    
    pos = [c for c in cands if c in true_m]
    neg = [c for c in cands if c not in true_m]
    
    max_neg = max(len(pos) * 5, 3)
    if len(neg) > max_neg:
        neg = random.sample(neg, max_neg)
        
    for c in pos + neg:
        is_s2 = c in s2_idx.data
        d = s2_idx.data[c] if is_s2 else s3_idx.data.get(c)
        if not d: continue
        c_nn, c_na, c_an, c_country = d
        feat = compute_features_v4(nn, c_nn, na, c_na, an, c_an, country, c_country, is_s2, name_idf, addr_idf)
        X.append(feat)
        y.append(1 if c in true_m else 0)

print(f"Training samples: {len(X)}", flush=True)
model = SGDLogistic(N_FEATURES, lr=SGD_LR, epochs=SGD_EPOCHS)
model.train(X, y)

# 4. Run inference on Validation Split
print("Running inference on validation set...", flush=True)
val_gt_stats = Counter()
val_pred_stats = Counter()

val_s2_only = 0
val_s3_only = 0
val_both = 0
val_0 = 0

for i, r in enumerate(val_s1[:10000]):
    s1_id = r['entity_id']
    nn = norm_name(r['business_name'])
    na = norm_addr(r['business_address'])
    an = addr_nums(r['business_address'])
    country = r['country']
    
    cands_s2 = s2_idx.get_candidates(nn, an, country)
    cands_s3 = s3_idx.get_candidates(nn, an, country)
    all_cands = cands_s2 | cands_s3
    
    matches = []
    for c in all_cands:
        is_s2 = c in s2_idx.data
        d = s2_idx.data[c] if is_s2 else s3_idx.data.get(c)
        if not d: continue
        c_nn, c_na, c_an, c_country = d
        
        if not fast_prefilter(nn, c_nn):
            continue
            
        feat = compute_features_v4(nn, c_nn, na, c_na, an, c_an, country, c_country, is_s2, name_idf, addr_idf)
        score = model.predict(feat)
        if score >= THRESHOLD:
            matches.append(c)
            
    # Record Pred Stats
    n_m = len(matches)
    val_pred_stats[n_m] += 1
    
    has_s2 = any(m.startswith('S2') for m in matches)
    has_s3 = any(m.startswith('S3') for m in matches)
    if has_s2 and not has_s3: val_s2_only += 1
    elif has_s3 and not has_s2: val_s3_only += 1
    elif has_s2 and has_s3: val_both += 1
    else: val_0 += 1
    
    # Record GT Stats for this S1
    true_m = gt.get(s1_id, set())
    val_gt_stats[len(true_m)] += 1
    
    if (i+1) % 10000 == 0:
        print(f"Processed {i+1}/10000 val S1s", flush=True)

# 5. Output Results
out_path = 'experiments/v6_2/v4_cardinality.tsv'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write("metric\tpred\tgt\n")
    f.write(f"0_match_entities\t{val_pred_stats[0]}\t{val_gt_stats[0]}\n")
    f.write(f"1_match_entities\t{val_pred_stats[1]}\t{val_gt_stats[1]}\n")
    f.write(f"2_match_entities\t{val_pred_stats[2]}\t{val_gt_stats[2]}\n")
    f.write(f"3_match_entities\t{val_pred_stats[3]}\t{val_gt_stats[3]}\n")
    pred_4_plus = sum(v for k, v in val_pred_stats.items() if k >= 4)
    gt_4_plus = sum(v for k, v in val_gt_stats.items() if k >= 4)
    f.write(f"4_plus_match_entities\t{pred_4_plus}\t{gt_4_plus}\n")
    
    f.write(f"S2_only_pred\t{val_s2_only}\t-\n")
    f.write(f"S3_only_pred\t{val_s3_only}\t-\n")
    f.write(f"both_S2_and_S3_pred\t{val_both}\t-\n")
    
print(f"Done. Saved to {out_path}", flush=True)
