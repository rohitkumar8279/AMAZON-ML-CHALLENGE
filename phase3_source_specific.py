import sys
import os
import csv
import random
import time
import math
import numpy as np
import lightgbm as lgb
from collections import Counter, defaultdict

# Add pre_submission_v4_real_frozen to path to import its modules
sys.path.append(os.path.abspath('experiments/pre_submission_v4_real_frozen'))
from inference_v4 import (
    BlockIndex, compute_features_v4, fast_prefilter,
    build_idf_vocab, norm_name, addr_nums, norm_addr, CHAR4_MAX_BLOCK
)

print("Phase 3: Source-Specific Matchers", flush=True)

random.seed(42)

TRAIN_S1 = 'dataset/train/train_source1.tsv'
TRAIN_S2 = 'dataset/train/train_source2.tsv'
TRAIN_S3 = 'dataset/train/train_source3.tsv'
TRAIN_GT = 'dataset/train/train_ground_truth.tsv'

# 1. Load GT
gt = {}
with open(TRAIN_GT, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        m = row['matched_entity_ids']
        if m:
            gt[row['source1_entity_id']] = set(m.split(','))

# 2. Split Data
s1_rows = []
with open(TRAIN_S1, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        s1_rows.append(row)

random.shuffle(s1_rows)
split_idx = int(len(s1_rows) * 0.8)
train_s1 = s1_rows[:split_idx]
val_s1 = s1_rows[split_idx:]
print(f"Train S1: {len(train_s1)}, Val S1: {len(val_s1)}", flush=True)

# 3. Build IDF
name_df, addr_df = Counter(), Counter()
for row in train_s1:
    for t in set(norm_name(row['business_name']).split()): name_df[t] += 1
    for t in set(norm_addr(row['business_address']).split()): addr_df[t] += 1
name_idf = {k: math.log((len(train_s1) + 1) / (v + 1)) + 1 for k, v in name_df.items()}
addr_idf = {k: math.log((len(train_s1) + 1) / (v + 1)) + 1 for k, v in addr_df.items()}

# 4. Index S2 and S3
s2_idx = BlockIndex()
with open(TRAIN_S2, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        s2_idx.add(row['entity_id'], row['business_name'], row['business_address'], row['country'])

s3_idx = BlockIndex()
with open(TRAIN_S3, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        s3_idx.add(row['entity_id'], row['business_name'], row['business_address'], row['country'])

# 5. Extract Train Features
print("Generating training dataset...", flush=True)
def get_dataset(s1_subset, max_samples=None):
    X, y, is_s2_list = [], [], []
    cands_map = defaultdict(list)
    sample_size = min(len(s1_subset), max_samples) if max_samples else len(s1_subset)
    
    for r in s1_subset[:sample_size]:
        s1_id = r['entity_id']
        nn, na, an, country = norm_name(r['business_name']), norm_addr(r['business_address']), addr_nums(r['business_address']), r['country']
        
        cands = s2_idx.get_candidates(nn, an, country) | s3_idx.get_candidates(nn, an, country)
        true_m = gt.get(s1_id, set())
        
        pos = [c for c in cands if c in true_m]
        neg = [c for c in cands if c not in true_m and fast_prefilter(nn, (s2_idx.data[c] if c in s2_idx.data else s3_idx.data[c])[0])]
        
        # Balance negatives for training
        if max_samples:
            max_neg = max(len(pos) * 10, 5)
            if len(neg) > max_neg:
                neg = random.sample(neg, max_neg)
                
        for c in pos + neg:
            is_s2 = c in s2_idx.data
            d = s2_idx.data[c] if is_s2 else s3_idx.data[c]
            c_nn, c_na, c_an, c_country = d
            feat = compute_features_v4(nn, c_nn, na, c_na, an, c_an, country, c_country, is_s2, name_idf, addr_idf)
            X.append(feat)
            y.append(1 if c in true_m else 0)
            is_s2_list.append(is_s2)
            cands_map[s1_id].append((c, len(X)-1))
            
    return np.array(X), np.array(y), np.array(is_s2_list), cands_map

X_train, y_train, s2_mask_train, _ = get_dataset(train_s1, max_samples=50000)

print("Training Unified LightGBM...", flush=True)
clf_unified = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_unified.fit(X_train, y_train)

print("Training S2 LightGBM...", flush=True)
clf_s2 = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
if sum(s2_mask_train) > 0:
    clf_s2.fit(X_train[s2_mask_train], y_train[s2_mask_train])

print("Training S3 LightGBM...", flush=True)
clf_s3 = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
s3_mask_train = ~s2_mask_train
if sum(s3_mask_train) > 0:
    clf_s3.fit(X_train[s3_mask_train], y_train[s3_mask_train])

# 6. Extract Val Features
print("Generating validation dataset...", flush=True)
X_val, y_val, s2_mask_val, val_cands_map = get_dataset(val_s1, max_samples=10000)

pred_unified = clf_unified.predict_proba(X_val)[:, 1]
pred_s2 = clf_s2.predict_proba(X_val[s2_mask_val])[:, 1] if sum(s2_mask_val) > 0 else np.zeros(0)
pred_s3 = clf_s3.predict_proba(X_val[~s2_mask_val])[:, 1] if sum(~s2_mask_val) > 0 else np.zeros(0)

pred_sep = np.zeros(len(X_val))
if sum(s2_mask_val) > 0: pred_sep[s2_mask_val] = pred_s2
if sum(~s2_mask_val) > 0: pred_sep[~s2_mask_val] = pred_s3

def evaluate_threshold(preds, y, s1_groups, thres, thres_s3=None, mask_s2=None):
    tp, fp, fn = 0, 0, 0
    for s1_id, cands in s1_groups.items():
        true_m = gt.get(s1_id, set())
        for c, idx in cands:
            p = preds[idx]
            if thres_s3 is not None and mask_s2 is not None:
                t = thres if mask_s2[idx] else thres_s3
            else:
                t = thres
            
            is_match = (p >= t)
            is_true = c in true_m
            
            if is_match and is_true: tp += 1
            elif is_match and not is_true: fp += 1
            elif not is_match and is_true: fn += 1
            
        # Add missed GT (not in candidates)
        c_set = {c for c, _ in cands}
        for t_c in true_m:
            if t_c not in c_set:
                fn += 1
                
    prec = tp / (tp + fp) if (tp+fp) > 0 else 0
    rec = tp / (tp + fn) if (tp+fn) > 0 else 0
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec+rec) > 0 else 0
    return prec, rec, f05

print("Tuning Unified Threshold...", flush=True)
best_u_f05, best_u_th = 0, 0
for th in np.arange(0.1, 0.9, 0.05):
    p, r, f = evaluate_threshold(pred_unified, y_val, val_cands_map, th)
    if f > best_u_f05: best_u_f05, best_u_th = f, th

print(f"Unified Best Thres: {best_u_th:.2f} -> F0.5: {best_u_f05:.4f}", flush=True)

print("Tuning S2/S3 Independent Thresholds...", flush=True)
best_sep_f05 = 0
best_sep_th2, best_sep_th3 = 0, 0
for th2 in np.arange(0.1, 0.9, 0.1):
    for th3 in np.arange(0.1, 0.9, 0.1):
        p, r, f = evaluate_threshold(pred_sep, y_val, val_cands_map, th2, th3, s2_mask_val)
        if f > best_sep_f05:
            best_sep_f05, best_sep_th2, best_sep_th3 = f, th2, th3

print(f"Separate Best Thres S2: {best_sep_th2:.2f}, S3: {best_sep_th3:.2f} -> F0.5: {best_sep_f05:.4f}", flush=True)

out_path = 'experiments/v6_2/source_specific.tsv'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write("Model\tThreshold_S2\tThreshold_S3\tF0.5\n")
    f.write(f"Unified_LightGBM\t{best_u_th:.2f}\t{best_u_th:.2f}\t{best_u_f05:.4f}\n")
    f.write(f"Source_Specific\t{best_sep_th2:.2f}\t{best_sep_th3:.2f}\t{best_sep_f05:.4f}\n")

print(f"Done. Saved to {out_path}", flush=True)
