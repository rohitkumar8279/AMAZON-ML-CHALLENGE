import sys
import os
import csv
import random
import time
import math
import numpy as np
import lightgbm as lgb
from collections import Counter, defaultdict

sys.path.append(os.path.abspath('experiments/pre_submission_v4_real_frozen'))
from inference_v4 import (
    BlockIndex, compute_features_v4, fast_prefilter,
    norm_name, addr_nums, norm_addr, str_ratio, token_jaccard, char_ngram_jaccard, numeric_token_overlap
)

print("Phase 3-7 Master Script", flush=True)

random.seed(42)

TRAIN_S1 = 'dataset/train/train_source1.tsv'
TRAIN_S2 = 'dataset/train/train_source2.tsv'
TRAIN_S3 = 'dataset/train/train_source3.tsv'
TRAIN_GT = 'dataset/train/train_ground_truth.tsv'

gt = {}
with open(TRAIN_GT, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        m = row['matched_entity_ids']
        if m: gt[row['source1_entity_id']] = set(m.split(','))

s1_rows = []
with open(TRAIN_S1, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter='\t'): s1_rows.append(row)

random.shuffle(s1_rows)
split_idx = int(len(s1_rows) * 0.8)
train_s1 = s1_rows[:split_idx]
val_s1 = s1_rows[split_idx:]
print(f"Train S1: {len(train_s1)}, Val S1: {len(val_s1)}", flush=True)

name_df, addr_df = Counter(), Counter()
for row in train_s1:
    for t in set(norm_name(row['business_name']).split()): name_df[t] += 1
    for t in set(norm_addr(row['business_address']).split()): addr_df[t] += 1
name_idf = {k: math.log((len(train_s1) + 1) / (v + 1)) + 1 for k, v in name_df.items()}
addr_idf = {k: math.log((len(train_s1) + 1) / (v + 1)) + 1 for k, v in addr_df.items()}

s2_idx = BlockIndex()
with open(TRAIN_S2, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        s2_idx.add(row['entity_id'], row['business_name'], row['business_address'], row['country'])

s3_idx = BlockIndex()
with open(TRAIN_S3, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        s3_idx.add(row['entity_id'], row['business_name'], row['business_address'], row['country'])

def get_dataset(s1_subset, max_samples=None):
    X, y, is_s2_list = [], [], []
    cands_map = defaultdict(list)
    s1_metadata = {}
    sample_size = min(len(s1_subset), max_samples) if max_samples else len(s1_subset)
    
    for r in s1_subset[:sample_size]:
        s1_id = r['entity_id']
        nn, na, an, country = norm_name(r['business_name']), norm_addr(r['business_address']), addr_nums(r['business_address']), r['country']
        s1_metadata[s1_id] = (nn, na, an, country)
        
        cands = s2_idx.get_candidates(nn, an, country) | s3_idx.get_candidates(nn, an, country)
        true_m = gt.get(s1_id, set())
        
        pos = [c for c in cands if c in true_m]
        neg = [c for c in cands if c not in true_m and fast_prefilter(nn, (s2_idx.data[c] if c in s2_idx.data else s3_idx.data[c])[0])]
        
        if max_samples:
            max_neg = max(len(pos) * 10, 5)
            if len(neg) > max_neg: neg = random.sample(neg, max_neg)
                
        for c in pos + neg:
            is_s2 = c in s2_idx.data
            d = s2_idx.data[c] if is_s2 else s3_idx.data[c]
            c_nn, c_na, c_an, c_country = d
            feat = compute_features_v4(nn, c_nn, na, c_na, an, c_an, country, c_country, is_s2, name_idf, addr_idf)
            X.append(feat)
            y.append(1 if c in true_m else 0)
            is_s2_list.append(is_s2)
            cands_map[s1_id].append((c, len(X)-1, is_s2))
            
    return np.array(X), np.array(y), np.array(is_s2_list), cands_map, s1_metadata

print("Generating train and val datasets...", flush=True)
X_train, y_train, s2_mask_train, _, _ = get_dataset(train_s1, max_samples=40000)
X_val, y_val, s2_mask_val, val_cands_map, s1_meta_val = get_dataset(val_s1, max_samples=10000)

print("Training base models...", flush=True)
clf_unified = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train)
clf_s2 = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
if sum(s2_mask_train) > 0: clf_s2.fit(X_train[s2_mask_train], y_train[s2_mask_train])
clf_s3 = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
if sum(~s2_mask_train) > 0: clf_s3.fit(X_train[~s2_mask_train], y_train[~s2_mask_train])

pred_unified = clf_unified.predict_proba(X_val)[:, 1]
pred_sep = np.zeros(len(X_val))
if sum(s2_mask_val) > 0: pred_sep[s2_mask_val] = clf_s2.predict_proba(X_val[s2_mask_val])[:, 1]
if sum(~s2_mask_val) > 0: pred_sep[~s2_mask_val] = clf_s3.predict_proba(X_val[~s2_mask_val])[:, 1]

def calc_f05(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    return (1.25 * prec * rec) / (0.25 * prec + rec) if prec + rec > 0 else 0

# === PHASE 3: Source-Specific Matchers ===
print("Phase 3: Source-Specific thresholds...", flush=True)
best_u_f05, best_u_th = 0, 0
for th in np.arange(0.1, 0.9, 0.05):
    tp, fp, fn = 0, 0, 0
    for s1_id, cands in val_cands_map.items():
        true_m = gt.get(s1_id, set())
        for c, idx, _ in cands:
            if pred_unified[idx] >= th:
                if c in true_m: tp += 1
                else: fp += 1
            elif c in true_m: fn += 1
        fn += len([tc for tc in true_m if tc not in {c for c,_,_ in cands}])
    f = calc_f05(tp, fp, fn)
    if f > best_u_f05: best_u_f05, best_u_th = f, th

best_sep_f05, best_th2, best_th3 = 0, 0, 0
for th2 in np.arange(0.1, 0.9, 0.1):
    for th3 in np.arange(0.1, 0.9, 0.1):
        tp, fp, fn = 0, 0, 0
        for s1_id, cands in val_cands_map.items():
            true_m = gt.get(s1_id, set())
            for c, idx, is_s2 in cands:
                th = th2 if is_s2 else th3
                if pred_sep[idx] >= th:
                    if c in true_m: tp += 1
                    else: fp += 1
                elif c in true_m: fn += 1
            fn += len([tc for tc in true_m if tc not in {c for c,_,_ in cands}])
        f = calc_f05(tp, fp, fn)
        if f > best_sep_f05: best_sep_f05, best_th2, best_th3 = f, th2, th3

with open('experiments/v6_2/source_specific.tsv', 'w') as f:
    f.write(f"Unified_LightGBM\t{best_u_th:.2f}\t{best_u_th:.2f}\t{best_u_f05:.4f}\n")
    f.write(f"Source_Specific\t{best_th2:.2f}\t{best_th3:.2f}\t{best_sep_f05:.4f}\n")

# === PHASE 4: Expected-F0.5 Decoder ===
print("Phase 4: Expected-F0.5 decoder...", flush=True)
def get_expected_f05_selections(preds, Kmax=10):
    selections = {}
    for s1_id, cands in val_cands_map.items():
        if not cands: continue
        sorted_cands = sorted(cands, key=lambda x: preds[x[1]], reverse=True)
        sum_all_pi = sum(preds[x[1]] for x in cands)
        best_k, best_ef05 = 0, 0.0
        sum_pi_k = 0.0
        for k in range(1, min(len(sorted_cands), Kmax) + 1):
            sum_pi_k += preds[sorted_cands[k-1][1]]
            ef05 = (1.25 * sum_pi_k) / (0.25 * sum_all_pi + k)
            if ef05 > best_ef05:
                best_ef05 = ef05
                best_k = k
        selections[s1_id] = [c[0] for c in sorted_cands[:best_k]]
    return selections

tp, fp, fn = 0, 0, 0
selections_ef05 = get_expected_f05_selections(pred_unified, Kmax=5)
for s1_id, cands in val_cands_map.items():
    true_m = gt.get(s1_id, set())
    sel = selections_ef05.get(s1_id, [])
    for c in sel:
        if c in true_m: tp += 1
        else: fp += 1
    for tc in true_m:
        if tc not in sel: fn += 1
ef05_score = calc_f05(tp, fp, fn)

with open('experiments/v6_2/entity_decoder.tsv', 'w') as f:
    f.write(f"Global_Threshold_{best_u_th:.2f}\t{best_u_f05:.4f}\n")
    f.write(f"Expected_F0.5_Kmax5\t{ef05_score:.4f}\n")

# === PHASE 5: Singleton Gate ===
print("Phase 5: Singleton Gate...", flush=True)
sg_X, sg_y = [], []
for s1_id, cands in val_cands_map.items():
    true_m = gt.get(s1_id, set())
    if not cands: continue
    scores = sorted([pred_unified[x[1]] for x in cands], reverse=True)
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else 0
    nn, na, _, _ = s1_meta_val[s1_id]
    sg_X.append([top1, top2, top1-top2, len(cands), len(nn), len(na)])
    sg_y.append(1 if len(true_m) > 0 else 0)

# Train simple Logistic on val itself with cross-val to avoid leakage? We just split val into 2 folds
sg_X, sg_y = np.array(sg_X), np.array(sg_y)
half = len(sg_X) // 2
clf_sg1 = lgb.LGBMClassifier(n_estimators=50).fit(sg_X[:half], sg_y[:half])
clf_sg2 = lgb.LGBMClassifier(n_estimators=50).fit(sg_X[half:], sg_y[half:])
gate_preds = np.concatenate([clf_sg1.predict_proba(sg_X[half:])[:,1], clf_sg2.predict_proba(sg_X[:half])[:,1]])
# evaluate
tp, fp, fn = 0, 0, 0
idx = 0
for s1_id, cands in val_cands_map.items():
    if not cands: continue
    true_m = gt.get(s1_id, set())
    gate = gate_preds[idx] > 0.5
    idx += 1
    
    sel = [c for c, cx, _ in cands if pred_unified[cx] >= best_u_th] if gate else []
    for c in sel:
        if c in true_m: tp += 1
        else: fp += 1
    for tc in true_m:
        if tc not in sel: fn += 1

sg_f05 = calc_f05(tp, fp, fn)
with open('experiments/v6_2/singleton_gate.tsv', 'w') as f:
    f.write(f"No_Gate\t{best_u_f05:.4f}\n")
    f.write(f"Learned_Gate\t{sg_f05:.4f}\n")

# === PHASE 7: Conflict Resolution ===
print("Phase 7: Conflict Resolution...", flush=True)
global_assignments = []
for s1_id, cands in val_cands_map.items():
    for c, cx, _ in cands:
        if pred_unified[cx] >= best_u_th:
            global_assignments.append((pred_unified[cx], s1_id, c))

global_assignments.sort(reverse=True, key=lambda x: x[0])
assigned_s2s3 = set()
tp, fp, fn = 0, 0, 0
resolved_preds = defaultdict(list)
for score, s1_id, c in global_assignments:
    if c not in assigned_s2s3:
        assigned_s2s3.add(c)
        resolved_preds[s1_id].append(c)

for s1_id in val_cands_map.keys():
    true_m = gt.get(s1_id, set())
    sel = resolved_preds.get(s1_id, [])
    for c in sel:
        if c in true_m: tp += 1
        else: fp += 1
    for tc in true_m:
        if tc not in sel: fn += 1

cr_f05 = calc_f05(tp, fp, fn)
with open('experiments/v6_2/conflict_resolution.tsv', 'w') as f:
    f.write(f"No_Resolver\t{best_u_f05:.4f}\n")
    f.write(f"Greedy_Resolver\t{cr_f05:.4f}\n")

print("Master Script Complete.", flush=True)
