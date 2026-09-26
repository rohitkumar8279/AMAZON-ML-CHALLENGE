import os
import csv
import sys
import random
import math
from collections import defaultdict
from difflib import SequenceMatcher
from normalization import normalize_business_name, normalize_business_address, extract_numbers_from_address
from blocking import build_blocks, generate_candidates

# Set seed for reproducibility
random.seed(42)

def token_jaccard(str1, str2):
    s1, s2 = set(str1.split()), set(str2.split())
    if not s1 and not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)

def str_ratio(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

def load_dict_data(file_path):
    d = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            d[row['entity_id']] = row
    return d

def compute_features(s1_row, s23_row, is_s2):
    n1 = normalize_business_name(s1_row['business_name'])
    n2 = normalize_business_name(s23_row['business_name'])
    a1 = normalize_business_address(s1_row['business_address'])
    a2 = normalize_business_address(s23_row['business_address'])
    c1 = s1_row['country']
    c2 = s23_row['country']
    
    num1 = extract_numbers_from_address(a1)
    num2 = extract_numbers_from_address(a2)
    
    features = [
        1.0, # bias
        1.0 if is_s2 else 0.0,
        1.0 if n1 == n2 and n1 else 0.0,
        abs(len(n1) - len(n2)),
        str_ratio(n1, n2),
        token_jaccard(n1, n2),
        1.0 if a1 == a2 and a1 else 0.0,
        abs(len(a1) - len(a2)),
        str_ratio(a1, a2),
        token_jaccard(num1, num2),
        1.0 if c1 == c2 and c1 else 0.0
    ]
    return features

class LogisticRegressionSGD:
    def __init__(self, lr=0.01, epochs=5):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        
    def sigmoid(self, z):
        if z < -20: return 0.0
        if z > 20: return 1.0
        return 1.0 / (1.0 + math.exp(-z))
        
    def train(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        for ep in range(self.epochs):
            indices = list(range(len(X)))
            random.shuffle(indices)
            loss = 0.0
            for i in indices:
                xi = X[i]
                yi = y[i]
                z = sum(w * x for w, x in zip(self.weights, xi))
                pred = self.sigmoid(z)
                error = pred - yi
                # Update weights
                for j in range(n_features):
                    self.weights[j] -= self.lr * error * xi[j]
                
                # Cross entropy for logging
                if yi == 1:
                    loss -= math.log(max(pred, 1e-15))
                else:
                    loss -= math.log(max(1 - pred, 1e-15))
            print(f"Epoch {ep+1}/{self.epochs}, Loss: {loss/len(X):.4f}")
            
    def predict_proba(self, X):
        return [self.sigmoid(sum(w * x for w, x in zip(self.weights, xi))) for xi in X]

def calculate_f05(y_true, y_pred):
    true_pos = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    false_pos = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    false_neg = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    
    if precision == 0 and recall == 0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return precision, recall, f05, false_pos, false_neg

def run_pipeline(data_dir):
    print("=== PHASE 4-13: END-TO-END BASELINE PIPELINE ===")
    
    s2_train = load_dict_data(f'{data_dir}/train/train_source2.tsv')
    s3_train = load_dict_data(f'{data_dir}/train/train_source3.tsv')
    
    # Load GT
    gt = defaultdict(set)
    with open(f'{data_dir}/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            m = row['matched_entity_ids']
            gt[row['source1_entity_id']] = set(m.split(',')) if m else set()
            
    s1_ids = []
    s1_rows = {}
    with open(f'{data_dir}/train/train_source1.tsv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            s1_ids.append(row['entity_id'])
            s1_rows[row['entity_id']] = row
            
    # Subsample for baseline speed if needed, but let's use a 10% sample for fast local iteration
    sample_size = min(len(s1_ids), 100000) 
    random.shuffle(s1_ids)
    s1_ids = s1_ids[:sample_size]
    
    split_idx = int(len(s1_ids) * 0.8)
    train_s1 = s1_ids[:split_idx]
    val_s1 = s1_ids[split_idx:]
    
    print(f"Sampled {sample_size} S1 entities. Train: {len(train_s1)}, Val: {len(val_s1)}")
    
    # Build blocks
    print("Building blocks...")
    s2_indices = build_blocks(s2_train.values())
    s3_indices = build_blocks(s3_train.values())
    
    def generate_pairs_and_features(s1_list):
        X, y = [], []
        s1_iter = (s1_rows[eid] for eid in s1_list)
        
        true_matches_total = 0
        true_matches_found = 0
        total_candidates = 0
        
        for s1_id, candidates in generate_candidates(s1_iter, s2_indices, s3_indices):
            s1_row = s1_rows[s1_id]
            s1_gt = gt.get(s1_id, set())
            
            true_matches_total += len(s1_gt)
            true_matches_found += len(s1_gt.intersection(candidates))
            total_candidates += len(candidates)
            
            # To avoid extreme class imbalance during training, sample negative candidates
            # Keep all positive candidates, sample up to 10 negatives
            pos_cands = [c for c in candidates if c in s1_gt]
            neg_cands = [c for c in candidates if c not in s1_gt]
            if len(neg_cands) > 10:
                neg_cands = random.sample(neg_cands, 10)
                
            selected_cands = pos_cands + neg_cands
            
            for c in selected_cands:
                is_s2 = c in s2_train
                s23_row = s2_train[c] if is_s2 else s3_train.get(c)
                if not s23_row: continue
                
                feat = compute_features(s1_row, s23_row, is_s2)
                X.append(feat)
                y.append(1 if c in s1_gt else 0)
                
        return X, y, true_matches_total, true_matches_found, total_candidates
        
    print("Generating train features...")
    X_train, y_train, tr_tot, tr_fnd, tr_cand = generate_pairs_and_features(train_s1)
    
    print("Generating val features...")
    X_val, y_val, val_tot, val_fnd, val_cand = generate_pairs_and_features(val_s1)
    
    val_recall = (val_fnd / max(1, val_tot)) * 100
    print(f"Validation Blocking Recall: {val_recall:.2f}%. Max F0.5 ceiling is ~{val_recall:.2f}%")
    
    # Train
    print("Training Logistic Regression (SGD)...")
    model = LogisticRegressionSGD(lr=0.01, epochs=10)
    model.train(X_train, y_train)
    
    print("Predicting on Validation...")
    val_probs = model.predict_proba(X_val)
    
    best_f05 = 0
    best_thresh = 0.5
    best_metrics = ()
    
    for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        y_pred = [1 if p >= thresh else 0 for p in val_probs]
        p, r, f, fp, fn = calculate_f05(y_val, y_pred)
        # Note: this recall is w.r.t candidates. Total recall w.r.t ground truth requires adjustment.
        # Adjusted recall = True positives / (True positives + False negatives + Missed by blocking)
        # However, for threshold tuning, we just tune on candidates to get a sense.
        adj_fn = fn + (val_tot - val_fnd)
        adj_r = p / (p + adj_fn) if p+adj_fn > 0 else 0.0 # wait, p is precision...
        true_pos = sum(1 for t, p in zip(y_val, y_pred) if t == 1 and p == 1)
        adj_r = true_pos / (true_pos + adj_fn) if (true_pos + adj_fn) > 0 else 0.0
        
        adj_f05 = (1.25 * p * adj_r) / (0.25 * p + adj_r) if (p+adj_r)>0 else 0.0
        
        print(f"Thresh {thresh:.2f}: P={p:.4f}, R={adj_r:.4f}, F0.5={adj_f05:.4f}, FP={fp}")
        if adj_f05 > best_f05:
            best_f05 = adj_f05
            best_thresh = thresh
            best_metrics = (p, adj_r, adj_f05, fp, fn)
            
    print(f"\nBest Threshold: {best_thresh} -> F0.5: {best_f05:.4f}")
    
    # Free memory
    import gc
    del s2_train, s3_train, s2_indices, s3_indices, X_train, X_val, y_train, y_val, s1_rows
    gc.collect()
    
    # Generate Output for Test
    print("Generating Test Predictions...")
    os.makedirs('output', exist_ok=True)
    
    s2_test = load_dict_data(f'{data_dir}/test/test_source2.tsv')
    s3_test = load_dict_data(f'{data_dir}/test/test_source3.tsv')
    
    s2_test_indices = build_blocks(s2_test.values())
    s3_test_indices = build_blocks(s3_test.values())
    
    out_matches = open('output/matching_results.tsv', 'w', newline='', encoding='utf-8')
    out_candidates = open('output/candidate_pairs.tsv', 'w', newline='', encoding='utf-8')
    
    m_writer = csv.writer(out_matches, delimiter='\t')
    m_writer.writerow(['source1_entity_id', 'matched_entity_ids'])
    
    c_writer = csv.writer(out_candidates, delimiter='\t')
    c_writer.writerow(['source1_entity_id', 'candidate_entity_ids'])
    
    def test_s1_iter():
        with open(f'{data_dir}/test/test_source1.tsv', 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                yield row
                
    # Stream test data
    for s1_row in test_s1_iter():
        s1_id = s1_row['entity_id']
        
        # Get candidates
        norm_name = normalize_business_name(s1_row['business_name'])
        addr_nums = extract_numbers_from_address(s1_row['business_address'])
        country = s1_row['country']
        name_tokens = norm_name.split()
        first_token = name_tokens[0] if name_tokens else ""
        
        candidates = set()
        if norm_name and country:
            candidates.update(s2_test_indices[0].get((norm_name, country), []))
            candidates.update(s3_test_indices[0].get((norm_name, country), []))
        if addr_nums and len(first_token) > 2 and country:
            candidates.update(s2_test_indices[1].get((addr_nums, first_token, country), []))
            candidates.update(s3_test_indices[1].get((addr_nums, first_token, country), []))
            
        c_writer.writerow([s1_id, ','.join(candidates)])
        
        final_matches = []
        for c in candidates:
            is_s2 = c in s2_test
            s23_row = s2_test[c] if is_s2 else s3_test.get(c)
            if not s23_row: continue
            
            feat = compute_features(s1_row, s23_row, is_s2)
            prob = model.predict_proba([feat])[0]
            if prob >= best_thresh:
                final_matches.append(c)
                
        m_writer.writerow([s1_id, ','.join(final_matches)])
        
    out_matches.close()
    out_candidates.close()
    
if __name__ == '__main__':
    run_pipeline('dataset')
