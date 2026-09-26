import sys
import csv
import random
import math
import os
from collections import defaultdict
from difflib import SequenceMatcher

def normalize_text_basic(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    import re
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_business_name(name):
    name = normalize_text_basic(name)
    import re
    legal_suffixes = [r'\binc\b', r'\bincorporated\b', r'\bllc\b', r'\bcorp\b', r'\bcorporation\b', r'\bltd\b', r'\blimited\b', r'\bco\b', r'\bcompany\b', r'\bplc\b', r'\bpvt\b', r'\bprivate\b']
    for suffix in legal_suffixes:
        name = re.sub(suffix, '', name)
    return re.sub(r'\s+', ' ', name).strip()

def extract_numbers_from_address(address):
    if not isinstance(address, str): return ""
    import re
    numbers = re.findall(r'\d+', address)
    return " ".join(numbers)

def token_jaccard(str1, str2):
    if not str1 or not str2: return 0.0
    s1, s2 = set(str1.split()), set(str2.split())
    if not s1 and not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)

def str_ratio(str1, str2):
    if not str1 or not str2: return 0.0
    return SequenceMatcher(None, str1, str2).ratio()

def compute_features(s1_name, s2_name, s1_addr, s2_addr, s1_country, s2_country, is_s2):
    return [
        1.0, 
        1.0 if is_s2 else 0.0,
        1.0 if s1_name == s2_name and s1_name else 0.0,
        abs(len(s1_name) - len(s2_name)),
        str_ratio(s1_name, s2_name),
        token_jaccard(s1_name, s2_name),
        1.0 if s1_addr == s2_addr and s1_addr else 0.0,
        abs(len(s1_addr) - len(s2_addr)),
        str_ratio(s1_addr, s2_addr),
        1.0 if s1_country == s2_country and s1_country else 0.0
    ]

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
            for i in indices:
                xi = X[i]
                yi = y[i]
                z = sum(w * x for w, x in zip(self.weights, xi))
                pred = self.sigmoid(z)
                error = pred - yi
                for j in range(n_features):
                    self.weights[j] -= self.lr * error * xi[j]
                    
    def predict_proba(self, X):
        return [self.sigmoid(sum(w * x for w, x in zip(self.weights, xi))) for xi in X]

def run():
    print("Loading GT...", flush=True)
    gt = defaultdict(set)
    with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            m = row['matched_entity_ids']
            if m:
                gt[row['source1_entity_id']] = set(m.split(','))
                
    print("Loading 10,000 S1 records...", flush=True)
    s1_rows = []
    with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if i >= 10000: break
            row['norm_name'] = normalize_business_name(row['business_name'])
            row['addr_nums'] = extract_numbers_from_address(row['business_address'])
            s1_rows.append(row)
            
    print("Loading S2 and S3 indexed blocks...", flush=True)
    # To keep memory low, we'll only load S2 and S3 entities that could match the 10,000 S1 records!
    # How? By keeping a set of required block keys.
    req_norm_names = set(r['norm_name'] for r in s1_rows if r['norm_name'])
    req_addr_nums = set(r['addr_nums'] for r in s1_rows if r['addr_nums'])
    req_prefixes = set(r['norm_name'][:5] for r in s1_rows if r['norm_name'] and len(r['norm_name'])>=5)
    
    s2_exact_blocks = defaultdict(list)
    s2_addr_blocks = defaultdict(list)
    s2_prefix_blocks = defaultdict(list)
    s2_data = {}
    
    with open('dataset/train/train_source2.tsv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            norm_name = normalize_business_name(row['business_name'])
            addr_nums = extract_numbers_from_address(row['business_address'])
            country = row['country']
            toks = norm_name.split()
            first_tok = toks[0] if toks else ""
            prefix = norm_name[:5] if len(norm_name)>=5 else ""
            
            keep = False
            if norm_name in req_norm_names:
                s2_exact_blocks[(norm_name, country)].append(row['entity_id'])
                keep = True
            if addr_nums in req_addr_nums and len(first_tok) > 2:
                s2_addr_blocks[(addr_nums, first_tok, country)].append(row['entity_id'])
                keep = True
            if prefix in req_prefixes and addr_nums in req_addr_nums:
                s2_prefix_blocks[(prefix, addr_nums, country)].append(row['entity_id'])
                keep = True
            if keep:
                row['norm_name'] = norm_name
                row['addr_nums'] = addr_nums
                s2_data[row['entity_id']] = row
                
    s3_exact_blocks = defaultdict(list)
    s3_addr_blocks = defaultdict(list)
    s3_prefix_blocks = defaultdict(list)
    s3_data = {}
    
    with open('dataset/train/train_source3.tsv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            norm_name = normalize_business_name(row['business_name'])
            addr_nums = extract_numbers_from_address(row['business_address'])
            country = row['country']
            toks = norm_name.split()
            first_tok = toks[0] if toks else ""
            prefix = norm_name[:5] if len(norm_name)>=5 else ""
            
            keep = False
            if norm_name in req_norm_names:
                s3_exact_blocks[(norm_name, country)].append(row['entity_id'])
                keep = True
            if addr_nums in req_addr_nums and len(first_tok) > 2:
                s3_addr_blocks[(addr_nums, first_tok, country)].append(row['entity_id'])
                keep = True
            if prefix in req_prefixes and addr_nums in req_addr_nums:
                s3_prefix_blocks[(prefix, addr_nums, country)].append(row['entity_id'])
                keep = True
            if keep:
                row['norm_name'] = norm_name
                row['addr_nums'] = addr_nums
                s3_data[row['entity_id']] = row
                
    print("Evaluating Blocking V0 vs V1...", flush=True)
    v0_true_matches, v1_true_matches = 0, 0
    v0_total_cands, v1_total_cands = 0, 0
    total_true = sum(len(gt.get(r['entity_id'], set())) for r in s1_rows)
    
    v0_X, v0_y = [], []
    v1_X, v1_y = [], []
    
    for r in s1_rows:
        s1_id = r['entity_id']
        norm = r['norm_name']
        addr = r['addr_nums']
        country = r['country']
        toks = norm.split()
        first_tok = toks[0] if toks else ""
        prefix = norm[:5] if len(norm)>=5 else ""
        
        v0_cands = set()
        v1_cands = set()
        
        # V0 Rules
        if norm and country:
            v0_cands.update(s2_exact_blocks.get((norm, country), []))
            v0_cands.update(s3_exact_blocks.get((norm, country), []))
        if addr and len(first_tok)>2 and country:
            v0_cands.update(s2_addr_blocks.get((addr, first_tok, country), []))
            v0_cands.update(s3_addr_blocks.get((addr, first_tok, country), []))
            
        v1_cands.update(v0_cands)
        # V1 Rule
        if prefix and addr and country:
            v1_cands.update(s2_prefix_blocks.get((prefix, addr, country), []))
            v1_cands.update(s3_prefix_blocks.get((prefix, addr, country), []))
            
        true_m = gt.get(s1_id, set())
        v0_true_matches += len(true_m.intersection(v0_cands))
        v1_true_matches += len(true_m.intersection(v1_cands))
        v0_total_cands += len(v0_cands)
        v1_total_cands += len(v1_cands)
        
        # Build features for V0
        # Positive and sampled negatives
        pos_cands = [c for c in v0_cands if c in true_m]
        neg_cands = [c for c in v0_cands if c not in true_m]
        if len(neg_cands) > 10: neg_cands = random.sample(neg_cands, 10)
        for c in pos_cands + neg_cands:
            is_s2 = c in s2_data
            s23_row = s2_data[c] if is_s2 else s3_data.get(c)
            if not s23_row: continue
            feat = compute_features(norm, s23_row['norm_name'], r['business_address'], s23_row['business_address'], country, s23_row['country'], is_s2)
            v0_X.append(feat)
            v0_y.append(1 if c in true_m else 0)
            
        # Build features for V1
        pos_cands1 = [c for c in v1_cands if c in true_m]
        neg_cands1 = [c for c in v1_cands if c not in true_m]
        if len(neg_cands1) > 10: neg_cands1 = random.sample(neg_cands1, 10)
        for c in pos_cands1 + neg_cands1:
            is_s2 = c in s2_data
            s23_row = s2_data[c] if is_s2 else s3_data.get(c)
            if not s23_row: continue
            feat = compute_features(norm, s23_row['norm_name'], r['business_address'], s23_row['business_address'], country, s23_row['country'], is_s2)
            v1_X.append(feat)
            v1_y.append(1 if c in true_m else 0)

    print(f"V0 Blocking Recall: {v0_true_matches/total_true*100:.2f}% | Cands: {v0_total_cands} | Avg: {v0_total_cands/10000:.2f}")
    print(f"V1 Blocking Recall: {v1_true_matches/total_true*100:.2f}% | Cands: {v1_total_cands} | Avg: {v1_total_cands/10000:.2f}")

    # Train and Eval V0
    print("Training V0 Model...", flush=True)
    split0 = int(len(v0_X) * 0.8)
    m0 = LogisticRegressionSGD(epochs=5)
    m0.train(v0_X[:split0], v0_y[:split0])
    p0 = m0.predict_proba(v0_X[split0:])
    p0_pred = [1 if p > 0.5 else 0 for p in p0]
    tp0 = sum(1 for t, p in zip(v0_y[split0:], p0_pred) if t==1 and p==1)
    fp0 = sum(1 for t, p in zip(v0_y[split0:], p0_pred) if t==0 and p==1)
    fn0 = sum(1 for t, p in zip(v0_y[split0:], p0_pred) if t==1 and p==0)
    prec0 = tp0 / (tp0+fp0) if (tp0+fp0)>0 else 0
    rec0 = tp0 / (tp0+fn0) if (tp0+fn0)>0 else 0
    f05_0 = (1.25*prec0*rec0)/(0.25*prec0+rec0) if (prec0+rec0)>0 else 0
    print(f"V0 ML -> P: {prec0:.4f}, R: {rec0:.4f}, F0.5: {f05_0:.4f}")
    
    # Train and Eval V1
    print("Training V1 Model...", flush=True)
    split1 = int(len(v1_X) * 0.8)
    m1 = LogisticRegressionSGD(epochs=5)
    m1.train(v1_X[:split1], v1_y[:split1])
    p1 = m1.predict_proba(v1_X[split1:])
    p1_pred = [1 if p > 0.5 else 0 for p in p1]
    tp1 = sum(1 for t, p in zip(v1_y[split1:], p1_pred) if t==1 and p==1)
    fp1 = sum(1 for t, p in zip(v1_y[split1:], p1_pred) if t==0 and p==1)
    fn1 = sum(1 for t, p in zip(v1_y[split1:], p1_pred) if t==1 and p==0)
    prec1 = tp1 / (tp1+fp1) if (tp1+fp1)>0 else 0
    rec1 = tp1 / (tp1+fn1) if (tp1+fn1)>0 else 0
    f05_1 = (1.25*prec1*rec1)/(0.25*prec1+rec1) if (prec1+rec1)>0 else 0
    print(f"V1 ML -> P: {prec1:.4f}, R: {rec1:.4f}, F0.5: {f05_1:.4f}")
    
    # Dummy Outputs
    os.makedirs('output', exist_ok=True)
    with open('output/matching_results.tsv', 'w') as f: f.write("source1_entity_id\tmatched_entity_ids\n1\t2\n")
    with open('output/candidate_pairs.tsv', 'w') as f: f.write("source1_entity_id\tcandidate_entity_ids\n1\t2\n")
    print("Done.", flush=True)

if __name__ == '__main__':
    run()
