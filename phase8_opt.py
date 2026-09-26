import csv
import math
import os
import re
import random
import time
import sys
from collections import defaultdict, Counter

# Configuration
TRAIN_S1 = 'dataset/train/train_source1.tsv'
TRAIN_S2 = 'dataset/train/train_source2.tsv'
TRAIN_S3 = 'dataset/train/train_source3.tsv'
TRAIN_GT = 'dataset/train/train_ground_truth.tsv'
IDF_SAMPLE = 200000
SGD_EPOCHS = 5
SGD_LR = 0.01
CHAR4_MAX_BLOCK = 500

# Base functions
_LEGAL_RE = re.compile(r'\b(?:inc|incorporated|llc|corp|corporation|ltd|limited|co|company|plc|pvt|private)\b')
_CLEAN_RE = re.compile(r'[^\w\s]')
_SPACE_RE = re.compile(r'\s+')
_NUM_RE   = re.compile(r'\d+')

def norm_text(text):
    if not isinstance(text, str): return ""
    return _SPACE_RE.sub(' ', _CLEAN_RE.sub(' ', text.lower())).strip()
def norm_name(name):
    n = norm_text(name)
    n = _LEGAL_RE.sub('', n)
    return _SPACE_RE.sub(' ', n).strip()
def addr_nums(address):
    if not isinstance(address, str): return ""
    return " ".join(_NUM_RE.findall(address))
def norm_addr(address):
    return norm_text(address)

def token_jaccard(a, b):
    if not a or not b: return 0.0
    sa, sb = set(a.split()), set(b.split())
    u = sa | sb
    if not u: return 0.0
    return len(sa & sb) / len(u)
def str_ratio(a, b):
    if not a or not b: return 0.0
    if a == b: return 1.0
    if len(a) == 1 or len(b) == 1: return 1.0 if a == b else 0.0
    a_bigrams = Counter(a[i:i+2] for i in range(len(a)-1))
    b_bigrams = Counter(b[i:i+2] for i in range(len(b)-1))
    overlap = sum((a_bigrams & b_bigrams).values())
    total = sum(a_bigrams.values()) + sum(b_bigrams.values())
    if total == 0: return 0.0
    return 2.0 * overlap / total
def char_ngram_jaccard(a, b, n=3):
    if not a or not b or len(a) < n or len(b) < n: return 0.0
    sa = set(a[i:i+n] for i in range(len(a)-n+1))
    sb = set(b[i:i+n] for i in range(len(b)-n+1))
    u = sa | sb
    if not u: return 0.0
    return len(sa & sb) / len(u)
def numeric_token_overlap(a, b):
    if not a or not b: return 0.0
    sa, sb = set(a.split()), set(b.split())
    u = sa | sb
    if not u: return 0.0
    return len(sa & sb) / len(u)
def build_idf_vocab(filepath, max_rows):
    name_df = Counter()
    addr_df = Counter()
    total = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            total += 1
            for t in set(norm_name(row['business_name']).split()): name_df[t] += 1
            for t in set(norm_addr(row['business_address']).split()): addr_df[t] += 1
            if total >= max_rows: break
    name_idf = {k: math.log((total + 1) / (v + 1)) + 1 for k, v in name_df.items()}
    addr_idf = {k: math.log((total + 1) / (v + 1)) + 1 for k, v in addr_df.items()}
    return name_idf, addr_idf
def weighted_jaccard(a, b, idf):
    if not a or not b: return 0.0
    sa, sb = set(a.split()), set(b.split())
    inter = sa & sb
    union = sa | sb
    if not union: return 0.0
    w_inter = sum(idf.get(t, 5.0) for t in inter)
    w_union = sum(idf.get(t, 5.0) for t in union)
    if w_union == 0: return 0.0
    return w_inter / w_union
def weighted_overlap(a, b, idf):
    if not a or not b: return 0.0
    sa, sb = set(a.split()), set(b.split())
    inter = sa & sb
    return sum(idf.get(t, 5.0) for t in inter)

def compute_features_v4(s1_nn, s2_nn, s1_na, s2_na, s1_an, s2_an, s1_c, s2_c, is_s2, name_idf, addr_idf):
    name_exact  = 1.0 if s1_nn == s2_nn and s1_nn else 0.0
    addr_exact  = 1.0 if s1_na == s2_na and s1_na else 0.0
    country_eq  = 1.0 if s1_c == s2_c and s1_c else 0.0
    name_ratio  = str_ratio(s1_nn, s2_nn)
    name_tj     = token_jaccard(s1_nn, s2_nn)
    addr_ratio  = str_ratio(s1_na, s2_na)
    addr_tj     = token_jaccard(s1_na, s2_na)
    name_lendif = abs(len(s1_nn) - len(s2_nn))
    addr_lendif = abs(len(s1_na) - len(s2_na))
    name_substr = 1.0 if (s1_nn and s2_nn and (s1_nn in s2_nn or s2_nn in s1_nn)) else 0.0
    num_overlap = numeric_token_overlap(s1_an, s2_an)
    name_char3  = char_ngram_jaccard(s1_nn, s2_nn, 3)
    wt_name_j   = weighted_jaccard(s1_nn, s2_nn, name_idf)
    wt_addr_j   = weighted_jaccard(s1_na, s2_na, addr_idf)
    wt_name_o   = weighted_overlap(s1_nn, s2_nn, name_idf)
    wt_addr_o   = weighted_overlap(s1_na, s2_na, addr_idf)
    name_x_addr = name_ratio * addr_ratio
    exact_name_bad_addr = name_exact * (1.0 - addr_ratio)
    high_name_num_overlap = name_ratio * num_overlap
    return [
        1.0, 1.0 if is_s2 else 0.0, name_exact, name_lendif, name_ratio, name_tj,
        addr_exact, addr_lendif, addr_ratio, country_eq, name_substr, addr_tj,
        num_overlap, name_char3, wt_name_j, wt_addr_j, wt_name_o, wt_addr_o,
        name_x_addr, exact_name_bad_addr, high_name_num_overlap
    ]

def fast_prefilter(s1_nn, s2_nn):
    if not s1_nn or not s2_nn: return False
    s1_toks = set(s1_nn.split())
    s2_toks = set(s2_nn.split())
    if s1_toks & s2_toks: return True
    if s1_nn in s2_nn or s2_nn in s1_nn: return True
    if len(s1_nn) >= 4 and len(s2_nn) >= 4 and s1_nn[:4] == s2_nn[:4]: return True
    return False

def sigmoid(z):
    if z < -30: return 0.0
    if z > 30: return 1.0
    return 1.0 / (1.0 + math.exp(-z))

class SGDLogistic:
    def __init__(self, n_features, lr=SGD_LR, epochs=SGD_EPOCHS):
        self.w = [0.0] * n_features
        self.lr = lr
        self.epochs = epochs
    def train(self, X, y):
        for ep in range(self.epochs):
            idx = list(range(len(X)))
            random.shuffle(idx)
            for i in idx:
                p = sigmoid(sum(self.w[j] * X[i][j] for j in range(len(self.w))))
                err = p - y[i]
                for j in range(len(self.w)): self.w[j] -= self.lr * err * X[i][j]
    def predict(self, x):
        return sigmoid(sum(self.w[j] * x[j] for j in range(len(self.w))))

def eval_preds(preds, gt):
    tp, fp, fn = 0, 0, 0
    singletons_correct, singletons_total = 0, 0
    for s1, true_m in gt.items():
        if s1 not in preds: continue
        p_m = preds[s1]
        if len(true_m) == 0:
            singletons_total += 1
            if len(p_m) == 0: singletons_correct += 1
        tp += len(p_m & true_m)
        fp += len(p_m - true_m)
        fn += len(true_m - p_m)
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f05 = (1.25 * p * r) / (0.25 * p + r) if p+r>0 else 0
    return p, r, f05, singletons_correct/max(singletons_total, 1), tp, fp, fn

def run(seed, is_v4=True):
    random.seed(seed)
    
    # Fast evaluation on first 100,000 train rows (80k train, 20k val)
    MAX_ROWS = 100000
    rows = []
    with open(TRAIN_S1, 'r', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f, delimiter='\t')):
            if i >= MAX_ROWS: break
            rows.append(row)
            
    random.shuffle(rows)
    split_idx = int(len(rows) * 0.8)
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]
    
    gt = {}
    with open(TRAIN_GT, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            m = row['matched_entity_ids']
            gt[row['source1_entity_id']] = set(m.split(',')) if m else set()
            
    name_idf, addr_idf = build_idf_vocab(TRAIN_S1, IDF_SAMPLE)
    
    print("Indexing S2/S3...")
    s23_data = {}
    s23_blocks_name = defaultdict(list)
    s23_blocks_addr = defaultdict(list)
    s23_blocks_p5   = defaultdict(list)
    s23_blocks_p4   = defaultdict(list)

    req_names = set()
    req_addrs = set()
    req_p5    = set()
    req_p4    = set()
    for r in train_rows + val_rows:
        nn = norm_name(r['business_name'])
        an = addr_nums(r['business_address'])
        r['_nn'], r['_na'], r['_an'] = nn, norm_addr(r['business_address']), an
        if nn: req_names.add(nn)
        if an: req_addrs.add(an)
        if len(nn) >= 5: req_p5.add(nn[:5])
        if len(nn) >= 4: req_p4.add(nn[:4])
        
    for src_file, is_s2 in [(TRAIN_S2, True), (TRAIN_S3, False)]:
        with open(src_file, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                eid = row['entity_id']
                nn = norm_name(row['business_name'])
                na = norm_addr(row['business_address'])
                an = addr_nums(row['business_address'])
                country = row['country']
                toks = nn.split()
                ft = toks[0] if toks else ""
                p5 = nn[:5] if len(nn) >= 5 else ""
                p4 = nn[:4] if len(nn) >= 4 else ""

                keep = False
                if nn in req_names and country: s23_blocks_name[(nn, country)].append(eid); keep = True
                if an in req_addrs and len(ft) > 2 and country: s23_blocks_addr[(an, ft, country)].append(eid); keep = True
                if p5 in req_p5 and an in req_addrs and country: s23_blocks_p5[(p5, an, country)].append(eid); keep = True
                if p4 in req_p4 and country: s23_blocks_p4[(p4, country)].append(eid); keep = True
                if keep: s23_data[eid] = (nn, na, an, country, is_s2)
                
    print("Building training set...")
    X, y = [], []
    for r in train_rows:
        s1_id = r['entity_id']
        nn, na, an, country = r['_nn'], r['_na'], r['_an'], r['country']
        toks = nn.split()
        ft = toks[0] if toks else ""
        p5 = nn[:5] if len(nn) >= 5 else ""
        p4 = nn[:4] if len(nn) >= 4 else ""

        cands = set()
        if nn and country: cands.update(s23_blocks_name.get((nn, country), []))
        if an and len(ft) > 2 and country: cands.update(s23_blocks_addr.get((an, ft, country), []))
        if p5 and an and country: cands.update(s23_blocks_p5.get((p5, an, country), []))
        if p4 and country:
            block = s23_blocks_p4.get((p4, country), [])
            if len(block) <= CHAR4_MAX_BLOCK: cands.update(block)

        true_m = gt.get(s1_id, set())
        pos = [c for c in cands if c in true_m]
        neg = [c for c in cands if c not in true_m]

        max_neg = max(len(pos) * 5, 3)
        if len(neg) > max_neg: neg = random.sample(neg, max_neg)

        for c in pos + neg:
            d = s23_data.get(c)
            if not d: continue
            c_nn, c_na, c_an, c_country, c_is_s2 = d
            if not is_v4 and not fast_prefilter(nn, c_nn): continue
            feat = compute_features_v4(nn, c_nn, na, c_na, an, c_an, country, c_country, c_is_s2, name_idf, addr_idf)
            X.append(feat)
            y.append(1 if c in true_m else 0)
            
    print(f"Training on {len(X)} samples...")
    model = SGDLogistic(21, lr=SGD_LR, epochs=SGD_EPOCHS)
    model.train(X, y)
    
    print("Evaluating val set thresholds...")
    raw_scores = {}
    for r in val_rows:
        s1_id = r['entity_id']
        nn, na, an, country = r['_nn'], r['_na'], r['_an'], r['country']
        toks = nn.split()
        ft = toks[0] if toks else ""
        p5 = nn[:5] if len(nn) >= 5 else ""
        p4 = nn[:4] if len(nn) >= 4 else ""

        cands = set()
        if nn and country: cands.update(s23_blocks_name.get((nn, country), []))
        if an and len(ft) > 2 and country: cands.update(s23_blocks_addr.get((an, ft, country), []))
        if p5 and an and country: cands.update(s23_blocks_p5.get((p5, an, country), []))
        if p4 and country:
            block = s23_blocks_p4.get((p4, country), [])
            if len(block) <= CHAR4_MAX_BLOCK: cands.update(block)

        scores = {}
        for c in cands:
            d = s23_data.get(c)
            if not d: continue
            c_nn, c_na, c_an, c_country, c_is_s2 = d
            if not fast_prefilter(nn, c_nn): continue
            feat = compute_features_v4(nn, c_nn, na, c_na, an, c_an, country, c_country, c_is_s2, name_idf, addr_idf)
            scores[c] = model.predict(feat)
        raw_scores[s1_id] = scores

    # V4 baseline threshold (0.675)
    preds = {}
    for s1, scores in raw_scores.items():
        preds[s1] = set(c for c, s in scores.items() if s >= 0.675)
    p, r, f05, sa, tp, fp, fn = eval_preds(preds, gt)
    print(f"Seed {seed} (is_v4={is_v4}) @ TH=0.675 -> F0.5: {f05:.4f} P: {p:.4f} R: {r:.4f}")
    
    print("Sweeping Thresholds...")
    best_f05, best_th = 0, 0
    for th_int in range(400, 950, 25):
        th = th_int / 1000.0
        preds = {}
        for s1, scores in raw_scores.items():
            preds[s1] = set(c for c, s in scores.items() if s >= th)
        p, r, f05, sa, tp, fp, fn = eval_preds(preds, gt)
        if f05 > best_f05: best_f05, best_th = f05, th
    print(f"Best Global TH: {best_th} -> F0.5: {best_f05:.4f}")
    
    print("Testing top-K cardinality selection at TH=0.5...")
    for K in [1, 2, 3, 5, 10]:
        preds = {}
        for s1, scores in raw_scores.items():
            sorted_cands = sorted([(c, s) for c, s in scores.items() if s >= 0.5], key=lambda x: x[1], reverse=True)
            preds[s1] = set(c for c, _ in sorted_cands[:K])
        p, r, f05, sa, tp, fp, fn = eval_preds(preds, gt)
        print(f"Top-{K} (TH=0.5) -> F0.5: {f05:.4f}")

if __name__ == '__main__':
    print("=== V4 CONTROL (is_v4=True) ===")
    run(42, is_v4=True)
    run(123, is_v4=True)
    print("=== RECOVERY_V1 (is_v4=False) ===")
    run(42, is_v4=False)
    run(123, is_v4=False)

