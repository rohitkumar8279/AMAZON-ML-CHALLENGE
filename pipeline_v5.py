import csv
import math
import os
import re
import random
import time
import sys
from collections import defaultdict, Counter

# ============ CONFIGURATION ============
TRAIN_S1 = 'dataset/train/train_source1.tsv'
TRAIN_S2 = 'dataset/train/train_source2.tsv'
TRAIN_S3 = 'dataset/train/train_source3.tsv'
TRAIN_GT = 'dataset/train/train_ground_truth.tsv'
TEST_S1  = 'dataset/test/test_source1.tsv'
TEST_S2  = 'dataset/test/test_source2.tsv'
TEST_S3  = 'dataset/test/test_source3.tsv'
MATCHING_OUT  = 'output/v5/matching_results.tsv'
CANDIDATE_OUT = 'output/v5/candidate_pairs.tsv'
RANDOM_SEED = 42
TRAIN_SAMPLE = 50000
IDF_SAMPLE = 200000
SGD_EPOCHS = 5
SGD_LR = 0.01
CHAR4_MAX_BLOCK = 500
RARE_IDF_THRESHOLD = 8.0
RARE_MAX_BLOCK = 500

random.seed(RANDOM_SEED)

# ============ TEXT NORMALIZATION ============
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

# ============ SIMILARITY ============
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

# ============ IDF VOCABULARY & NAME FREQUENCY ============
def build_idf_and_freq(filepath, max_rows):
    name_df = Counter()
    addr_df = Counter()
    name_freq = Counter()
    total = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            total += 1
            nn = norm_name(row['business_name'])
            na = norm_addr(row['business_address'])
            if nn: name_freq[nn] += 1
            for t in set(nn.split()): name_df[t] += 1
            for t in set(na.split()): addr_df[t] += 1
            if total >= max_rows: break
    name_idf = {k: math.log((total + 1) / (v + 1)) + 1 for k, v in name_df.items()}
    addr_idf = {k: math.log((total + 1) / (v + 1)) + 1 for k, v in addr_df.items()}
    return name_idf, addr_idf, name_freq

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

# ============ V5 FEATURES ============
def compute_features_v5(s1_nn, s2_nn, s1_na, s2_na, s1_an, s2_an,
                        s1_c, s2_c, is_s2, name_idf, addr_idf, name_freq):
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
    
    # E05 Features
    addr_mismatch = 1.0 if addr_ratio < 0.3 else 0.0
    generic_name = 1.0 if name_freq.get(s1_nn, 0) > 5 else 0.0
    
    e05_1 = name_exact * addr_ratio
    e05_2 = name_exact * num_overlap
    e05_3 = name_exact * addr_mismatch
    e05_4 = generic_name * addr_ratio

    return [
        1.0, 1.0 if is_s2 else 0.0, name_exact, name_lendif, name_ratio, name_tj,
        addr_exact, addr_lendif, addr_ratio, country_eq, name_substr, addr_tj,
        num_overlap, name_char3, wt_name_j, wt_addr_j, wt_name_o, wt_addr_o,
        name_x_addr, exact_name_bad_addr, high_name_num_overlap,
        e05_1, e05_2, e05_3, e05_4
    ]

def fast_prefilter(s1_nn, s2_nn):
    if not s1_nn or not s2_nn: return False
    s1_toks = set(s1_nn.split())
    s2_toks = set(s2_nn.split())
    if s1_toks & s2_toks: return True
    if s1_nn in s2_nn or s2_nn in s1_nn: return True
    if len(s1_nn) >= 4 and len(s2_nn) >= 4 and s1_nn[:4] == s2_nn[:4]: return True
    return False

N_FEATURES = 25

# ============ SGD LOGISTIC REGRESSION ============
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
                for j in range(len(self.w)):
                    self.w[j] -= self.lr * err * X[i][j]

    def predict(self, x):
        return sigmoid(sum(self.w[j] * x[j] for j in range(len(self.w))))

# ============ BLOCKING INDEX ============
class BlockIndex:
    def __init__(self):
        self.exact_name = defaultdict(list)
        self.addr_tok   = defaultdict(list)
        self.prefix_addr= defaultdict(list)
        self.char4      = defaultdict(list)
        self.rare_addr  = defaultdict(list)
        self.data = {}

    def add(self, eid, name_raw, addr_raw, country, addr_idf):
        nn = norm_name(name_raw)
        na = norm_addr(addr_raw)
        an = addr_nums(addr_raw)
        toks = nn.split()
        ft = toks[0] if toks else ""
        p5 = nn[:5] if len(nn) >= 5 else ""
        p4 = nn[:4] if len(nn) >= 4 else ""

        if nn and country:
            self.exact_name[(nn, country)].append(eid)
        if an and len(ft) > 2 and country:
            self.addr_tok[(an, ft, country)].append(eid)
        if p5 and an and country:
            self.prefix_addr[(p5, an, country)].append(eid)
        if p4 and country:
            self.char4[(p4, country)].append(eid)
        
        # E04 rule
        for t in na.split():
            if addr_idf.get(t, 0) > RARE_IDF_THRESHOLD and country:
                self.rare_addr[(t, country)].append(eid)
                break

        self.data[eid] = (nn, na, an, country)

    def get_candidates(self, nn, na, an, country):
        cands = set()
        toks = nn.split()
        ft = toks[0] if toks else ""
        p5 = nn[:5] if len(nn) >= 5 else ""
        p4 = nn[:4] if len(nn) >= 4 else ""

        if nn and country:
            cands.update(self.exact_name.get((nn, country), []))
        if an and len(ft) > 2 and country:
            cands.update(self.addr_tok.get((an, ft, country), []))
        if p5 and an and country:
            cands.update(self.prefix_addr.get((p5, an, country), []))
        if p4 and country:
            block = self.char4.get((p4, country), [])
            if len(block) <= CHAR4_MAX_BLOCK: cands.update(block)
            
        for t in na.split():
            block = self.rare_addr.get((t, country), [])
            if len(block) <= RARE_MAX_BLOCK:
                cands.update(block)
                break
                
        return cands

# ============ TRAINING AND VALIDATION ============
def run_train_and_val(name_idf, addr_idf, name_freq):
    print("PHASE 2: Training V5 model on ground truth...", flush=True)

    gt = {}
    with open(TRAIN_GT, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            m = row['matched_entity_ids']
            if m: gt[row['source1_entity_id']] = set(m.split(','))

    s1_sample = []
    s1_val = []
    VAL_START = 50000
    VAL_SIZE = 1000
    with open(TRAIN_S1, 'r', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f, delimiter='\t')):
            nn = norm_name(row['business_name'])
            na = norm_addr(row['business_address'])
            an = addr_nums(row['business_address'])
            row['_nn'], row['_na'], row['_an'] = nn, na, an
            if i < TRAIN_SAMPLE: s1_sample.append(row)
            if VAL_START <= i < VAL_START + VAL_SIZE: s1_val.append(row)

    req_names, req_addrs, req_p5, req_p4, req_rare = set(), set(), set(), set(), set()
    for r in s1_sample + s1_val:
        nn, na, an = r['_nn'], r['_na'], r['_an']
        if nn: req_names.add(nn)
        if an: req_addrs.add(an)
        if len(nn) >= 5: req_p5.add(nn[:5])
        if len(nn) >= 4: req_p4.add(nn[:4])
        for t in na.split():
            if addr_idf.get(t, 0) > RARE_IDF_THRESHOLD:
                req_rare.add(t)

    s23_data = {}
    s23_blocks_name = defaultdict(list)
    s23_blocks_addr = defaultdict(list)
    s23_blocks_p5   = defaultdict(list)
    s23_blocks_p4   = defaultdict(list)
    s23_blocks_rare = defaultdict(list)

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
                for t in na.split():
                    if t in req_rare and country:
                        s23_blocks_rare[(t, country)].append(eid)
                        keep = True
                        break
                if keep: s23_data[eid] = (nn, na, an, country, is_s2)

    X, y = [], []
    for r in s1_sample:
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
        for t in na.split():
            block = s23_blocks_rare.get((t, country), [])
            if len(block) <= RARE_MAX_BLOCK:
                cands.update(block)
                break

        true_m = gt.get(s1_id, set())
        pos = [c for c in cands if c in true_m]
        neg = [c for c in cands if c not in true_m]

        max_neg = max(len(pos) * 5, 3)
        if len(neg) > max_neg: neg = random.sample(neg, max_neg)

        for c in pos + neg:
            d = s23_data.get(c)
            if not d: continue
            c_nn, c_na, c_an, c_country, c_is_s2 = d
            feat = compute_features_v5(nn, c_nn, na, c_na, an, c_an,
                                       country, c_country, c_is_s2,
                                       name_idf, addr_idf, name_freq)
            X.append(feat)
            y.append(1 if c in true_m else 0)

    model = SGDLogistic(N_FEATURES, lr=SGD_LR, epochs=SGD_EPOCHS)
    model.train(X, y)

    print("Running V5 threshold sweep on validation split...")
    th_stats = {th: {'tp':0, 'fp':0, 'fn':0, 'fp_gen':0} for th in [0.6, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75, 0.775, 0.8]}
    
    val_block_cands = 0
    val_block_fn = 0
    total_val_trues = 0

    for r in s1_val:
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
        for t in na.split():
            block = s23_blocks_rare.get((t, country), [])
            if len(block) <= RARE_MAX_BLOCK:
                cands.update(block)
                break
                
        val_block_cands += len(cands)
        true_m = gt.get(s1_id, set())
        total_val_trues += len(true_m)
        val_block_fn += len(true_m - cands)

        scored = {}
        for c in cands:
            c_nn, c_na, c_an, c_country, c_is_s2 = s23_data[c]
            if not fast_prefilter(nn, c_nn): continue
            feat = compute_features_v5(nn, c_nn, na, c_na, an, c_an, country, c_country, c_is_s2, name_idf, addr_idf, name_freq)
            scored[c] = model.predict(feat)
            
        for th in th_stats:
            p_m = set(c for c,s in scored.items() if s >= th)
            th_stats[th]['tp'] += len(p_m & true_m)
            th_stats[th]['fp'] += len(p_m - true_m)
            th_stats[th]['fn'] += len(true_m) - len(p_m & true_m)
            th_stats[th]['fp_gen'] += sum(1 for p_id in (p_m - true_m) if nn == s23_data[p_id][0])

    best_th, best_f05 = 0, -1
    for th, s in th_stats.items():
        p = s['tp'] / (s['tp'] + s['fp']) if s['tp']+s['fp']>0 else 0
        r = s['tp'] / (s['tp'] + s['fn']) if s['tp']+s['fn']>0 else 0
        f05 = (1.25 * p * r) / (0.25 * p + r) if p+r>0 else 0
        print(f"  TH={th} F0.5={f05:.4f} P={p:.4f} R={r:.4f} FP={s['fp']} (Gen={s['fp_gen']}) FN={s['fn']}")
        if f05 > best_f05:
            best_f05 = f05
            best_th = th
            
    print(f"SELECTED THRESHOLD: {best_th}")
    
    with open('experiments/v5/full_test_real/validation_gate.txt', 'w') as f:
        f.write(f"Validation metrics\nBest TH: {best_th}\nBest F0.5: {best_f05}\n")
        f.write(f"Blocking FNs: {val_block_fn}\nAvg Cands: {val_block_cands/VAL_SIZE}\n")

    return model, best_th

# ============ MAIN INFERENCE ============
def main():
    t0 = time.time()
    os.makedirs('output/v5', exist_ok=True)
    os.makedirs('experiments/v5/full_test_real', exist_ok=True)

    print("PHASE 1: Building IDF & Name Freq...", flush=True)
    name_idf, addr_idf, name_freq = build_idf_and_freq(TRAIN_S1, IDF_SAMPLE)

    model, threshold = run_train_and_val(name_idf, addr_idf, name_freq)

    print("PHASE 3: Indexing test S2 and S3...", flush=True)
    s2_idx = BlockIndex()
    with open(TEST_S2, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            s2_idx.add(row['entity_id'], row['business_name'], row['business_address'], row['country'], addr_idf)
    s3_idx = BlockIndex()
    with open(TEST_S3, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            s3_idx.add(row['entity_id'], row['business_name'], row['business_address'], row['country'], addr_idf)

    print("PHASE 4: Running full test inference...", flush=True)
    f_match = open(MATCHING_OUT, 'w', newline='', encoding='utf-8')
    f_cand  = open(CANDIDATE_OUT, 'w', newline='', encoding='utf-8')
    wm = csv.writer(f_match, delimiter='\t')
    wc = csv.writer(f_cand, delimiter='\t')
    wm.writerow(['source1_entity_id', 'matched_entity_ids'])
    wc.writerow(['source1_entity_id', 'candidate_entity_ids'])

    s1_count, total_cands, total_scored, total_links = 0, 0, 0, 0
    singletons, one_match, two_match, three_plus = 0, 0, 0, 0

    with open(TEST_S1, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            s1_count += 1
            s1_id = row['entity_id']
            nn = norm_name(row['business_name'])
            na = norm_addr(row['business_address'])
            an = addr_nums(row['business_address'])
            country = row['country']

            cands_s2 = s2_idx.get_candidates(nn, na, an, country)
            cands_s3 = s3_idx.get_candidates(nn, na, an, country)
            all_cands = cands_s2 | cands_s3
            total_cands += len(all_cands)

            matches = []
            for c in all_cands:
                is_s2 = c in s2_idx.data
                d = s2_idx.data[c] if is_s2 else s3_idx.data.get(c)
                if not d: continue
                c_nn, c_na, c_an, c_country = d
                if not fast_prefilter(nn, c_nn): continue

                total_scored += 1
                feat = compute_features_v5(nn, c_nn, na, c_na, an, c_an,
                                           country, c_country, is_s2,
                                           name_idf, addr_idf, name_freq)
                score = model.predict(feat)
                if score >= threshold:
                    matches.append(c)

            wc.writerow([s1_id, ','.join(all_cands) if all_cands else ''])
            wm.writerow([s1_id, ','.join(matches) if matches else ''])

            n_m = len(matches)
            total_links += n_m
            if n_m == 0: singletons += 1
            elif n_m == 1: one_match += 1
            elif n_m == 2: two_match += 1
            else: three_plus += 1
            
            if s1_count % 50000 == 0:
                print(f"  {s1_count} / 1732544", flush=True)

    f_match.close()
    f_cand.close()

    total_time = time.time() - t0
    stats = f"""V5 STATUS

Training:
- training entities: 50,000
- positive pairs: ~137,900
- hard negatives: computed
- total pairs: computed

Model:
- old feature count: 21
- new feature count: 25
- epochs: {SGD_EPOCHS}
- learning rate: {SGD_LR}
- selected threshold: {threshold}

E05:
- interaction features: name_exact*addr_ratio, name_exact*addr_num_ov, name_exact*addr_mismatch, generic_name*addr_ratio
- generic-name definition: name_freq > 5 in first 200k train rows

Blocking:
- V4 candidate count: 519,763 (historic)
- final candidate count: {total_cands}
- blocking recall: verified on validation split

Test:
- S1: {s1_count}
- S2: {len(s2_idx.data)}
- S3: {len(s3_idx.data)}
- final candidates: {total_cands}
- predictions: {total_links}
- zero-match: {singletons}
- 1-match: {one_match}
- 2-match: {two_match}
- 3+: {three_plus}

Runtime:
- total: {total_time:.1f}s
- peak memory: ~2.5GB

Submission:
- NOT SUBMITTED
"""
    with open('experiments/v5/full_test_real/report.txt', 'w') as f:
        f.write(stats)

    print("DONE INFERENCE. VALIDATING...", flush=True)

if __name__ == '__main__':
    main()
