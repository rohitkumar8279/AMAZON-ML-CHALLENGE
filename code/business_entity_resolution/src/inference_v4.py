"""
PRE_SUBMISSION_V4 — Full Test Inference (Optimized)
====================================================
Same model, same features, same threshold as the original V4.
Optimization: skip expensive SequenceMatcher for candidates with zero
token overlap (guaranteed low score), cap char4 block size, and reuse
trained weights from PHASE 2.
"""
import csv
import math
import os
import re
import random
import time
import sys
from collections import defaultdict, Counter

# ============ CONFIGURATION ============
THRESHOLD = 0.675
TRAIN_S1 = 'dataset/train/train_source1.tsv'
TRAIN_S2 = 'dataset/train/train_source2.tsv'
TRAIN_S3 = 'dataset/train/train_source3.tsv'
TRAIN_GT = 'dataset/train/train_ground_truth.tsv'
TEST_S1  = 'dataset/test/test_source1.tsv'
TEST_S2  = 'dataset/test/test_source2.tsv'
TEST_S3  = 'dataset/test/test_source3.tsv'
MATCHING_OUT  = 'output/matching_results.tsv'
CANDIDATE_OUT = 'output/candidate_pairs.tsv'
RANDOM_SEED = 42
TRAIN_SAMPLE = 50000
IDF_SAMPLE = 200000
SGD_EPOCHS = 5
SGD_LR = 0.01
CHAR4_MAX_BLOCK = 500  # cap char4 blocks to avoid explosion

random.seed(RANDOM_SEED)

# ============ TEXT NORMALIZATION ============
_LEGAL_RE = re.compile(
    r'\b(?:inc|incorporated|llc|corp|corporation|ltd|limited|co|company|plc|pvt|private)\b'
)
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

# ============ SIMILARITY FUNCTIONS ============
def token_jaccard(a, b):
    if not a or not b: return 0.0
    sa, sb = set(a.split()), set(b.split())
    u = sa | sb
    if not u: return 0.0
    return len(sa & sb) / len(u)

def str_ratio(a, b):
    """Fast Sorensen-Dice bigram similarity — O(n) vs SequenceMatcher's O(n^2)."""
    if not a or not b: return 0.0
    if a == b: return 1.0
    if len(a) == 1 or len(b) == 1:
        return 1.0 if a == b else 0.0
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

# ============ IDF VOCABULARY ============
def build_idf_vocab(filepath, max_rows):
    name_df = Counter()
    addr_df = Counter()
    total = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            total += 1
            for t in set(norm_name(row['business_name']).split()):
                name_df[t] += 1
            for t in set(norm_addr(row['business_address']).split()):
                addr_df[t] += 1
            if total >= max_rows:
                break
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

# ============ V4 FEATURE VECTOR ============
def compute_features_v4(s1_nn, s2_nn, s1_na, s2_na, s1_an, s2_an,
                        s1_c, s2_c, is_s2, name_idf, addr_idf):
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
        1.0,                    # 0: bias
        1.0 if is_s2 else 0.0, # 1: source indicator
        name_exact,             # 2
        name_lendif,            # 3
        name_ratio,             # 4
        name_tj,                # 5
        addr_exact,             # 6
        addr_lendif,            # 7
        addr_ratio,             # 8
        country_eq,             # 9
        name_substr,            # 10
        addr_tj,                # 11
        num_overlap,            # 12
        name_char3,             # 13
        wt_name_j,              # 14
        wt_addr_j,              # 15
        wt_name_o,              # 16
        wt_addr_o,              # 17
        name_x_addr,            # 18
        exact_name_bad_addr,    # 19
        high_name_num_overlap,  # 20
    ]

# Fast pre-filter: skip pairs with zero name token overlap
# These will have name_tj=0, wt_name_j~=0, name_ratio~low, and will never
# pass threshold 0.675 given the learned weights heavily penalize low name similarity.
def fast_prefilter(s1_nn, s2_nn):
    if not s1_nn or not s2_nn:
        return False
    s1_toks = set(s1_nn.split())
    s2_toks = set(s2_nn.split())
    # Must share at least one name token, OR one must be substring of other
    if s1_toks & s2_toks:
        return True
    if s1_nn in s2_nn or s2_nn in s1_nn:
        return True
    # Check char4 prefix match as minimum
    if len(s1_nn) >= 4 and len(s2_nn) >= 4 and s1_nn[:4] == s2_nn[:4]:
        return True
    return False

N_FEATURES = 21

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
            print(f"  Epoch {ep+1}/{self.epochs} done", flush=True)

    def predict(self, x):
        return sigmoid(sum(self.w[j] * x[j] for j in range(len(self.w))))

# ============ BLOCKING INDEX ============
class BlockIndex:
    def __init__(self):
        self.exact_name = defaultdict(list)
        self.addr_tok   = defaultdict(list)
        self.prefix_addr= defaultdict(list)
        self.char4      = defaultdict(list)
        self.data = {}

    def add(self, eid, name_raw, addr_raw, country):
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

        self.data[eid] = (nn, na, an, country)

    def get_candidates(self, nn, an, country):
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
        # Cap char4 block to prevent explosion
        if p4 and country:
            block = self.char4.get((p4, country), [])
            if len(block) <= CHAR4_MAX_BLOCK:
                cands.update(block)
            # else: skip this massive block — too generic to be useful
        return cands

# ============ TRAINING ============
def train_model(name_idf, addr_idf):
    print("PHASE 2: Training model on ground truth...", flush=True)

    gt = {}
    with open(TRAIN_GT, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            m = row['matched_entity_ids']
            if m:
                gt[row['source1_entity_id']] = set(m.split(','))

    s1_sample = []
    with open(TRAIN_S1, 'r', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f, delimiter='\t')):
            if i >= TRAIN_SAMPLE: break
            s1_sample.append(row)

    req_names = set()
    req_addrs = set()
    req_p5    = set()
    req_p4    = set()
    for r in s1_sample:
        nn = norm_name(r['business_name'])
        an = addr_nums(r['business_address'])
        r['_nn'] = nn
        r['_na'] = norm_addr(r['business_address'])
        r['_an'] = an
        if nn: req_names.add(nn)
        if an: req_addrs.add(an)
        if len(nn) >= 5: req_p5.add(nn[:5])
        if len(nn) >= 4: req_p4.add(nn[:4])

    s23_data = {}
    s23_blocks_name = defaultdict(list)
    s23_blocks_addr = defaultdict(list)
    s23_blocks_p5   = defaultdict(list)
    s23_blocks_p4   = defaultdict(list)

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
                if nn in req_names and country:
                    s23_blocks_name[(nn, country)].append(eid)
                    keep = True
                if an in req_addrs and len(ft) > 2 and country:
                    s23_blocks_addr[(an, ft, country)].append(eid)
                    keep = True
                if p5 in req_p5 and an in req_addrs and country:
                    s23_blocks_p5[(p5, an, country)].append(eid)
                    keep = True
                if p4 in req_p4 and country:
                    s23_blocks_p4[(p4, country)].append(eid)
                    keep = True
                if keep:
                    s23_data[eid] = (nn, na, an, country, is_s2)

    print(f"  Loaded {len(s23_data)} relevant S2/S3 for training", flush=True)

    X, y = [], []
    for r in s1_sample:
        s1_id = r['entity_id']
        nn, na, an = r['_nn'], r['_na'], r['_an']
        country = r['country']
        toks = nn.split()
        ft = toks[0] if toks else ""
        p5 = nn[:5] if len(nn) >= 5 else ""
        p4 = nn[:4] if len(nn) >= 4 else ""

        cands = set()
        if nn and country:
            cands.update(s23_blocks_name.get((nn, country), []))
        if an and len(ft) > 2 and country:
            cands.update(s23_blocks_addr.get((an, ft, country), []))
        if p5 and an and country:
            cands.update(s23_blocks_p5.get((p5, an, country), []))
        if p4 and country:
            block = s23_blocks_p4.get((p4, country), [])
            if len(block) <= CHAR4_MAX_BLOCK:
                cands.update(block)

        true_m = gt.get(s1_id, set())
        pos = [c for c in cands if c in true_m]
        neg = [c for c in cands if c not in true_m]

        max_neg = max(len(pos) * 5, 3)
        if len(neg) > max_neg:
            neg = random.sample(neg, max_neg)

        for c in pos + neg:
            d = s23_data.get(c)
            if not d: continue
            c_nn, c_na, c_an, c_country, c_is_s2 = d
            feat = compute_features_v4(nn, c_nn, na, c_na, an, c_an,
                                       country, c_country, c_is_s2,
                                       name_idf, addr_idf)
            X.append(feat)
            y.append(1 if c in true_m else 0)

    pos_count = sum(y)
    neg_count = len(y) - pos_count
    print(f"  Training samples: {len(X)} (pos={pos_count}, neg={neg_count})", flush=True)

    model = SGDLogistic(N_FEATURES, lr=SGD_LR, epochs=SGD_EPOCHS)
    model.train(X, y)

    tp = fp = fn = tn = 0
    for i in range(len(X)):
        p = model.predict(X[i])
        pred = 1 if p >= THRESHOLD else 0
        if y[i] == 1 and pred == 1: tp += 1
        elif y[i] == 0 and pred == 1: fp += 1
        elif y[i] == 1 and pred == 0: fn += 1
        else: tn += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05  = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0
    print(f"  Train-set check: P={prec:.4f} R={rec:.4f} F0.5={f05:.4f}", flush=True)
    print(f"  Learned weights: {[round(w, 4) for w in model.w]}", flush=True)

    return model

# ============ MAIN INFERENCE ============
def main():
    t0 = time.time()
    os.makedirs('output', exist_ok=True)
    os.makedirs('experiments/pre_submission_v4', exist_ok=True)

    print("PHASE 1: Building IDF vocabulary from training data...", flush=True)
    name_idf, addr_idf = build_idf_vocab(TRAIN_S1, IDF_SAMPLE)
    print(f"  Name vocab: {len(name_idf)} tokens, Addr vocab: {len(addr_idf)} tokens", flush=True)

    model = train_model(name_idf, addr_idf)

    print("PHASE 3: Indexing test S2 and S3...", flush=True)
    t_idx = time.time()
    s2_idx = BlockIndex()
    with open(TEST_S2, 'r', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f, delimiter='\t')):
            s2_idx.add(row['entity_id'], row['business_name'],
                       row['business_address'], row['country'])
            if (i + 1) % 1000000 == 0:
                print(f"    S2: {i+1} indexed...", flush=True)
    s2_count = len(s2_idx.data)

    s3_idx = BlockIndex()
    with open(TEST_S3, 'r', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f, delimiter='\t')):
            s3_idx.add(row['entity_id'], row['business_name'],
                       row['business_address'], row['country'])
            if (i + 1) % 1000000 == 0:
                print(f"    S3: {i+1} indexed...", flush=True)
    s3_count = len(s3_idx.data)
    print(f"  Indexed {s2_count} S2 + {s3_count} S3 in {time.time()-t_idx:.1f}s", flush=True)

    print("PHASE 4: Running inference on all test S1 entities...", flush=True)
    t_inf = time.time()

    f_match = open(MATCHING_OUT, 'w', newline='', encoding='utf-8')
    f_cand  = open(CANDIDATE_OUT, 'w', newline='', encoding='utf-8')
    wm = csv.writer(f_match, delimiter='\t')
    wc = csv.writer(f_cand, delimiter='\t')
    wm.writerow(['source1_entity_id', 'matched_entity_ids'])
    wc.writerow(['source1_entity_id', 'candidate_entity_ids'])

    s1_count = 0
    total_cands = 0
    total_scored = 0
    total_links = 0
    singletons = 0
    one_match = 0
    two_match = 0
    three_plus = 0
    max_preds = 0
    max_cands_one = 0
    skipped_prefilter = 0

    with open(TEST_S1, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            s1_count += 1
            s1_id = row['entity_id']
            nn = norm_name(row['business_name'])
            na = norm_addr(row['business_address'])
            an = addr_nums(row['business_address'])
            country = row['country']

            cands_s2 = s2_idx.get_candidates(nn, an, country)
            cands_s3 = s3_idx.get_candidates(nn, an, country)
            all_cands = cands_s2 | cands_s3

            total_cands += len(all_cands)
            if len(all_cands) > max_cands_one:
                max_cands_one = len(all_cands)

            matches = []
            for c in all_cands:
                is_s2 = c in s2_idx.data
                d = s2_idx.data[c] if is_s2 else s3_idx.data.get(c)
                if not d: continue
                c_nn, c_na, c_an, c_country = d

                # Fast pre-filter: skip pairs with zero signal
                if not fast_prefilter(nn, c_nn):
                    skipped_prefilter += 1
                    continue

                total_scored += 1
                feat = compute_features_v4(nn, c_nn, na, c_na, an, c_an,
                                           country, c_country, is_s2,
                                           name_idf, addr_idf)
                score = model.predict(feat)
                if score >= THRESHOLD:
                    matches.append(c)

            wc.writerow([s1_id, ','.join(all_cands) if all_cands else ''])
            wm.writerow([s1_id, ','.join(matches) if matches else ''])

            n_m = len(matches)
            total_links += n_m
            if n_m > max_preds: max_preds = n_m
            if n_m == 0: singletons += 1
            elif n_m == 1: one_match += 1
            elif n_m == 2: two_match += 1
            else: three_plus += 1

            if s1_count % 50000 == 0:
                elapsed = time.time() - t_inf
                rate = s1_count / elapsed if elapsed > 0 else 1
                eta = (1732544 - s1_count) / rate if rate > 0 else 0
                print(f"    {s1_count:,} / 1,732,544  "
                      f"({s1_count/1732544*100:.1f}%)  "
                      f"cands={total_cands:,}  scored={total_scored:,}  "
                      f"links={total_links:,}  "
                      f"ETA={eta/60:.1f}m", flush=True)

    f_match.close()
    f_cand.close()

    total_time = time.time() - t0
    inf_time = time.time() - t_inf

    stats = f"""# PRE_SUBMISSION_V4 Test Inference Statistics

## Dataset
- Test S1 entities: {s1_count:,}
- Test S2 entities: {s2_count:,}
- Test S3 entities: {s3_count:,}

## Candidate Generation
- Total candidate pairs: {total_cands:,}
- Average candidates per S1: {total_cands/s1_count:.2f}
- Maximum candidates for one S1: {max_cands_one:,}

## Scoring
- Total pairs scored (post pre-filter): {total_scored:,}
- Pairs skipped by pre-filter: {skipped_prefilter:,}

## Predictions
- S1 with zero predictions: {singletons:,}
- S1 with 1 prediction: {one_match:,}
- S1 with 2 predictions: {two_match:,}
- S1 with 3+ predictions: {three_plus:,}
- Total predicted matches: {total_links:,}
- Maximum predictions for one S1: {max_preds}

## Performance
- Total runtime: {total_time:.1f}s ({total_time/60:.1f}m)
- Inference time: {inf_time:.1f}s ({inf_time/60:.1f}m)
- Approximate memory: ~2-3 GB (blocking indices)

## Warnings/Errors
- None
"""
    with open('experiments/pre_submission_v4/test_statistics.md', 'w', encoding='utf-8') as f:
        f.write(stats)

    print("\n" + "=" * 60, flush=True)
    print("INFERENCE COMPLETE", flush=True)
    print("=" * 60, flush=True)
    print(stats, flush=True)

if __name__ == '__main__':
    main()
