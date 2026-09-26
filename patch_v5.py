import os

with open('c:/Users/rk100/Desktop/ML CHALLENGE/code/business_entity_resolution/src/inference_v5.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update Threshold and N_FEATURES
content = content.replace('THRESHOLD = 0.675', 'THRESHOLD = 0.700')
content = content.replace('N_FEATURES = 21', 'N_FEATURES = 25')

# 2. Update build_idf_vocab
old_build = '''def build_idf_vocab(filepath, max_rows):
    name_df = Counter()
    addr_df = Counter()
    total = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\\t'):
            total += 1
            for t in set(norm_name(row['business_name']).split()):
                name_df[t] += 1
            for t in set(norm_addr(row['business_address']).split()):
                addr_df[t] += 1
            if total >= max_rows:
                break
    name_idf = {k: math.log((total + 1) / (v + 1)) + 1 for k, v in name_df.items()}
    addr_idf = {k: math.log((total + 1) / (v + 1)) + 1 for k, v in addr_df.items()}
    return name_idf, addr_idf'''

new_build = '''def build_idf_vocab(filepath, max_rows):
    name_df = Counter()
    addr_df = Counter()
    name_freq = Counter()
    total = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\\t'):
            total += 1
            nn = norm_name(row['business_name'])
            if nn: name_freq[nn] += 1
            for t in set(nn.split()):
                name_df[t] += 1
            for t in set(norm_addr(row['business_address']).split()):
                addr_df[t] += 1
            if total >= max_rows:
                break
    name_idf = {k: math.log((total + 1) / (v + 1)) + 1 for k, v in name_df.items()}
    addr_idf = {k: math.log((total + 1) / (v + 1)) + 1 for k, v in addr_df.items()}
    return name_idf, addr_idf, name_freq'''
content = content.replace(old_build, new_build)

# 3. Update compute_features_v4 to compute_features_v5
old_feat = '''def compute_features_v4(s1_nn, s2_nn, s1_na, s2_na, s1_an, s2_an,
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
    ]'''

new_feat = '''def compute_features_v5(s1_nn, s2_nn, s1_na, s2_na, s1_an, s2_an,
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

    # V5 Interaction features
    f5_1 = name_exact * addr_ratio
    f5_2 = name_exact * num_overlap
    f5_3 = name_exact * (1.0 if addr_ratio < 0.3 else 0.0)
    generic_name = 1.0 if name_freq.get(s1_nn, 0) > 5 else 0.0
    f5_4 = generic_name * addr_ratio

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
        f5_1,                   # 21
        f5_2,                   # 22
        f5_3,                   # 23
        f5_4,                   # 24
    ]'''
content = content.replace(old_feat, new_feat)
content = content.replace('compute_features_v4', 'compute_features_v5')

# 4. Update BlockIndex
old_block_init = '''class BlockIndex:
    def __init__(self):
        self.exact_name = defaultdict(list)
        self.addr_tok   = defaultdict(list)
        self.prefix_addr= defaultdict(list)
        self.char4      = defaultdict(list)
        self.data = {}'''

new_block_init = '''class BlockIndex:
    def __init__(self, addr_idf):
        self.exact_name = defaultdict(list)
        self.addr_tok   = defaultdict(list)
        self.prefix_addr= defaultdict(list)
        self.char4      = defaultdict(list)
        self.rare_addr  = defaultdict(list)
        self.addr_idf   = addr_idf
        self.data = {}'''
content = content.replace(old_block_init, new_block_init)

old_block_add = '''        if p4 and country:
            self.char4[(p4, country)].append(eid)

        self.data[eid] = (nn, na, an, country)'''

new_block_add = '''        if p4 and country:
            self.char4[(p4, country)].append(eid)
            
        for t in na.split():
            if self.addr_idf.get(t, 0) > 8.0 and country:
                self.rare_addr[(t, country)].append(eid)
                break

        self.data[eid] = (nn, na, an, country)'''
content = content.replace(old_block_add, new_block_add)

old_block_get = '''    def get_candidates(self, nn, an, country):'''
new_block_get = '''    def get_candidates(self, nn, na, an, country):'''
content = content.replace(old_block_get, new_block_get)

old_block_get_cands = '''        if p4 and country:
            block = self.char4.get((p4, country), [])
            if len(block) <= CHAR4_MAX_BLOCK:
                cands.update(block)
            # else: skip this massive block — too generic to be useful
        return cands'''

new_block_get_cands = '''        if p4 and country:
            block = self.char4.get((p4, country), [])
            if len(block) <= CHAR4_MAX_BLOCK:
                cands.update(block)
            # else: skip this massive block — too generic to be useful
            
        for t in na.split():
            if self.addr_idf.get(t, 0) > 8.0 and country:
                block = self.rare_addr.get((t, country), [])
                if len(block) <= 500:
                    cands.update(block)
                    
        return cands'''
content = content.replace(old_block_get_cands, new_block_get_cands)

# 5. Update train_model and main
content = content.replace('def train_model(name_idf, addr_idf):', 'def train_model(name_idf, addr_idf, name_freq):')

old_train_init = '''    s23_blocks_name = defaultdict(list)
    s23_blocks_addr = defaultdict(list)
    s23_blocks_p5   = defaultdict(list)
    s23_blocks_p4   = defaultdict(list)'''

new_train_init = '''    s23_blocks_name = defaultdict(list)
    s23_blocks_addr = defaultdict(list)
    s23_blocks_p5   = defaultdict(list)
    s23_blocks_p4   = defaultdict(list)
    s23_blocks_rare = defaultdict(list)'''
content = content.replace(old_train_init, new_train_init)

old_train_add = '''                if p4 in req_p4 and country:
                    s23_blocks_p4[(p4, country)].append(eid)
                    keep = True
                if keep:'''

new_train_add = '''                if p4 in req_p4 and country:
                    s23_blocks_p4[(p4, country)].append(eid)
                    keep = True
                for t in na.split():
                    if addr_idf.get(t, 0) > 8.0 and country:
                        s23_blocks_rare[(t, country)].append(eid)
                        keep = True
                        break
                if keep:'''
content = content.replace(old_train_add, new_train_add)

old_train_cand = '''        if p4 and country:
            block = s23_blocks_p4.get((p4, country), [])
            if len(block) <= CHAR4_MAX_BLOCK:
                cands.update(block)

        true_m = gt.get(s1_id, set())'''

new_train_cand = '''        if p4 and country:
            block = s23_blocks_p4.get((p4, country), [])
            if len(block) <= CHAR4_MAX_BLOCK:
                cands.update(block)
        for t in na.split():
            if addr_idf.get(t, 0) > 8.0 and country:
                block = s23_blocks_rare.get((t, country), [])
                if len(block) <= 500:
                    cands.update(block)

        true_m = gt.get(s1_id, set())'''
content = content.replace(old_train_cand, new_train_cand)

content = content.replace('name_idf, addr_idf)', 'name_idf, addr_idf, name_freq)')
content = content.replace('name_idf, addr_idf = build_idf_vocab(TRAIN_S1, IDF_SAMPLE)', 'name_idf, addr_idf, name_freq = build_idf_vocab(TRAIN_S1, IDF_SAMPLE)')
content = content.replace('model = train_model(name_idf, addr_idf)', 'model = train_model(name_idf, addr_idf, name_freq)')
content = content.replace('s2_idx = BlockIndex()', 's2_idx = BlockIndex(addr_idf)')
content = content.replace('s3_idx = BlockIndex()', 's3_idx = BlockIndex(addr_idf)')
content = content.replace('cands_s2 = s2_idx.get_candidates(nn, an, country)', 'cands_s2 = s2_idx.get_candidates(nn, na, an, country)')
content = content.replace('cands_s3 = s3_idx.get_candidates(nn, an, country)', 'cands_s3 = s3_idx.get_candidates(nn, na, an, country)')
content = content.replace('pre_submission_v4', 'pre_submission_v5')
content = content.replace('PRE_SUBMISSION_V4', 'PRE_SUBMISSION_V5')
content = content.replace('V4', 'V5')

with open('c:/Users/rk100/Desktop/ML CHALLENGE/code/business_entity_resolution/src/inference_v5.py', 'w', encoding='utf-8') as f:
    f.write(content)
