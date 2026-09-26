import os
import csv
import sys
from collections import defaultdict
import time
from normalization import normalize_business_name, normalize_business_address, extract_numbers_from_address

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            yield row

def load_ground_truth(file_path):
    gt = defaultdict(set)
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            s1_id = row['source1_entity_id']
            matches = row['matched_entity_ids']
            if matches:
                gt[s1_id] = set(matches.split(','))
            else:
                gt[s1_id] = set()
    return gt

def build_blocks(data_iter):
    exact_name_blocks = defaultdict(list)
    address_num_blocks = defaultdict(list)
    for row in data_iter:
        eid = row['entity_id']
        name = row['business_name']
        address = row['business_address']
        country = row['country']
        norm_name = normalize_business_name(name)
        addr_nums = extract_numbers_from_address(address)
        if norm_name and country:
            exact_name_blocks[(norm_name, country)].append(eid)
        name_tokens = norm_name.split()
        if addr_nums and name_tokens and country:
            first_token = name_tokens[0]
            if len(first_token) > 2: 
                address_num_blocks[(addr_nums, first_token, country)].append(eid)
    return exact_name_blocks, address_num_blocks

def generate_candidates(s1_data_iter, s2_indices, s3_indices):
    s2_exact_name, s2_addr_num = s2_indices
    s3_exact_name, s3_addr_num = s3_indices
    for row in s1_data_iter:
        s1_id = row['entity_id']
        name = row['business_name']
        address = row['business_address']
        country = row['country']
        norm_name = normalize_business_name(name)
        addr_nums = extract_numbers_from_address(address)
        name_tokens = norm_name.split()
        first_token = name_tokens[0] if name_tokens else ""
        candidates = set()
        if norm_name and country:
            candidates.update(s2_exact_name.get((norm_name, country), []))
            candidates.update(s3_exact_name.get((norm_name, country), []))
        if addr_nums and len(first_token) > 2 and country:
            candidates.update(s2_addr_num.get((addr_nums, first_token, country), []))
            candidates.update(s3_addr_num.get((addr_nums, first_token, country), []))
        yield s1_id, candidates
