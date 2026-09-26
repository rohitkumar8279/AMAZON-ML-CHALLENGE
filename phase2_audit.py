import csv
import random
from collections import Counter

matching_file = 'experiments/score_recovery/previous_submission/matching_results.tsv'
cand_file = 'experiments/score_recovery/previous_submission/candidate_pairs.tsv'

cand_dict = {}
with open(cand_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        cands = [c for c in row['candidate_entity_ids'].split(',') if c.strip()] if 'candidate_entity_ids' in row else []
        cand_dict[row['source1_entity_id']] = set(cands)

total_s1 = 0
unique_s1 = set()
duplicate_s1 = 0
total_matches = 0
cardinality = Counter()
source_counts = Counter()
invalid_ids = 0
duplicate_ids_in_pred = 0
not_in_cand = 0

rows = []
zero_matches = []
high_matches = []

with open(matching_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for idx, row in enumerate(reader):
        s1 = row['source1_entity_id']
        matches_str = row['matched_entity_ids']
        matches = [m for m in matches_str.split(',') if m.strip()]
        
        total_s1 += 1
        if s1 in unique_s1:
            duplicate_s1 += 1
        unique_s1.add(s1)
        
        if idx < 20:
            rows.append((s1, matches_str))
            
        c = len(matches)
        cardinality[c] += 1
        total_matches += c
        
        if c == 0:
            zero_matches.append((s1, matches_str))
        elif c >= 4:
            high_matches.append((s1, matches_str))
            
        seen = set()
        for m in matches:
            if not m.startswith('S2-') and not m.startswith('S3-'):
                invalid_ids += 1
            if m.startswith('S2-'): source_counts['S2'] += 1
            if m.startswith('S3-'): source_counts['S3'] += 1
            if m in seen:
                duplicate_ids_in_pred += 1
            seen.add(m)
            if m not in cand_dict.get(s1, set()):
                not_in_cand += 1

print("=== OUTPUT SEMANTICS AUDIT ===")
print(f"Total S1 rows: {total_s1}")
print(f"Unique S1 rows: {len(unique_s1)}")
print(f"Duplicate S1 rows: {duplicate_s1}")
print(f"Total predicted matches: {total_matches}")
print(f"Total empty predictions: {cardinality[0]}")
print(f"Total 1-match entities: {cardinality[1]}")
print(f"Total 2-match entities: {cardinality[2]}")
print(f"Total 3-match entities: {cardinality[3]}")
print(f"Total 4+ match entities: {sum(v for k,v in cardinality.items() if k>=4)}")
print(f"S2 source: {source_counts['S2']}")
print(f"S3 source: {source_counts['S3']}")
print(f"Invalid IDs: {invalid_ids}")
print(f"Duplicate IDs within prediction: {duplicate_ids_in_pred}")
print(f"Matches not present in candidate_pairs: {not_in_cand}")

print("\n--- First 20 rows ---")
for r in rows: print(r)

print("\n--- 20 Random rows ---")
with open(matching_file, 'r', encoding='utf-8') as f:
    reader = list(csv.DictReader(f, delimiter='\t'))
    for r in random.sample(reader, min(20, len(reader))):
        print(r)

print("\n--- 20 High-match-count rows ---")
for r in high_matches[:20]: print(r)

print("\n--- 20 Zero-match rows ---")
for r in zero_matches[:20]: print(r)
