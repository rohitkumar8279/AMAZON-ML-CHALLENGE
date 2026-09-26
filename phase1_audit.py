import pandas as pd
import numpy as np
from collections import defaultdict
import os

os.makedirs('experiments/v6_2', exist_ok=True)

print("Reading ground truth...")
gt_path = 'dataset/train/train_ground_truth.tsv'

# Read line by line to handle varying number of matches, or just use pandas if format is strictly 2 columns
df = pd.read_csv(gt_path, sep='\t', header=0, names=['source1_entity_id', 'matched_entity_ids'], dtype=str)

# Handle empty matches just in case
df['matched_entity_ids'] = df['matched_entity_ids'].fillna('')

total_matches = []
s2_matches = []
s3_matches = []
s2s3_to_s1 = defaultdict(list)

for _, row in df.iterrows():
    s1_id = row['source1_entity_id']
    matches = str(row['matched_entity_ids']).split(',') if row['matched_entity_ids'] else []
    matches = [m.strip() for m in matches if m.strip()]
    
    total = len(matches)
    s2_count = sum(1 for m in matches if m.startswith('S2-'))
    s3_count = sum(1 for m in matches if m.startswith('S3-'))
    
    total_matches.append(total)
    s2_matches.append(s2_count)
    s3_matches.append(s3_count)
    
    for m in matches:
        s2s3_to_s1[m].append(s1_id)

total_matches = np.array(total_matches)
s2_matches = np.array(s2_matches)
s3_matches = np.array(s3_matches)

zero_match = np.sum(total_matches == 0)
one_match = np.sum(total_matches == 1)
two_match = np.sum(total_matches == 2)
three_plus_match = np.sum(total_matches >= 3)

s2s3_counts = np.array([len(v) for v in s2s3_to_s1.values()])

if len(s2s3_counts) > 0:
    max_s1_per_s2s3 = np.max(s2s3_counts)
    mean_s1_per_s2s3 = np.mean(s2s3_counts)
    p99_s1_per_s2s3 = np.percentile(s2s3_counts, 99)
    frac_gt_1 = np.sum(s2s3_counts > 1) / len(s2s3_counts)
else:
    max_s1_per_s2s3 = mean_s1_per_s2s3 = p99_s1_per_s2s3 = frac_gt_1 = 0

with open('experiments/v6_2/cardinality_audit.tsv', 'w') as f:
    f.write(f"zero_match_S1\t{zero_match}\n")
    f.write(f"one_match_S1\t{one_match}\n")
    f.write(f"two_match_S1\t{two_match}\n")
    f.write(f"three_plus_match_S1\t{three_plus_match}\n")
    f.write(f"total_matches_mean\t{np.mean(total_matches):.4f}\n")
    f.write(f"total_matches_max\t{np.max(total_matches)}\n")
    f.write(f"s2_matches_mean\t{np.mean(s2_matches):.4f}\n")
    f.write(f"s2_matches_max\t{np.max(s2_matches)}\n")
    f.write(f"s3_matches_mean\t{np.mean(s3_matches):.4f}\n")
    f.write(f"s3_matches_max\t{np.max(s3_matches)}\n")
    f.write(f"s2s3_max_s1\t{max_s1_per_s2s3}\n")
    f.write(f"s2s3_mean_s1\t{mean_s1_per_s2s3:.4f}\n")
    f.write(f"s2s3_p99_s1\t{p99_s1_per_s2s3:.4f}\n")
    f.write(f"s2s3_fraction_gt1_s1\t{frac_gt_1:.6f}\n")
    if frac_gt_1 == 0:
        f.write("EMPIRICALLY VERIFIED TRAINING STRUCTURE\n")

print("Done. Saved to experiments/v6_2/cardinality_audit.tsv")
