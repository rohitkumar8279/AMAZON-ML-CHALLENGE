import csv
import numpy as np

def compute():
    print("Computing stats on final candidate...")
    cand_file = 'experiments/score_recovery/final_candidate/candidate_pairs.tsv'
    match_file = 'experiments/score_recovery/final_candidate/matching_results.tsv'
    
    cand_counts = []
    with open(cand_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            c = row['candidate_entity_ids']
            cnt = len([x for x in c.split(',') if x.strip()]) if c else 0
            cand_counts.append(cnt)
            
    match_counts = []
    with open(match_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            m = row['matched_entity_ids']
            cnt = len([x for x in m.split(',') if x.strip()]) if m else 0
            match_counts.append(cnt)
            
    print(f"Candidate count: {sum(cand_counts)}")
    print(f"Avg candidates/entity: {np.mean(cand_counts):.2f}")
    print(f"P95: {np.percentile(cand_counts, 95)}")
    print(f"P99: {np.percentile(cand_counts, 99)}")
    print(f"Max: {np.max(cand_counts)}")
    
    c = Counter(match_counts)
    print(f"\nPredicted 0-match: {c[0]}")
    print(f"Predicted 1-match: {c[1]}")
    print(f"Predicted 2-match: {c[2]}")
    print(f"Predicted 3+: {sum(v for k,v in c.items() if k>=3)}")
    print(f"Total predictions: {sum(match_counts)}")

if __name__ == '__main__':
    from collections import Counter
    compute()
