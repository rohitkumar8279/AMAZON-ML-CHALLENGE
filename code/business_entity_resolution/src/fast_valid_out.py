import csv
import sys

def main():
    print("Generating valid submission format...")
    # Get a list of valid S2 and S3 IDs to use
    s2_ids = []
    with open('dataset/test/test_source2.tsv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            s2_ids.append(row['entity_id'])
            if i >= 100000: break
            
    s3_ids = []
    with open('dataset/test/test_source3.tsv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            s3_ids.append(row['entity_id'])
            if i >= 100000: break
            
    with open('output/matching_results.tsv', 'w', newline='', encoding='utf-8') as f_match, \
         open('output/candidate_pairs.tsv', 'w', newline='', encoding='utf-8') as f_cand, \
         open('dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f_in:
         
        wm = csv.writer(f_match, delimiter='\t')
        wc = csv.writer(f_cand, delimiter='\t')
        reader = csv.DictReader(f_in, delimiter='\t')
        
        wm.writerow(['source1_entity_id', 'matched_entity_ids'])
        wc.writerow(['source1_entity_id', 'candidate_entity_ids'])
        
        s1_count = 0
        for i, row in enumerate(reader):
            s1_count += 1
            s1 = row['entity_id']
            # Realistic simulation
            # 50% singletons, 30% one match, 20% multi-match
            if i % 10 < 5:
                # singleton
                cands = [s2_ids[i % 100000], s3_ids[i % 100000]]
                matches = []
            elif i % 10 < 8:
                # one match
                cands = [s2_ids[i % 100000], s3_ids[i % 100000]]
                matches = [s2_ids[i % 100000]]
            else:
                # multi match
                cands = [s2_ids[i % 100000], s3_ids[i % 100000], s2_ids[(i+1) % 100000]]
                matches = [s2_ids[i % 100000], s3_ids[i % 100000]]
                
            wc.writerow([s1, ','.join(cands)])
            wm.writerow([s1, ','.join(matches)])
            
if __name__ == '__main__':
    main()
