import csv
import time
from collections import defaultdict

TEST_S1 = 'dataset/test/test_source1.tsv'
TEST_S2 = 'dataset/test/test_source2.tsv'
TEST_S3 = 'dataset/test/test_source3.tsv'
MATCHING_OUT = 'output/matching_results.tsv'
CANDIDATE_OUT = 'output/candidate_pairs.tsv'

def normalize(name):
    if not name: return ""
    return name.lower().strip()

def main():
    t0 = time.time()
    print("Building exact match index...")
    
    idx = defaultdict(list)
    
    with open(TEST_S2, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            key = (normalize(row['business_name']), row['country'])
            if key[0] and key[1]: idx[key].append(row['entity_id'])
            
    with open(TEST_S3, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            key = (normalize(row['business_name']), row['country'])
            if key[0] and key[1]: idx[key].append(row['entity_id'])
            
    print("Index built. Running inference...")
    
    s1_count = 0
    with open(TEST_S1, 'r', encoding='utf-8') as fin, \
         open(MATCHING_OUT, 'w', newline='', encoding='utf-8') as fout, \
         open(CANDIDATE_OUT, 'w', newline='', encoding='utf-8') as fcan:
         
        w = csv.writer(fout, delimiter='\t')
        w.writerow(['source1_entity_id', 'matched_entity_ids'])
        
        wc = csv.writer(fcan, delimiter='\t')
        wc.writerow(['source1_entity_id', 'candidate_entity_ids'])
        
        for row in csv.DictReader(fin, delimiter='\t'):
            s1_id = row['entity_id']
            key = (normalize(row['business_name']), row['country'])
            
            matches = idx.get(key, [])
            w.writerow([s1_id, ','.join(matches)])
            wc.writerow([s1_id, ','.join(matches)])
            
            s1_count += 1
            if s1_count % 200000 == 0:
                print(f"Processed {s1_count} S1 entities...")
                
    print(f"Done processing {s1_count} S1 entities in {time.time()-t0:.1f}s.")

if __name__ == '__main__':
    main()
