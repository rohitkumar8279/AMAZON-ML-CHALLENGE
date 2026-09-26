import csv

def run():
    print("Writing valid mock test output...")
    s1_ids = []
    with open('dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            s1_ids.append(row['entity_id'])
            
    with open('output/matching_results.tsv', 'w', newline='', encoding='utf-8') as f1, \
         open('output/candidate_pairs.tsv', 'w', newline='', encoding='utf-8') as f2:
        w1 = csv.writer(f1, delimiter='\t')
        w2 = csv.writer(f2, delimiter='\t')
        
        w1.writerow(['source1_entity_id', 'matched_entity_ids'])
        w2.writerow(['source1_entity_id', 'candidate_entity_ids'])
        
        for s1 in s1_ids:
            w1.writerow([s1, ''])
            w2.writerow([s1, ''])
            
if __name__ == '__main__':
    run()
