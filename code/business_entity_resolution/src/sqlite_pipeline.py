import sqlite3
import csv
import os
import time
from collections import defaultdict
from normalization import normalize_business_name, normalize_business_address, extract_numbers_from_address

def create_db(data_dir):
    db_path = 'dataset.db'
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create tables
    for tbl in ['s1', 's2', 's3']:
        c.execute(f'''
            CREATE TABLE {tbl} (
                entity_id TEXT PRIMARY KEY,
                business_name TEXT,
                business_address TEXT,
                country TEXT,
                norm_name TEXT,
                addr_nums TEXT,
                first_token TEXT
            )
        ''')
        
    conn.commit()
    return conn

def load_data_to_db(conn, data_dir):
    c = conn.cursor()
    
    def process_file(file_path, table_name):
        print(f"Loading {file_path} into {table_name}...")
        batch = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                name = row['business_name']
                addr = row['business_address']
                norm_name = normalize_business_name(name)
                addr_nums = extract_numbers_from_address(addr)
                name_tokens = norm_name.split()
                first_token = name_tokens[0] if name_tokens else ""
                
                batch.append((
                    row['entity_id'], name, addr, row['country'],
                    norm_name, addr_nums, first_token
                ))
                
                if len(batch) >= 100000:
                    c.executemany(f'INSERT INTO {table_name} VALUES (?,?,?,?,?,?,?)', batch)
                    batch = []
        if batch:
            c.executemany(f'INSERT INTO {table_name} VALUES (?,?,?,?,?,?,?)', batch)
        conn.commit()
        
    process_file(f'{data_dir}/train/train_source1.tsv', 's1')
    process_file(f'{data_dir}/train/train_source2.tsv', 's2')
    process_file(f'{data_dir}/train/train_source3.tsv', 's3')
    
    print("Creating indexes...")
    for tbl in ['s2', 's3']:
        c.execute(f'CREATE INDEX idx_{tbl}_name_country ON {tbl}(norm_name, country)')
        c.execute(f'CREATE INDEX idx_{tbl}_addr_first_country ON {tbl}(addr_nums, first_token, country)')
    conn.commit()

def evaluate_blocking(conn, data_dir):
    c = conn.cursor()
    
    # Load GT
    gt = defaultdict(set)
    with open(f'{data_dir}/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            m = row['matched_entity_ids']
            if m:
                gt[row['source1_entity_id']] = set(m.split(','))
                
    # Evaluate a sample of S1 to be fast
    c.execute("SELECT entity_id, norm_name, addr_nums, first_token, country FROM s1 LIMIT 100000")
    s1_sample = c.fetchall()
    
    total_true = 0
    total_found = 0
    total_cands = 0
    
    print("Evaluating V0 Baseline Blocking...")
    t0 = time.time()
    
    for row in s1_sample:
        eid, norm_name, addr_nums, first_token, country = row
        true_matches = gt.get(eid, set())
        total_true += len(true_matches)
        
        cands = set()
        
        if norm_name and country:
            # S2
            c.execute('SELECT entity_id FROM s2 WHERE norm_name=? AND country=?', (norm_name, country))
            cands.update([r[0] for r in c.fetchall()])
            # S3
            c.execute('SELECT entity_id FROM s3 WHERE norm_name=? AND country=?', (norm_name, country))
            cands.update([r[0] for r in c.fetchall()])
            
        if addr_nums and len(first_token) > 2 and country:
            c.execute('SELECT entity_id FROM s2 WHERE addr_nums=? AND first_token=? AND country=?', (addr_nums, first_token, country))
            cands.update([r[0] for r in c.fetchall()])
            c.execute('SELECT entity_id FROM s3 WHERE addr_nums=? AND first_token=? AND country=?', (addr_nums, first_token, country))
            cands.update([r[0] for r in c.fetchall()])
            
        total_found += len(true_matches.intersection(cands))
        total_cands += len(cands)
        
    print(f"Evaluated 100k S1 entities in {time.time()-t0:.2f}s")
    
    recall = (total_found / total_true) * 100 if total_true else 0
    print(f"V0 Blocking Recall: {recall:.2f}%")
    print(f"Total True Matches in Sample: {total_true}")
    print(f"Found Matches: {total_found}")
    print(f"Total Candidates Generated: {total_cands}")
    print(f"Avg Candidates per S1: {total_cands / 100000:.2f}")
    
    # Extract missed matches
    print("Extracting missed matches...")
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/blocking_miss_analysis.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source1_entity_id', 'true_entity_id', 'source', 'name_s1', 'name_candidate', 'address_s1', 'address_candidate', 'country_s1', 'country_candidate', 'reason'])
        
        missed_count = 0
        for row in s1_sample:
            if missed_count > 500: break # just get a good sample for diagnosis
            
            eid, norm_name, addr_nums, first_token, country = row
            true_matches = gt.get(eid, set())
            
            cands = set()
            if norm_name and country:
                c.execute('SELECT entity_id FROM s2 WHERE norm_name=? AND country=?', (norm_name, country))
                cands.update([r[0] for r in c.fetchall()])
                c.execute('SELECT entity_id FROM s3 WHERE norm_name=? AND country=?', (norm_name, country))
                cands.update([r[0] for r in c.fetchall()])
            if addr_nums and len(first_token) > 2 and country:
                c.execute('SELECT entity_id FROM s2 WHERE addr_nums=? AND first_token=? AND country=?', (addr_nums, first_token, country))
                cands.update([r[0] for r in c.fetchall()])
                c.execute('SELECT entity_id FROM s3 WHERE addr_nums=? AND first_token=? AND country=?', (addr_nums, first_token, country))
                cands.update([r[0] for r in c.fetchall()])
                
            missed = true_matches - cands
            if missed:
                # get s1 details
                c.execute('SELECT business_name, business_address, country FROM s1 WHERE entity_id=?', (eid,))
                s1_det = c.fetchone()
                for m in missed:
                    c.execute('SELECT business_name, business_address, country FROM s2 WHERE entity_id=?', (m,))
                    s2_det = c.fetchone()
                    source = 'S2'
                    if not s2_det:
                        c.execute('SELECT business_name, business_address, country FROM s3 WHERE entity_id=?', (m,))
                        s2_det = c.fetchone()
                        source = 'S3'
                        
                    if s2_det:
                        writer.writerow([eid, m, source, s1_det[0], s2_det[0], s1_det[1], s2_det[1], s1_det[2], s2_det[2], 'missed by V0 blocks'])
                        missed_count += 1

if __name__ == '__main__':
    conn = create_db('dataset')
    load_data_to_db(conn, 'dataset')
    evaluate_blocking(conn, 'dataset')
