import sqlite3
import csv
import os
import time

def setup_db(db_path, data_dir):
    print("Setting up GT table in SQLite...")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS gt')
    c.execute('CREATE TABLE gt (s1_id TEXT, s23_id TEXT)')
    
    batch = []
    with open(f'{data_dir}/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            s1_id = row['source1_entity_id']
            matches = row['matched_entity_ids']
            if matches:
                for m in matches.split(','):
                    batch.append((s1_id, m))
                    
    c.executemany('INSERT INTO gt VALUES (?, ?)', batch)
    c.execute('CREATE INDEX idx_gt_s1 ON gt(s1_id)')
    c.execute('CREATE INDEX idx_gt_pair ON gt(s1_id, s23_id)')
    conn.commit()
    return conn

def run_v0_blocking(conn):
    print("Running V0 Blocking...")
    c = conn.cursor()
    t0 = time.time()
    
    c.execute('DROP TABLE IF EXISTS v0_cands')
    # Using rowid % 10 < 2 to subsample 20% of S1 for validation? No, the prompt says "evaluate locally... entity level".
    # Since SQLite is fast, let's just do 20% validation split for the blocking evaluation.
    # To keep seed deterministic, we can use a WHERE clause based on hashing, or just `abs(random())`. 
    # But for blocking recall, let's just do the whole dataset for V0/V1 since it's just SQL joins!
    # The user asked for "Candidate Count" and "Blocking Recall", which is easier to measure on the whole train set.
    
    query = """
    CREATE TABLE v0_cands AS
    SELECT s1.entity_id as s1_id, s2.entity_id as s23_id
    FROM s1 JOIN s2 ON s1.norm_name = s2.norm_name AND s1.country = s2.country
    WHERE s1.norm_name != ''
    UNION
    SELECT s1.entity_id as s1_id, s3.entity_id as s23_id
    FROM s1 JOIN s3 ON s1.norm_name = s3.norm_name AND s1.country = s3.country
    WHERE s1.norm_name != ''
    UNION
    SELECT s1.entity_id as s1_id, s2.entity_id as s23_id
    FROM s1 JOIN s2 ON s1.addr_nums = s2.addr_nums AND s1.first_token = s2.first_token AND s1.country = s2.country
    WHERE s1.addr_nums != '' AND length(s1.first_token) > 2
    UNION
    SELECT s1.entity_id as s1_id, s3.entity_id as s23_id
    FROM s1 JOIN s3 ON s1.addr_nums = s3.addr_nums AND s1.first_token = s3.first_token AND s1.country = s3.country
    WHERE s1.addr_nums != '' AND length(s1.first_token) > 2;
    """
    c.execute(query)
    c.execute('CREATE INDEX idx_v0_s1 ON v0_cands(s1_id)')
    c.execute('CREATE INDEX idx_v0_pair ON v0_cands(s1_id, s23_id)')
    conn.commit()
    print(f"V0 Blocking complete in {time.time()-t0:.2f}s")
    
    # Metrics
    c.execute('SELECT COUNT(*) FROM v0_cands')
    v0_cand_count = c.fetchone()[0]
    
    c.execute('SELECT COUNT(DISTINCT entity_id) FROM s1')
    total_s1 = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM gt')
    total_true = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM v0_cands c JOIN gt ON c.s1_id = gt.s1_id AND c.s23_id = gt.s23_id')
    v0_found = c.fetchone()[0]
    
    v0_recall = (v0_found / total_true) * 100 if total_true else 0
    print(f"V0 Candidate count: {v0_cand_count}")
    print(f"V0 Avg per S1: {v0_cand_count / total_s1:.2f}")
    print(f"V0 True Matches Found: {v0_found} / {total_true}")
    print(f"V0 Blocking Recall: {v0_recall:.2f}%\n")
    
    return v0_cand_count, v0_found, v0_recall, total_true, total_s1

def analyze_missed_matches(conn):
    print("Extracting V0 Missed Matches...")
    c = conn.cursor()
    # Missed matches are in GT but not in v0_cands
    query = """
    SELECT gt.s1_id, gt.s23_id
    FROM gt
    LEFT JOIN v0_cands c ON gt.s1_id = c.s1_id AND gt.s23_id = c.s23_id
    WHERE c.s1_id IS NULL
    LIMIT 200;
    """
    c.execute(query)
    missed = c.fetchall()
    
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/blocking_miss_analysis.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source1_entity_id', 'true_entity_id', 'source', 'name_s1', 'name_candidate', 'address_s1', 'address_candidate', 'country_s1', 'country_candidate', 'reason'])
        
        for s1_id, s23_id in missed:
            c.execute('SELECT business_name, business_address, country FROM s1 WHERE entity_id=?', (s1_id,))
            s1_row = c.fetchone()
            if not s1_row: continue
            
            c.execute('SELECT business_name, business_address, country FROM s2 WHERE entity_id=?', (s23_id,))
            s23_row = c.fetchone()
            source = 'S2'
            if not s23_row:
                c.execute('SELECT business_name, business_address, country FROM s3 WHERE entity_id=?', (s23_id,))
                s23_row = c.fetchone()
                source = 'S3'
                
            if s23_row:
                writer.writerow([s1_id, s23_id, source, s1_row[0], s23_row[0], s1_row[1], s23_row[1], s1_row[2], s23_row[2], 'Investigate diff'])

def run_v1_blocking(conn):
    print("Running V1 Blocking...")
    c = conn.cursor()
    t0 = time.time()
    
    # Let's add new columns to S1, S2, S3 for V1 blocking rules if needed.
    # We will use substr(norm_name, 1, 5) as prefix block, and lengths.
    c.execute('DROP TABLE IF EXISTS v1_cands')
    query = """
    CREATE TABLE v1_cands AS
    SELECT * FROM v0_cands
    UNION
    SELECT s1.entity_id as s1_id, s2.entity_id as s23_id
    FROM s1 JOIN s2 ON substr(s1.norm_name, 1, 5) = substr(s2.norm_name, 1, 5) AND s1.addr_nums = s2.addr_nums AND s1.country = s2.country
    WHERE s1.addr_nums != '' AND length(s1.norm_name) >= 5
    UNION
    SELECT s1.entity_id as s1_id, s3.entity_id as s23_id
    FROM s1 JOIN s3 ON substr(s1.norm_name, 1, 5) = substr(s3.norm_name, 1, 5) AND s1.addr_nums = s3.addr_nums AND s1.country = s3.country
    WHERE s1.addr_nums != '' AND length(s1.norm_name) >= 5;
    """
    c.execute(query)
    c.execute('CREATE INDEX idx_v1_pair ON v1_cands(s1_id, s23_id)')
    conn.commit()
    print(f"V1 Blocking complete in {time.time()-t0:.2f}s")
    
    c.execute('SELECT COUNT(*) FROM v1_cands')
    v1_cand_count = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM gt')
    total_true = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM v1_cands c JOIN gt ON c.s1_id = gt.s1_id AND c.s23_id = gt.s23_id')
    v1_found = c.fetchone()[0]
    
    c.execute('SELECT COUNT(DISTINCT s1_id) FROM s1')
    total_s1 = c.fetchone()[0]
    
    v1_recall = (v1_found / total_true) * 100 if total_true else 0
    print(f"V1 Candidate count: {v1_cand_count}")
    print(f"V1 Avg per S1: {v1_cand_count / total_s1:.2f}")
    print(f"V1 True Matches Found: {v1_found} / {total_true}")
    print(f"V1 Blocking Recall: {v1_recall:.2f}%\n")
    
    return v1_cand_count, v1_found, v1_recall

def extract_hard_negatives(conn):
    print("Extracting Hard Negatives...")
    c = conn.cursor()
    # Hard negative: in v1_cands but NOT in gt, and exact name matches!
    query = """
    SELECT c.s1_id, c.s23_id
    FROM v1_cands c
    LEFT JOIN gt ON c.s1_id = gt.s1_id AND c.s23_id = gt.s23_id
    JOIN s1 ON c.s1_id = s1.entity_id
    JOIN s2 ON c.s23_id = s2.entity_id
    WHERE gt.s1_id IS NULL AND s1.norm_name = s2.norm_name
    LIMIT 200;
    """
    c.execute(query)
    negatives = c.fetchall()
    
    with open('experiments/hard_negatives.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source1_entity_id', 'candidate_entity_id', 'source', 'name_s1', 'name_candidate', 'address_s1', 'address_candidate', 'country_s1', 'country_candidate'])
        for s1_id, s23_id in negatives:
            c.execute('SELECT business_name, business_address, country FROM s1 WHERE entity_id=?', (s1_id,))
            s1_row = c.fetchone()
            c.execute('SELECT business_name, business_address, country FROM s2 WHERE entity_id=?', (s23_id,))
            s23_row = c.fetchone()
            if s1_row and s23_row:
                writer.writerow([s1_id, s23_id, 'S2', s1_row[0], s23_row[0], s1_row[1], s23_row[1], s1_row[2], s23_row[2]])

if __name__ == '__main__':
    conn = sqlite3.connect('dataset.db')
    # setup_db('dataset.db', 'dataset')
    # run_v0_blocking(conn)
    analyze_missed_matches(conn)
    run_v1_blocking(conn)
    extract_hard_negatives(conn)
