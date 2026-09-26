import sqlite3

def run():
    conn = sqlite3.connect('dataset.db')
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM v0_cands')
    v0_cand_count = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM gt')
    total_true = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM v0_cands c JOIN gt ON c.s1_id = gt.s1_id AND c.s23_id = gt.s23_id')
    v0_found = c.fetchone()[0]
    
    c.execute('SELECT COUNT(DISTINCT entity_id) FROM s1')
    total_s1 = c.fetchone()[0]
    
    v0_recall = (v0_found / total_true) * 100 if total_true else 0
    print(f"V0 Candidate count: {v0_cand_count}")
    print(f"V0 Avg per S1: {v0_cand_count / total_s1:.2f}")
    print(f"V0 True Matches Found: {v0_found} / {total_true}")
    print(f"V0 Blocking Recall: {v0_recall:.2f}%\n")

if __name__ == '__main__':
    run()
