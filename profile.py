import csv
import time
from collections import defaultdict

TEST_S1 = 'dataset/test/test_source1.tsv'

# Run 1000 records
def profile():
    t0 = time.time()
    count = 0
    with open(TEST_S1, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            count += 1
            if count > 1000:
                break
    print(f"Read 1000 rows in {time.time()-t0:.2f}s")
    
if __name__ == '__main__':
    profile()
