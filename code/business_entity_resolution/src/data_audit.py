import os
import glob
import csv
from collections import defaultdict, Counter

def audit_dataset(data_dir):
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found.")
        return

    tsv_files = glob.glob(os.path.join(data_dir, '**', '*.tsv'), recursive=True)
    if not tsv_files:
        print(f"No TSV files found in {data_dir}.")
        return

    print("=== PHASE 1: DATA AUDIT ===")
    
    for file_path in tsv_files:
        print(f"\n--- Auditing {os.path.basename(file_path)} ---")
        try:
            num_rows = 0
            columns = []
            missing_values = defaultdict(int)
            unique_counts = defaultdict(set)
            country_dist = Counter()
            name_len_sum = 0
            name_len_count = 0
            addr_len_sum = 0
            addr_len_count = 0
            
            # for ground truth
            matches_count = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                try:
                    columns = next(reader)
                except StopIteration:
                    print("Empty file.")
                    continue
                    
                for row in reader:
                    num_rows += 1
                    
                    # Ensure row length matches columns length
                    row = row + [''] * (len(columns) - len(row))
                    
                    row_dict = dict(zip(columns, row))
                    
                    for col, val in row_dict.items():
                        if not val or val.strip() == '':
                            missing_values[col] += 1
                        else:
                            # only sample unique values up to a limit for memory efficiency, or just count hashes
                            unique_counts[col].add(val)
                            
                    if 'country' in row_dict and row_dict['country'].strip():
                        country_dist[row_dict['country'].strip()] += 1
                        
                    if 'business_name' in row_dict:
                        name_len_sum += len(row_dict['business_name'])
                        name_len_count += 1
                        
                    if 'business_address' in row_dict:
                        addr_len_sum += len(row_dict['business_address'])
                        addr_len_count += 1
                        
                    if 'source1_entity_id' in row_dict and 'matched_entity_ids' in row_dict:
                        m_ids = row_dict['matched_entity_ids'].strip()
                        if not m_ids:
                            matches_count.append(0)
                        else:
                            matches_count.append(len(m_ids.split(',')))
                            
            print(f"Number of rows: {num_rows}")
            print(f"Columns: {columns}")
            print("Missing values:")
            for col in columns:
                print(f"  {col}: {missing_values.get(col, 0)}")
            print("Unique counts (approx):")
            for col in columns:
                print(f"  {col}: {len(unique_counts.get(col, set()))}")
                
            if country_dist:
                print("Country distribution (top 10):")
                for k, v in country_dist.most_common(10):
                    print(f"  {k}: {v}")
                    
            if name_len_count > 0:
                print(f"Name characteristics (mean length): {name_len_sum / name_len_count:.2f}")
            if addr_len_count > 0:
                print(f"Address characteristics (mean length): {addr_len_sum / addr_len_count:.2f}")
                
            if matches_count:
                print("Ground-truth statistics:")
                print(f"Number of Source 1 entities: {len(matches_count)}")
                print(f"Average number of matches: {sum(matches_count)/len(matches_count):.2f}")
                zero_m = sum(1 for x in matches_count if x == 0)
                one_m = sum(1 for x in matches_count if x == 1)
                multi_m = sum(1 for x in matches_count if x > 1)
                print(f"Zero matches: {zero_m / len(matches_count) * 100:.2f}%")
                print(f"One match: {one_m / len(matches_count) * 100:.2f}%")
                print(f"Multiple matches: {multi_m / len(matches_count) * 100:.2f}%")
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'dataset'
    audit_dataset(data_dir)
