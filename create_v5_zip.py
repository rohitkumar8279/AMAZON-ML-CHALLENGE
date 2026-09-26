import zipfile
import os
import shutil

def create_zip():
    zip_name = 'AmazonML_PreSubmission_V5_REAL.zip'
    print(f"Creating {zip_name}...")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add output files
        if os.path.exists('output/matching_results.tsv'):
            zf.write('output/matching_results.tsv', 'output/matching_results.tsv')
        else:
            print("WARNING: matching_results.tsv missing")
            
        if os.path.exists('output/candidate_pairs.tsv'):
            zf.write('output/candidate_pairs.tsv', 'output/candidate_pairs.tsv')
        else:
            print("WARNING: candidate_pairs.tsv missing")
            
        # Add code files
        code_dir = 'code/business_entity_resolution'
        for root, dirs, files in os.walk(code_dir):
            if '__pycache__' in root: continue
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.pyc'): continue
                arcname = file_path.replace('\\', '/')
                zf.write(file_path, arcname)
                
        # Add documentation
        if os.path.exists('Documentation_template.md'):
            zf.write('Documentation_template.md', 'Documentation_template.md')
        
    print("ZIP created successfully.")

    # Create frozen copy of the experiment
    frozen_dir = 'experiments/pre_submission_v5_real_frozen'
    os.makedirs(frozen_dir, exist_ok=True)
    if os.path.exists('experiments/pre_submission_v5/test_statistics.md'):
        shutil.copy('experiments/pre_submission_v5/test_statistics.md', frozen_dir)
        
    # Copy code snapshot
    shutil.copy('code/business_entity_resolution/src/inference_v5.py', frozen_dir)
    print("Frozen snapshot created.")

if __name__ == '__main__':
    create_zip()
