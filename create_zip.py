import zipfile
import os

def create_zip():
    print("Creating Antigravity_submission.zip...")
    with zipfile.ZipFile('Antigravity_submission.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add output files
        zf.write('output/matching_results.tsv', 'output/matching_results.tsv')
        # Only output files included per request
        
    print("ZIP created successfully.")

if __name__ == '__main__':
    create_zip()
