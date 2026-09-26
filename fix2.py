import glob
custom_reader = """
def tsv_dict_reader(filepath, max_rows=None):
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\\n').split('\\t')
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows: break
            yield dict(zip(header, line.rstrip('\\n').split('\\t')))
"""

for f in glob.glob('experiments/recovery_v2/phase*.py'):
    content = open(f, 'r', encoding='utf-8').read()
    
    # Inject custom reader
    if 'def tsv_dict_reader' not in content:
        content = content.replace("import sys", "import sys" + custom_reader)
        
    # Replace csv.DictReader calls with tsv_dict_reader
    # Example: csv.DictReader(f, delimiter='\t')
    import re
    content = re.sub(r'csv\.DictReader\([^)]+\)', r'tsv_dict_reader(src_file if "src_file" in locals() else f.name)', content)
    
    # We will just do a simpler string replacement for the specific loop patterns
    content = content.replace("for i, row in enumerate(csv.DictReader(f, delimiter='\\t')):", "for i, row in enumerate(tsv_dict_reader(f.name)):")
    content = content.replace("for row in csv.DictReader(f, delimiter='\\t'):", "for row in tsv_dict_reader(f.name):")
    
    open(f, 'w', encoding='utf-8').write(content)
