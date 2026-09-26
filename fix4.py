import glob

custom_reader = """
def tsv_dict_reader(filepath, max_rows=None):
    with open(filepath, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip('\\r\\n').split('\\t')
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows: break
            yield dict(zip(header, line.decode('utf-8').rstrip('\\r\\n').split('\\t')))
"""

for file in glob.glob('experiments/recovery_v2/phase*.py'):
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the old definition
    import re
    # We want to replace the old definition of tsv_dict_reader
    content = re.sub(r"def tsv_dict_reader.*?yield dict\(zip\(header, line\.rstrip\('\\n'\)\.split\('\\t'\)\)\)", custom_reader.strip(), content, flags=re.DOTALL)
    
    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)
