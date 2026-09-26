import glob
import re

for file in glob.glob('experiments/recovery_v2/phase*.py'):
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find and replace all csv.DictReader calls with tsv_dict_reader
    def replacer(match):
        return f"tsv_dict_reader(f.name)"
        
    # Pattern to match anything like csv.DictReader(f, ...)
    content = re.sub(r"csv\.DictReader\([^)]+\)", replacer, content)
    
    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)
