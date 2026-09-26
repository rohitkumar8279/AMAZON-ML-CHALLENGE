import glob
for f in glob.glob('experiments/recovery_v2/phase*.py'):
    content = open(f, 'r', encoding='utf-8').read()
    content = content.replace("'r', encoding='utf-8'", "'r', encoding='utf-8', newline=''")
    if 'csv.field_size_limit' not in content:
        content = content.replace("import csv", "import csv\nimport sys\ncsv.field_size_limit(2147483647)")
    open(f, 'w', encoding='utf-8').write(content)
