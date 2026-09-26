import csv
from collections import Counter
pred_file = 'experiments/score_recovery/val_pred_42.tsv'
card = Counter()
total = 0
with open(pred_file, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        m = row['matched_entity_ids']
        matches = [x for x in m.split(',') if x.strip()]
        card[len(matches)] += 1
        total += 1
print("Validation Predicted Cardinality:")
print(f"Total S1: {total}")
for i in range(4):
    print(f"{i}-match: {card[i]} ({card[i]/total*100:.2f}%)")
print(f"4+ match: {sum(v for k,v in card.items() if k>=4)} ({sum(v for k,v in card.items() if k>=4)/total*100:.2f}%)")
