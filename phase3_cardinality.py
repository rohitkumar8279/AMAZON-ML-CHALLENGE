import csv
from collections import Counter

train_gt = 'dataset/train/train_ground_truth.tsv'

# Read GT
true_card = Counter()
total_train_s1 = 0

with open(train_gt, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        s1 = row['source1_entity_id']
        matches_str = row['matched_entity_ids']
        matches = [m for m in matches_str.split(',') if m.strip()]
        true_card[len(matches)] += 1
        total_train_s1 += 1

true_pct = {k: v / total_train_s1 * 100 for k, v in true_card.items()}
true_3_plus = sum(v for k, v in true_card.items() if k >= 3)
true_3_plus_pct = true_3_plus / total_train_s1 * 100

print(f"Total Train S1: {total_train_s1}")
print(f"True 0-match: {true_card[0]} ({true_pct.get(0, 0):.2f}%)")
print(f"True 1-match: {true_card[1]} ({true_pct.get(1, 0):.2f}%)")
print(f"True 2-match: {true_card[2]} ({true_pct.get(2, 0):.2f}%)")
print(f"True 3+ match: {true_3_plus} ({true_3_plus_pct:.2f}%)")

total_pred_s1 = 1732544
pred_0 = 790023
pred_1 = 369151
pred_2 = 145948
pred_3_plus = 70378 + 357044

print(f"\nTotal Pred S1: {total_pred_s1}")
print(f"Pred 0-match: {pred_0} ({pred_0 / total_pred_s1 * 100:.2f}%)")
print(f"Pred 1-match: {pred_1} ({pred_1 / total_pred_s1 * 100:.2f}%)")
print(f"Pred 2-match: {pred_2} ({pred_2 / total_pred_s1 * 100:.2f}%)")
print(f"Pred 3+ match: {pred_3_plus} ({pred_3_plus / total_pred_s1 * 100:.2f}%)")
