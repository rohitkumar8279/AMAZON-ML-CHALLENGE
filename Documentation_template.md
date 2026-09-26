# Amazon ML Challenge 2026 — Pre-Submission V4 (Real)

## 1. Problem
Business Entity Resolution: Matching Source-1 business entities to their corresponding records in Source-2 and Source-3.

## 2. Approach
A pure-Python memory-efficient pipeline that uses structured text blocking, fast token/character pre-filtering, and an SGD Logistic Regression model over 21 text similarity and interaction features.

## 3. Candidate Generation
Candidates are retrieved from SQLite-like in-memory dictionary blocks using four rules (capping the 4-char rule at 500 to avoid candidate explosion):
1. Exact normalized name + country
2. Address numbers + first token + country
3. 5-char name prefix + address numbers + country
4. 4-char name prefix + country (capped at 500)

## 4. Feature Engineering
We compute 21 features per candidate pair, including:
- Jaccard and Sorensen-Dice similarities on names and addresses
- IDF-weighted token overlap (vocabulary built strictly from 200k training rows: 46,699 name tokens, 111,881 address tokens)
- Structural overlap (numeric extraction)
- Interaction terms (e.g. high name similarity but low address similarity)

## 5. Model
Pure-Python SGD Logistic Regression
- 21 features
- Threshold: 0.675
- Learning rate: 0.01, Epochs: 5

## 6. Hard-Negative Strategy
Training pairs were heavily augmented by sampling hard negatives from the blocking index.
- Training Source-1 entities: 50,000
- Positive pairs: 137,938
- Hard negatives: 685,245
- Total training pairs: 823,183

## 7. Validation
LOCAL VALIDATION:
- Blocking Recall: 74.20%
- Precision: 0.9850
- Recall: 0.8550
- F0.5: 0.9565
- Secondary F0.5: 0.9558

Public Leaderboard Score: PENDING

## 8. Scalability
All inference runs in a streaming fashion, with maximum memory peaking around 2-3GB for index storage. The `char4` block cap and fast pre-filter ensure O(n) runtime over the 1.7M S1 entities.

## 9. Output Generation
Outputs are deterministic. The final candidate set represents exactly the union of candidates scored.
