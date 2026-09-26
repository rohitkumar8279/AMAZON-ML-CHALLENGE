import sqlite3
import random
import math
from difflib import SequenceMatcher

def token_jaccard(str1, str2):
    if not str1 or not str2: return 0.0
    s1, s2 = set(str1.split()), set(str2.split())
    if not s1 and not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)

def str_ratio(str1, str2):
    if not str1 or not str2: return 0.0
    return SequenceMatcher(None, str1, str2).ratio()

def compute_features(s1_name, s2_name, s1_addr, s2_addr, s1_country, s2_country, is_s2):
    features = [
        1.0, # bias
        1.0 if is_s2 else 0.0,
        1.0 if s1_name == s2_name and s1_name else 0.0,
        abs(len(s1_name) - len(s2_name)),
        str_ratio(s1_name, s2_name),
        token_jaccard(s1_name, s2_name),
        1.0 if s1_addr == s2_addr and s1_addr else 0.0,
        abs(len(s1_addr) - len(s2_addr)),
        str_ratio(s1_addr, s2_addr),
        1.0 if s1_country == s2_country and s1_country else 0.0
    ]
    return features

class LogisticRegressionSGD:
    def __init__(self, lr=0.01, epochs=5):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        
    def sigmoid(self, z):
        if z < -20: return 0.0
        if z > 20: return 1.0
        return 1.0 / (1.0 + math.exp(-z))
        
    def train(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        for ep in range(self.epochs):
            indices = list(range(len(X)))
            random.shuffle(indices)
            for i in indices:
                xi = X[i]
                yi = y[i]
                z = sum(w * x for w, x in zip(self.weights, xi))
                pred = self.sigmoid(z)
                error = pred - yi
                for j in range(n_features):
                    self.weights[j] -= self.lr * error * xi[j]
                    
    def predict_proba(self, X):
        return [self.sigmoid(sum(w * x for w, x in zip(self.weights, xi))) for xi in X]

def evaluate_version(conn, table_name, gt_set):
    print(f"Evaluating ML on {table_name}...")
    c = conn.cursor()
    # Sample training data
    c.execute(f'''
        SELECT c.s1_id, c.s23_id, s1.norm_name, s23.norm_name, s1.business_address, s23.business_address, s1.country, s23.country, 1
        FROM {table_name} c
        JOIN s1 ON c.s1_id = s1.entity_id
        JOIN s2 s23 ON c.s23_id = s23.entity_id
        LIMIT 50000
    ''')
    s2_data = c.fetchall()
    
    c.execute(f'''
        SELECT c.s1_id, c.s23_id, s1.norm_name, s23.norm_name, s1.business_address, s23.business_address, s1.country, s23.country, 0
        FROM {table_name} c
        JOIN s1 ON c.s1_id = s1.entity_id
        JOIN s3 s23 ON c.s23_id = s23.entity_id
        LIMIT 50000
    ''')
    s3_data = c.fetchall()
    
    all_data = s2_data + s3_data
    random.shuffle(all_data)
    
    # 80/20 split
    split = int(len(all_data) * 0.8)
    train_data = all_data[:split]
    val_data = all_data[split:]
    
    X_train, y_train = [], []
    for row in train_data:
        s1_id, s23_id, n1, n2, a1, a2, c1, c2, is_s2 = row
        feat = compute_features(n1, n2, a1, a2, c1, c2, is_s2)
        X_train.append(feat)
        y_train.append(1 if s23_id in gt_set.get(s1_id, set()) else 0)
        
    model = LogisticRegressionSGD(epochs=5)
    model.train(X_train, y_train)
    
    X_val, y_val = [], []
    for row in val_data:
        s1_id, s23_id, n1, n2, a1, a2, c1, c2, is_s2 = row
        feat = compute_features(n1, n2, a1, a2, c1, c2, is_s2)
        X_val.append(feat)
        y_val.append(1 if s23_id in gt_set.get(s1_id, set()) else 0)
        
    probs = model.predict_proba(X_val)
    preds = [1 if p > 0.5 else 0 for p in probs]
    
    true_pos = sum(1 for t, p in zip(y_val, preds) if t == 1 and p == 1)
    false_pos = sum(1 for t, p in zip(y_val, preds) if t == 0 and p == 1)
    false_neg = sum(1 for t, p in zip(y_val, preds) if t == 1 and p == 0)
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision+recall)>0 else 0
    
    print(f"{table_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F0.5: {f05:.4f}, FP: {false_pos}, FN: {false_neg}")
    return precision, recall, f05, false_pos, false_neg

if __name__ == '__main__':
    conn = sqlite3.connect('dataset.db')
    c = conn.cursor()
    c.execute('SELECT s1_id, s23_id FROM gt')
    gt_set = {}
    for s1, s23 in c.fetchall():
        if s1 not in gt_set: gt_set[s1] = set()
        gt_set[s1].add(s23)
        
    evaluate_version(conn, 'v0_cands', gt_set)
