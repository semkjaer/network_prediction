"""Why is Logistic Regression perfect on your splits? Looks for shortcut features.

Run from the project root:  python diagnose_leak.py

1. Single-feature AUC on validation: any feature near 1.0 (or 0.0) is a shortcut.
2. LR validation accuracy after dropping the strongest features.
3. Category-pair table: which category pairs your positives, negatives and the
   solution pairs use. If the solution pairs do not look like a 50/50 mix of
   your positives and negatives, they were built differently.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

from autoscaler import AutoScaler

DATASET_PATH = Path("data/processed/link_prediction_dataset.npz")
GRAPHS_PATH = Path("data/processed/link_prediction_graphs.pt")
METADATA_PATH = Path("data/processed/feature_metadata.json")
SOLUTION_PATH = Path("data/input/solutionInput.csv")

d = np.load(DATASET_PATH)
X_train, y_train = d["X_train"], d["y_train"].astype(int)
X_val, y_val = d["X_val"], d["y_val"].astype(int)
names = json.loads(METADATA_PATH.read_text(encoding="utf-8"))["feature_names"]

# ------------------------------------------------------------------
# 1. Single-feature AUC
# ------------------------------------------------------------------
auc = np.array([
    roc_auc_score(y_val, np.nan_to_num(X_val[:, j]))
    for j in range(X_val.shape[1])
])
order = np.argsort(-np.abs(auc - 0.5))

print("1. Single-feature AUC on validation (0.5 = useless, 1.0 or 0.0 = perfect)")
for j in order[:10]:
    print(f"   {names[j]:<28s} {auc[j]:.3f}")


# ------------------------------------------------------------------
# 2. Drop the strongest features
# ------------------------------------------------------------------
def lr_accuracy(cols):
    model = Pipeline([
        ("scaler", AutoScaler()),
        ("clf", LogisticRegression(max_iter=5000)),
    ])
    model.fit(X_train[:, cols], y_train)
    return accuracy_score(y_val, model.predict(X_val[:, cols]))


print("\n2. Logistic regression validation accuracy after dropping the top-k features")
all_cols = list(range(X_train.shape[1]))
for k in [0, 1, 2, 3, 5, 8]:
    dropped = set(order[:k].tolist())
    keep = [c for c in all_cols if c not in dropped]
    print(f"   drop {k}: {lr_accuracy(keep):.4f}")

# ------------------------------------------------------------------
# 3. Category-pair table
# ------------------------------------------------------------------
saved = torch.load(GRAPHS_PATH, map_location="cpu", weights_only=False)
cat = np.argmax(saved["x"].numpy(), axis=1)
td = saved["test_data"]
labels = td.edge_label.numpy().ravel()
pairs = td.edge_label_index.numpy().T
sol = pd.read_csv(SOLUTION_PATH, index_col=0).iloc[:, :2].to_numpy(dtype=int)

letters = "abcdefghij"


def pair_share(p):
    lo = np.minimum(cat[p[:, 0]], cat[p[:, 1]])
    hi = np.maximum(cat[p[:, 0]], cat[p[:, 1]])
    keys = [letters[a] + letters[b] for a, b in zip(lo, hi)]
    return pd.Series(keys).value_counts(normalize=True)


table = pd.DataFrame({
    "test pos": pair_share(pairs[labels == 1]),
    "test neg": pair_share(pairs[labels == 0]),
    "solution": pair_share(sol),
}).fillna(0.0)
table["50/50 mix"] = (table["test pos"] + table["test neg"]) / 2
denominator = (table["test pos"] + table["test neg"]).replace(0, np.nan)
table["positive rate"] = table["test pos"] / denominator

print("\n3. Share of pairs per category pair (rows sum to 100% per column)")
pd.set_option("display.width", 140)
print(table.sort_values("test pos", ascending=False).round(3).to_string())
print("\n   Largest gap, solution vs 50/50 mix: "
      f"{(table['solution'] - table['50/50 mix']).abs().max():.3f}")
