"""Generate the solution file from the finished pipeline, using ONE model.

Run:  python make_submission.py

Needs next to it: autoscaler.py, link_features.py
Needs on disk:    the files written by preprocessing and the tuned model
                  (finetuning/*/best_*.joblib|cbm), which supply the hyper-parameters.

Steps
  1. Check that node ids in the saved graphs match the original edge list.
  2. (optional) Check link_features.py reproduces the saved X_test.
  3. Build features for the solution pairs on the FULL known graph.
  4. Choose the model (FINAL_MODEL, or the best by test top-K accuracy if None)
     and, optionally, estimate its accuracy: fit on train only, score on test.
  5. Fit a fresh copy of that model on train+val+test, score the solution pairs,
     label the top K as real links.
"""
from sklearn.preprocessing import StandardScaler
import __main__
import json
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from torch_geometric.data import Data
from sklearn.base import BaseEstimator, TransformerMixin
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
class AutoScaler(BaseEstimator, TransformerMixin):
    """Clean inf/NaN, log1p heavy-tailed columns, then standardise."""

    def __init__(self, skew_threshold=1.0):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X).astype(float).replace([np.inf, -np.inf], np.nan)

        # Replace inf/NaN with the column's max finite value + 1 (or median for NaN)
        self.fill_ = X.max() + 1
        X = X.fillna(self.fill_)

        # Columns whose absolute skew exceeds the threshold get log1p
        # (only non-negative columns, since log1p needs values > -1)
        nonneg = (X >= 0).all()
        skew = X.skew().abs()
        self.log_cols_ = X.columns[(skew > self.skew_threshold) & nonneg].tolist()

        X = self._log(X)
        self.scaler_ = StandardScaler().fit(X)
        self.columns_ = X.columns
        return self

    def _log(self, X):
        X = X.copy()
        X[self.log_cols_] = np.log1p(X[self.log_cols_])
        return X

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_).astype(float)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(self.fill_)
        X = self._log(X)
        return self.scaler_.transform(X).astype(np.float32)
from link_features import compute_katz_scores, create_edge_features

# Models pickled from a notebook/script refer to __main__.AutoScaler.
__main__.AutoScaler = AutoScaler

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
EDGES_PATH = Path("data/input/edges_train.edgelist")
SOLUTION_PATH = Path("data/input/solutionInput.csv")
DATASET_PATH = Path("data/processed/link_prediction_dataset.npz")
GRAPHS_PATH = Path("data/processed/link_prediction_graphs.pt")
METADATA_PATH = Path("data/processed/feature_metadata.json")
OUT_DIR = Path("data/output")


N_POSITIVES = None         # None = exactly half of the solution pairs (708 of 1416)
SUBMISSION_COLUMN = "Predicted"   # match the column name of the sample submission
VERIFY_FEATURES = True     # recompute X_test with link_features.py and compare
SANITY_CHECK = True        # train-only fit, top-K accuracy on the test split

FINAL_MODEL = "GNN"

MODEL_PATHS = {
    "Logistic Regression": Path(
        "finetuning/logistic_regression_accuracy_grid_results/best_logistic_regression.joblib"
    ),
    "Random Forest": Path(
        "finetuning/randomforest_accuracy_grid_results/best_random_forest.joblib"
    ),
    "XGBoost": Path(
        "finetuning/xgboost_accuracy_grid_results/best_xgboost.joblib"
    ),
    "CatBoost": Path(
        "finetuning/catboost_accuracy_grid_results/best_catboost.cbm"
    ),
    "GNN": Path(
        "finetuning/gnn_accuracy_grid_results/best_gnn.pt"
    ),
}


class Net(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        out_channels=16,
        edge_feature_channels=34,
        dropout=0.2,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

        decoder_input_channels = 3 * out_channels + edge_feature_channels

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def encode(self, x, edge_index):
        z = self.conv1(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.conv2(z, edge_index)
        return z

    def decode(self, z, edge_label_index, edge_features):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        edge_embedding = torch.cat(
            [src + dst, torch.abs(src - dst), src * dst, edge_features],
            dim=-1,
        )
        return self.decoder(edge_embedding).squeeze(-1)

@torch.no_grad()
def predict_gnn_probabilities(model, data_obj, edge_features, device="cpu"):
    model.eval()
    model.to(device)
    x_tensor = data_obj.x.to(device)
    edge_index = data_obj.edge_index.to(device)
    edge_label_index = data_obj.edge_label_index.to(device)
    edge_feats = torch.as_tensor(edge_features, dtype=torch.float32, device=device)

    z = model.encode(x_tensor, edge_index)
    logits = model.decode(z, edge_label_index, edge_feats)
    return torch.sigmoid(logits).cpu().numpy()

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def canonical(pairs):
    return {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in pairs
        if int(a) != int(b)
    }


def to_edge_index(edge_set):
    e = np.array(sorted(edge_set), dtype=np.int64)
    both = np.concatenate([e, e[:, ::-1]], axis=0)
    return torch.as_tensor(both.T.copy(), dtype=torch.long)


def check_alignment(saved, edge_set):
    """Saved graphs must use the same node ids as the original edge list."""
    td = saved["test_data"]
    labels = td.edge_label.detach().cpu().numpy().ravel()
    pos = td.edge_label_index.detach().cpu()[:, torch.as_tensor(labels == 1)]

    saved_edges = (
        canonical(td.edge_index.detach().cpu().t().tolist())
        | canonical(pos.t().tolist())
    )
    overlap = len(saved_edges & edge_set) / max(len(edge_set), 1)
    print(f"Node-id alignment: {overlap:.1%} of original edges found in saved graphs")

    if overlap < 0.999:
        raise RuntimeError(
            "Node ids in the saved graphs do not match the edge list. "
            "from_networkx() numbers nodes in the order they first appear in G, "
            "so edge_index no longer lines up with the attribute rows. "
            "Fix preprocessing, re-run it, then re-run all tuning scripts."
        )


def fresh_model(name, path):
    """Unfitted model with the tuned hyper-parameters."""
    return clone(joblib.load(path))


def top_k_labels(score, k):
    label = np.zeros(len(score), dtype=int)
    label[np.argsort(-np.asarray(score))[:k]] = 1
    return label


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    dataset = np.load(DATASET_PATH)
    X_train, y_train = dataset["X_train"], dataset["y_train"].astype(int)
    X_val, y_val = dataset["X_val"], dataset["y_val"].astype(int)
    X_test, y_test = dataset["X_test"], dataset["y_test"].astype(int)

    saved = torch.load(GRAPHS_PATH, map_location="cpu", weights_only=False)
    x = saved["x"]
    num_nodes = x.size(0)

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    katz_beta = float(metadata["katz_beta"])
    feature_names = metadata["feature_names"]

    # 1. Full known graph with identity node ids -------------------------
    G_file = nx.read_edgelist(
        EDGES_PATH, delimiter=",", nodetype=int, create_using=nx.Graph()
    )
    if not set(G_file.nodes()) <= set(range(num_nodes)):
        raise ValueError("Edge list contains node ids outside 0..N-1.")
    edge_set = canonical(G_file.edges())
    print(f"Known graph: {num_nodes} nodes, {len(edge_set)} edges")

    check_alignment(saved, edge_set)

    # 2. Does link_features.py reproduce the saved training features? ----
    if VERIFY_FEATURES:
        print("\nVerifying feature code against saved X_test ...")
        td = saved["test_data"]
        _, katz_td = compute_katz_scores(td, beta=katz_beta)
        X_re = create_edge_features(td, katz_td)
        close = np.isclose(X_re, X_test, rtol=1e-3, atol=1e-4)
        if close.all():
            print("Feature code matches the saved X_test.")
        else:
            bad = np.flatnonzero(~close.all(axis=0))
            print("WARNING: features differ from saved X_test:",
                  [feature_names[i] for i in bad])

    # 3. Solution pairs and their features -------------------------------
    df = pd.read_csv(SOLUTION_PATH, index_col=0)
    print("\nSolution file:", df.shape)
    print(df.head())

    src = df.iloc[:, 0].to_numpy(dtype=np.int64)
    dst = df.iloc[:, 1].to_numpy(dtype=np.int64)
    if src.min() < 0 or dst.min() < 0 or max(src.max(), dst.max()) >= num_nodes:
        raise ValueError("Solution pairs contain node ids outside 0..N-1.")

    pairs = torch.as_tensor(np.vstack([src, dst]), dtype=torch.long)
    already = canonical(pairs.t().tolist()) & edge_set
    print(f"Pairs that are already edges in the graph: {len(already)} (expect 0)")

    n_pos = N_POSITIVES if N_POSITIVES is not None else len(df) // 2
    hidden = n_pos / (len(edge_set) + n_pos)
    print(f"Real links among the pairs: {n_pos}; implied hidden-edge fraction "
          f"{hidden:.1%} (your splits hid 10% per held-out split)")

    solution_graph = Data(
        x=x,
        edge_index=to_edge_index(edge_set),
        edge_label_index=pairs,
        num_nodes=num_nodes,
    )
    _, katz_sol = compute_katz_scores(solution_graph, beta=katz_beta)
    X_sol = create_edge_features(solution_graph, katz_sol)
    assert X_sol.shape == (len(df), X_train.shape[1]), X_sol.shape

    shift = (X_sol.mean(0) - X_test.mean(0)) / (X_test.std(0) + 1e-9)
    print("\nLargest feature shifts, solution vs test split (std units):")
    for i in np.argsort(-np.abs(shift))[:5]:
        print(f"  {feature_names[i]:<28s} {shift[i]:+.2f}")

    # 4. Choose the model ------------------------------------------------
    available = {n: p for n, p in MODEL_PATHS.items() if p.is_file()}
    if FINAL_MODEL is not None:
        if FINAL_MODEL not in available:
            raise RuntimeError(
                f"FINAL_MODEL '{FINAL_MODEL}' not found. "
                f"Available: {list(available) or 'none'}"
            )
        candidates = {FINAL_MODEL: available[FINAL_MODEL]}
    else:
        if not available:
            raise RuntimeError("No tuned models found; check MODEL_PATHS.")
        candidates = available

    chosen = FINAL_MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if SANITY_CHECK or chosen is None:
        print("\nTop-K accuracy on the test split (fit on train only):")
        k = int(y_test.sum())
        accuracies = {}
        for name, path in candidates.items():
            if name == "GNN":
                # For GNN sanity check, load saved checkpoint configuration
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                params = checkpoint["model_params"]
                gnn_model = Net(
                    in_channels=x.size(1),
                    hidden_channels=params["hidden_channels"],
                    out_channels=params["out_channels"],
                    edge_feature_channels=X_train.shape[1],
                    dropout=params["dropout"],
                ).to(device)
                gnn_model.load_state_dict(checkpoint["model_state_dict"])
                
                td = saved["test_data"]
                test_edge_feats = X_test # from dataset npz or recomputed
                p = predict_gnn_probabilities(gnn_model, td, test_edge_feats, device)
            else:
                model = fresh_model(name, path)
                model.fit(X_train, y_train)
                p = model.predict_proba(X_test)[:, 1]

            accuracies[name] = accuracy_score(y_test, top_k_labels(p, k))
            print(f"  {name:<20s} {accuracies[name]:.4f}")
        if chosen is None:
            chosen = max(accuracies, key=accuracies.get)
            print(f"Selected the best model: {chosen}")

    # 5. Predict on solution pairs ---------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if chosen == "GNN":
        print(f"\nLoading best GNN checkpoint for solution scoring...")
        checkpoint = torch.load(available["GNN"], map_location=device, weights_only=False)
        params = checkpoint["model_params"]
        
        final_gnn = Net(
            in_channels=x.size(1),
            hidden_channels=params["hidden_channels"],
            out_channels=params["out_channels"],
            edge_feature_channels=X_sol.shape[1],
            dropout=params["dropout"],
        ).to(device)
        final_gnn.load_state_dict(checkpoint["model_state_dict"])

        prob = predict_gnn_probabilities(final_gnn, solution_graph, X_sol, device)
    else:
        # Standard scikit-learn / CatBoost full retrain pathway
        X_all = np.vstack([X_train, X_val, X_test]).astype(np.float32)
        y_all = np.concatenate([y_train, y_val, y_test])
        print(f"\nFitting {chosen} on all labelled pairs: {X_all.shape}, "
              f"positive rate {y_all.mean():.3f}")

        model = fresh_model(chosen, available[chosen])
        model.fit(X_all, y_all)
        prob = model.predict_proba(X_sol)[:, 1]

    print("Probability quantiles (min, 25%, 50%, 75%, max):",
          np.round(np.quantile(prob, [0, 0.25, 0.5, 0.75, 1]), 4))
    print("Distinct probability values:", len(np.unique(np.round(prob, 6))))

    label = top_k_labels(prob, n_pos)

    detail = df.copy()
    detail["probability"] = prob
    detail[SUBMISSION_COLUMN] = label
    detail.to_csv(OUT_DIR / "solution_predictions.csv")

    submission = pd.DataFrame({SUBMISSION_COLUMN: label}, index=df.index)
    submission.to_csv(OUT_DIR / "submission.csv")

    print(f"\nModel: {chosen}")
    print(f"Labelled {label.sum()} of {len(label)} pairs as real links.")
    print(f"Pairs with probability >= 0.5: {(prob >= 0.5).sum()}")
    print("Saved:", (OUT_DIR / "submission.csv").resolve())


if __name__ == "__main__":
    main()