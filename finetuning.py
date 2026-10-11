
# Standard library
import copy
import json
import os
import random
from itertools import product

# Data processing
import joblib
import networkx as nx
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit

# Classical machine learning models
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class Net(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        out_channels=16,
        edge_feature_channels=29,
    ):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        decoder_input_channels = (
            3 * out_channels + edge_feature_channels
        )

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index, edge_features):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]

        edge_embedding = torch.cat(
            [
                src + dst,
                torch.abs(src - dst),
                src * dst,
                edge_features,
            ],
            dim=-1,
        )

        return self.decoder(edge_embedding).squeeze(-1)


# ============================================================
# 1. CONFIGURATION
# ============================================================
def finetune_gnn():

    SEED = 42
    OUTPUT_DIR = "finetuning/gnn_accuracy_grid_results"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_edge_features = torch.as_tensor(
        X_train, dtype=torch.float32, device=device
    )
    val_edge_features = torch.as_tensor(
        X_val, dtype=torch.float32, device=device
    )
    test_edge_features = torch.as_tensor(
        X_test, dtype=torch.float32, device=device
    )

    train_labels = train_data.edge_label.float()
    val_labels = val_data.edge_label.float()
    test_labels = test_data.edge_label.float()

    assert train_edge_features.shape[0] == len(train_labels)
    assert val_edge_features.shape[0] == len(val_labels)
    assert test_edge_features.shape[0] == len(test_labels)

    THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.01), 2)


    # ============================================================
    # 2. MODEL
    # ============================================================

    class Net(nn.Module):
        def __init__(
            self,
            in_channels,
            hidden_channels=32,
            out_channels=16,
            edge_feature_channels=29,
            dropout=0.2,
        ):
            super().__init__()

            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            self.dropout = dropout

            decoder_input_channels = (
                3 * out_channels + edge_feature_channels
            )

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
            z = F.dropout(
                z, p=self.dropout, training=self.training
            )
            z = self.conv2(z, edge_index)
            return z

        def decode(self, z, edge_label_index, edge_features):
            src = z[edge_label_index[0]]
            dst = z[edge_label_index[1]]

            edge_embedding = torch.cat(
                [
                    src + dst,
                    torch.abs(src - dst),
                    src * dst,
                    edge_features,
                ],
                dim=-1,
            )

            return self.decoder(edge_embedding).squeeze(-1)


    # ============================================================
    # 3. HELPERS
    # ============================================================

    def get_edge_index(split_data):
        return split_data.edge_label_index


    def select_threshold(y_true, probabilities):
        """Maximize validation accuracy; prefer threshold near 0.5 on ties."""
        predictions = (
            probabilities[:, None] >= THRESHOLDS[None, :]
        ).astype(np.int8)

        accuracies = np.mean(
            predictions == y_true[:, None], axis=0
        )

        best_accuracy = accuracies.max()
        tied = np.flatnonzero(
            np.isclose(accuracies, best_accuracy)
        )

        best_idx = tied[
            np.argmin(np.abs(THRESHOLDS[tied] - 0.5))
        ]

        return (
            float(THRESHOLDS[best_idx]),
            float(accuracies[best_idx]),
        )


    @torch.no_grad()
    def predict_probabilities(
        model, split_data, edge_features
    ):
        model.eval()

        # Each split uses its own message-passing graph.
        z = model.encode(x, split_data.edge_index)

        logits = model.decode(
            z,
            get_edge_index(split_data),
            edge_features,
        )

        return torch.sigmoid(logits).cpu().numpy()


    def train_one_epoch(
        model, optimizer, criterion, split_data, edge_features
    ):
        model.train()
        optimizer.zero_grad()

        z = model.encode(x, split_data.edge_index)

        logits = model.decode(
            z,
            get_edge_index(split_data),
            edge_features,
        )

        loss = criterion(logits, split_data.edge_label.float())
        loss.backward()
        optimizer.step()

        return float(loss.item())


    # ============================================================
    # 4. HYPERPARAMETER GRID
    # ============================================================

    # Start with this manageable grid.
    # 2 * 2 * 2 * 2 * 2 = 32 configurations.

    PARAM_GRID = {
        "hidden_channels": [32, 64],
        "out_channels": [16, 32],
        "dropout": [0.2, 0.4],
        "lr": [0.001, 0.01],
        "weight_decay": [0.0, 0.0001],
    }

    EPOCHS = 150
    PATIENCE = 20

    # Use validation loss for early stopping to avoid repeatedly
    # choosing epochs by noisy accuracy. Threshold is still selected
    # by validation accuracy after training.

    # ============================================================
    # 5. GRID SEARCH
    # ============================================================

    grid_keys = list(PARAM_GRID)
    combinations = list(
        product(*(PARAM_GRID[k] for k in grid_keys))
    )

    results = []
    best_accuracy = -np.inf
    best_params = None
    best_threshold = 0.5
    best_state = None

    y_val_np = val_labels.cpu().numpy().astype(int)

    for run_idx, values in enumerate(combinations, start=1):
        params = dict(zip(grid_keys, values))

        print(
            f"\n[{run_idx}/{len(combinations)}] "
            f"{params}"
        )

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        model = Net(
            in_channels=x.size(1),
            hidden_channels=params["hidden_channels"],
            out_channels=params["out_channels"],
            edge_feature_channels=train_edge_features.shape[1],
            dropout=params["dropout"],
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )

        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = np.inf
        best_epoch_state = None
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(
                model,
                optimizer,
                criterion,
                train_data,
                train_edge_features,
            )

            model.eval()
            with torch.no_grad():
                z_val = model.encode(x, val_data.edge_index)
                val_logits = model.decode(
                    z_val,
                    val_data.edge_label_index,
                    val_edge_features,
                )
                val_loss = criterion(
                    val_logits, val_labels
                ).item()

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_epoch = epoch
                best_epoch_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                break

        # Restore the epoch with lowest validation loss.
        model.load_state_dict(best_epoch_state)

        val_probs = predict_probabilities(
            model, val_data, val_edge_features
        )

        threshold, val_accuracy = select_threshold(
            y_val_np, val_probs
        )

        val_preds = (val_probs >= threshold).astype(int)

        val_f1 = f1_score(
            y_val_np, val_preds, zero_division=0
        )
        val_ap = average_precision_score(
            y_val_np, val_probs
        )
        val_auc = roc_auc_score(y_val_np, val_probs)

        result = {
            **params,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "validation_accuracy": val_accuracy,
            "validation_threshold": threshold,
            "validation_F1": val_f1,
            "validation_AP": val_ap,
            "validation_ROC_AUC": val_auc,
        }
        results.append(result)

        print(
            f"Epoch={best_epoch} | "
            f"Accuracy={val_accuracy:.5f} | "
            f"Threshold={threshold:.2f} | "
            f"F1={val_f1:.5f} | "
            f"AP={val_ap:.5f} | "
            f"ROC-AUC={val_auc:.5f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = params.copy()
            best_threshold = threshold
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            print("  --> New best model")


    # ============================================================
    # 6. SAVE SEARCH RESULTS
    # ============================================================

    results_df = pd.DataFrame(results).sort_values(
        ["validation_accuracy", "validation_AP"],
        ascending=[False, False],
    ).reset_index(drop=True)

    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "grid_search_results.csv"),
        index=False,
    )

    print("\nTop configurations:")
    print(results_df.head(10).to_string(index=False))

    print("\nBest parameters:")
    print(json.dumps(best_params, indent=2))
    print("Best validation accuracy:", best_accuracy)
    print("Selected threshold:", best_threshold)


    # ============================================================
    # 7. REBUILD BEST MODEL AND SAVE
    # ============================================================

    final_model = Net(
        in_channels=x.size(1),
        hidden_channels=best_params["hidden_channels"],
        out_channels=best_params["out_channels"],
        edge_feature_channels=train_edge_features.shape[1],
        dropout=best_params["dropout"],
    ).to(device)

    final_model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "model_params": best_params,
            "threshold": best_threshold,
            "input_channels": int(x.size(1)),
            "edge_feature_channels": int(train_edge_features.shape[1]),
        },
        os.path.join(OUTPUT_DIR, "best_gnn.pt"),
    )


    # ============================================================
    # 8. TEST EVALUATION — RUN ONCE AFTER MODEL SELECTION
    # ============================================================

    y_test_np = test_labels.cpu().numpy().astype(int)

    test_probs = predict_probabilities(
        final_model, test_data, test_edge_features
    )

    test_preds = (test_probs >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test_np, test_preds, labels=[0, 1]
    ).ravel()

    test_metrics = {
        "threshold": best_threshold,
        "accuracy": accuracy_score(y_test_np, test_preds),
        "balanced_accuracy": balanced_accuracy_score(
            y_test_np, test_preds
        ),
        "precision": precision_score(
            y_test_np, test_preds, zero_division=0
        ),
        "recall": recall_score(
            y_test_np, test_preds, zero_division=0
        ),
        "f1": f1_score(
            y_test_np, test_preds, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test_np, test_probs),
        "average_precision": average_precision_score(
            y_test_np, test_probs
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    print("\nTest metrics:")
    print(pd.Series(test_metrics).to_string())

    with open(
        os.path.join(OUTPUT_DIR, "test_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(test_metrics, f, indent=2)


    # ============================================================
    # 9. PRECISION-RECALL CURVE
    # ============================================================

    precision, recall, _ = precision_recall_curve(
        y_test_np, test_probs
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        label=f"GNN (AP={test_metrics['average_precision']:.3f})",
    )
    plt.axhline(
        y=np.mean(y_test_np),
        linestyle="--",
        label=f"Positive rate ({np.mean(y_test_np):.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("GNN Precision-Recall Curve — Test Set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "precision_recall_curve.png"),
        dpi=150,
    )
    plt.show()

    print(f"\nResults saved in: {OUTPUT_DIR}")


# ============================================================
# 1. CONFIGURATION
# ============================================================
def finetune_xgboost():
    SEED = 42
    OUTPUT_DIR = "finetuning/xgboost_accuracy_grid_results"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============================================================
    # 2. HYPERPARAMETER GRID
    # ============================================================

    PARAM_GRID = {
        "n_estimators": [300, 500, 800],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.07],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 2, 5],
        "reg_lambda": [1.0, 5.0],
        "reg_alpha": [0.0, 0.1],
        "gamma": [0.0, 0.2],
    }

    # Full grid: 3 * 3 * 2 * 2 * 2 * 3 * 2 * 2 * 2 = 1,728 configs.
    # Start with QUICK_SEARCH=True for a smaller search.

    QUICK_SEARCH = True

    if QUICK_SEARCH:
        PARAM_GRID = {
            "n_estimators": [300, 500],
            "max_depth": [3, 4, 6],
            "learning_rate": [0.03, 0.07],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "min_child_weight": [1, 2, 5],
            "reg_lambda": [1.0, 5.0],
            "reg_alpha": [0.0, 0.1],
            "gamma": [0.0],
        }
        # 2 * 3 * 2 * 1 * 1 * 3 * 2 * 2 * 1 = 144 configs.

    THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.01), 2)


    def best_accuracy_threshold(y_true, probabilities):
        """Select the threshold with highest validation accuracy."""
        predictions = (
            probabilities[:, None] >= THRESHOLDS[None, :]
        ).astype(np.int8)

        accuracies = np.mean(
            predictions == y_true[:, None],
            axis=0,
        )

        max_accuracy = accuracies.max()
        tied = np.flatnonzero(np.isclose(accuracies, max_accuracy))

        # Prefer 0.50 when multiple thresholds give equal accuracy.
        best_idx = tied[
            np.argmin(np.abs(THRESHOLDS[tied] - 0.5))
        ]

        return float(THRESHOLDS[best_idx]), float(accuracies[best_idx])


    # ============================================================
    # 3. GRID SEARCH
    # ============================================================

    grid_keys = list(PARAM_GRID)
    combinations = list(product(*(PARAM_GRID[k] for k in grid_keys)))

    print(f"Configurations to evaluate: {len(combinations)}")

    search_results = []
    best_accuracy = -np.inf
    best_params = None
    best_threshold = 0.5

    for run_number, values in enumerate(combinations, start=1):
        params = dict(zip(grid_keys, values))

        print(f"\n[{run_number}/{len(combinations)}] {params}")

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
            **params,
        )

        model.fit(X_train, y_train)

        val_probs = model.predict_proba(X_val)[:, 1]
        threshold, val_accuracy = best_accuracy_threshold(
            y_val, val_probs
        )
        val_preds = (val_probs >= threshold).astype(int)

        val_f1 = f1_score(y_val, val_preds, zero_division=0)
        val_ap = average_precision_score(y_val, val_probs)
        val_auc = roc_auc_score(y_val, val_probs)

        result = {
            **params,
            "validation_accuracy": val_accuracy,
            "validation_threshold": threshold,
            "validation_F1": val_f1,
            "validation_AP": val_ap,
            "validation_ROC_AUC": val_auc,
        }
        search_results.append(result)

        print(
            f"Accuracy={val_accuracy:.5f} | "
            f"Threshold={threshold:.2f} | "
            f"F1={val_f1:.5f} | "
            f"AP={val_ap:.5f} | "
            f"ROC-AUC={val_auc:.5f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = params.copy()
            best_threshold = threshold
            print("  --> New best configuration")


    # ============================================================
    # 4. SAVE GRID SEARCH RESULTS
    # ============================================================

    results_df = pd.DataFrame(search_results).sort_values(
        ["validation_accuracy", "validation_AP"],
        ascending=[False, False],
    ).reset_index(drop=True)

    results_path = os.path.join(
        OUTPUT_DIR, "grid_search_results.csv"
    )
    results_df.to_csv(results_path, index=False)

    print("\nTop 10 configurations:")
    print(results_df.head(10).to_string(index=False))

    print("\nBest parameters:")
    print(json.dumps(best_params, indent=2))
    print(f"Best validation accuracy: {best_accuracy:.5f}")
    print(f"Selected threshold: {best_threshold:.2f}")


    # ============================================================
    # 5. REFIT THE BEST MODEL
    # ============================================================
    # Fit on X_train only, preserving validation for model selection.
    # Keep the validation-selected threshold fixed.

    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
        **best_params,
    )

    final_model.fit(X_train, y_train)

    model_path = os.path.join(
        OUTPUT_DIR, "best_xgboost.joblib"
    )
    joblib.dump(final_model, model_path)


    # ============================================================
    # 6. VALIDATION METRICS AFTER REFIT
    # ============================================================

    val_probs = final_model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= best_threshold).astype(int)

    print("\nRefitted model validation metrics:")
    print(f"Accuracy: {accuracy_score(y_val, val_preds):.5f}")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"F1: {f1_score(y_val, val_preds, zero_division=0):.5f}")
    print(f"AP: {average_precision_score(y_val, val_probs):.5f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, val_probs):.5f}")


    # ============================================================
    # 7. FINAL TEST EVALUATION
    # ============================================================

    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_preds, labels=[0, 1]
    ).ravel()

    test_metrics = {
        "threshold": best_threshold,
        "accuracy": accuracy_score(y_test, test_preds),
        "balanced_accuracy": balanced_accuracy_score(y_test, test_preds),
        "precision": precision_score(y_test, test_preds, zero_division=0),
        "recall": recall_score(y_test, test_preds, zero_division=0),
        "f1": f1_score(y_test, test_preds, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "ROC_AUC": roc_auc_score(y_test, test_probs),
        "average_precision": average_precision_score(y_test, test_probs),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    print("\nFinal test metrics:")
    print(pd.Series(test_metrics).to_string())

    with open(
        os.path.join(OUTPUT_DIR, "test_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(test_metrics, f, indent=2)


    # ============================================================
    # 8. PRECISION-RECALL CURVE
    # ============================================================

    precision, recall, _ = precision_recall_curve(y_test, test_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        label=f"XGBoost (AP={test_metrics['average_precision']:.3f})",
    )
    plt.axhline(
        y=np.mean(y_test),
        linestyle="--",
        label=f"Positive rate ({np.mean(y_test):.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("XGBoost Precision-Recall Curve — Test Set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "precision_recall_curve.png"),
        dpi=150,
    )
    plt.show()


    # ============================================================
    # 9. FEATURE IMPORTANCE
    # ============================================================

    importance_df = pd.DataFrame({
        "feature_index": np.arange(X_train.shape[1]),
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(
        os.path.join(OUTPUT_DIR, "feature_importances.csv"),
        index=False,
    )

    top_features = importance_df.head(20).sort_values("importance")

    plt.figure(figsize=(10, 6))
    plt.barh(
        top_features["feature_index"].astype(str),
        top_features["importance"],
    )
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Index")
    plt.title("Top XGBoost Feature Importances")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "feature_importances.png"),
        dpi=150,
    )
    plt.show()

    print(f"\nSaved model and results in: {OUTPUT_DIR}")
    print("Best model:", model_path)
    print("Grid results:", results_path)


# ============================================================
# 1. CONFIGURATION
# ============================================================
def finetune_logistic_regression():

    RANDOM_SEED = 42
    OUTPUT_DIR = "finetuning/logistic_regression_accuracy_grid_results"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    assert X_train.ndim == X_val.ndim == X_test.ndim == 2
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert set(np.unique(y_train)) == {0, 1}
    assert set(np.unique(y_val)) == {0, 1}
    assert set(np.unique(y_test)).issubset({0, 1})

    print("Dataset shapes:")
    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)
    print("Train positive rate:", y_train.mean())
    print("Validation positive rate:", y_val.mean())
    print("Test positive rate:", y_test.mean())


    # ============================================================
    # 2. HYPERPARAMETER GRID
    # ============================================================

    PARAM_GRID = {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "class_weight": [None, "balanced"],
    }

    # 5 * 2 * 2 * 2 = 40 configurations.
    # All combinations below support the chosen penalties.
    THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.01), 2)


    def best_accuracy_threshold(y_true, probabilities):
        """Maximize accuracy using validation labels only."""
        predictions = (
            probabilities[:, None] >= THRESHOLDS[None, :]
        ).astype(np.int8)

        accuracies = np.mean(
            predictions == y_true[:, None],
            axis=0,
        )

        max_accuracy = accuracies.max()

        # Break ties by choosing the threshold closest to 0.5.
        tied = np.flatnonzero(np.isclose(accuracies, max_accuracy))
        best_idx = tied[
            np.argmin(np.abs(THRESHOLDS[tied] - 0.5))
        ]

        return (
            float(THRESHOLDS[best_idx]),
            float(accuracies[best_idx]),
        )


    # ============================================================
    # 3. GRID SEARCH
    # ============================================================

    grid_keys = list(PARAM_GRID)
    all_combinations = list(
        product(*(PARAM_GRID[key] for key in grid_keys))
    )

    total_runs = len(all_combinations)
    print(f"\nConfigurations to evaluate: {total_runs}")

    search_results = []
    best_accuracy = -np.inf
    best_params = None
    best_threshold = 0.5
    best_model = None

    for run_number, values in enumerate(all_combinations, start=1):
        params = dict(zip(grid_keys, values))

        print(f"\n[{run_number}/{total_runs}] Testing: {params}")

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=params["C"],
                penalty=params["penalty"],
                solver=params["solver"],
                class_weight=params["class_weight"],
                max_iter=5000,
                random_state=RANDOM_SEED,
                tol=1e-4,
            )),
        ])

        model.fit(X_train, y_train)

        val_probs = model.predict_proba(X_val)[:, 1]

        threshold, val_accuracy = best_accuracy_threshold(
            y_val, val_probs
        )

        val_preds = (val_probs >= threshold).astype(int)

        val_f1 = f1_score(
            y_val, val_preds, zero_division=0
        )
        val_ap = average_precision_score(y_val, val_probs)
        val_auc = roc_auc_score(y_val, val_probs)

        result = {
            **params,
            "validation_accuracy": val_accuracy,
            "validation_threshold": threshold,
            "validation_F1": val_f1,
            "validation_AP": val_ap,
            "validation_ROC_AUC": val_auc,
        }
        search_results.append(result)

        print(
            f"Accuracy={val_accuracy:.5f} | "
            f"Threshold={threshold:.2f} | "
            f"F1={val_f1:.5f} | "
            f"AP={val_ap:.5f} | "
            f"ROC-AUC={val_auc:.5f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = params.copy()
            best_threshold = threshold
            best_model = model

            print("  --> New best model by validation accuracy")


    # ============================================================
    # 4. SAVE GRID SEARCH RESULTS
    # ============================================================

    results_df = pd.DataFrame(search_results)
    results_df = results_df.sort_values(
        ["validation_accuracy", "validation_AP"],
        ascending=[False, False],
    ).reset_index(drop=True)

    results_path = os.path.join(
        OUTPUT_DIR, "grid_search_results.csv"
    )
    results_df.to_csv(results_path, index=False)

    print("\nTop 10 configurations:")
    print(results_df.head(10).to_string(index=False))

    print("\nBest hyperparameters:")
    print(json.dumps(best_params, indent=2))
    print(f"Best validation accuracy: {best_accuracy:.5f}")
    print(f"Selected threshold: {best_threshold:.2f}")


    # ============================================================
    # 5. REFIT SELECTED MODEL ON TRAINING DATA
    # ============================================================
    # Refit the winning configuration on training data only.
    # Keep the validation-selected threshold fixed.

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=best_params["C"],
            penalty=best_params["penalty"],
            solver=best_params["solver"],
            class_weight=best_params["class_weight"],
            max_iter=5000,
            random_state=RANDOM_SEED,
            tol=1e-4,
        )),
    ])

    final_model.fit(X_train, y_train)

    model_path = os.path.join(
        OUTPUT_DIR, "best_logistic_regression.joblib"
    )
    joblib.dump(final_model, model_path)


    # ============================================================
    # 6. VALIDATION METRICS AFTER REFIT
    # ============================================================

    val_probs = final_model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= best_threshold).astype(int)

    print("\nRefitted model validation metrics:")
    print(f"Accuracy: {accuracy_score(y_val, val_preds):.5f}")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"F1: {f1_score(y_val, val_preds, zero_division=0):.5f}")
    print(f"AP: {average_precision_score(y_val, val_probs):.5f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, val_probs):.5f}")


    # ============================================================
    # 7. FINAL TEST EVALUATION
    # ============================================================

    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_preds, labels=[0, 1]
    ).ravel()

    test_metrics = {
        "threshold": best_threshold,
        "accuracy": accuracy_score(y_test, test_preds),
        "balanced_accuracy": balanced_accuracy_score(y_test, test_preds),
        "precision": precision_score(y_test, test_preds, zero_division=0),
        "recall": recall_score(y_test, test_preds, zero_division=0),
        "f1": f1_score(y_test, test_preds, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "ROC_AUC": roc_auc_score(y_test, test_probs),
        "average_precision": average_precision_score(y_test, test_probs),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    print("\nFinal test metrics:")
    print(pd.Series(test_metrics).to_string())

    with open(
        os.path.join(OUTPUT_DIR, "test_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(test_metrics, f, indent=2)


    # ============================================================
    # 8. PRECISION-RECALL CURVE
    # ============================================================

    precision, recall, _ = precision_recall_curve(
        y_test, test_probs
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        label=(
            f"Logistic Regression "
            f"(AP={test_metrics['average_precision']:.3f})"
        ),
    )
    plt.axhline(
        y=np.mean(y_test),
        linestyle="--",
        label=f"Positive rate ({np.mean(y_test):.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Logistic Regression Precision-Recall Curve — Test Set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "precision_recall_curve.png"),
        dpi=150,
    )
    plt.show()


    # ============================================================
    # 9. COEFFICIENT IMPORTANCE
    # ============================================================

    classifier = final_model.named_steps["classifier"]
    coefficients = classifier.coef_.ravel()

    coefficient_df = pd.DataFrame({
        "feature_index": np.arange(X_train.shape[1]),
        "coefficient": coefficients,
        "absolute_coefficient": np.abs(coefficients),
    }).sort_values("absolute_coefficient", ascending=False)

    coefficient_df.to_csv(
        os.path.join(OUTPUT_DIR, "feature_coefficients.csv"),
        index=False,
    )

    top_coefficients = coefficient_df.head(20).sort_values(
        "coefficient", ascending=True
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        top_coefficients["feature_index"].astype(str),
        top_coefficients["coefficient"],
    )
    plt.axvline(0, linewidth=1)
    plt.xlabel("Coefficient (standardized features)")
    plt.ylabel("Feature Index")
    plt.title("Top Logistic Regression Coefficients")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "feature_coefficients.png"),
        dpi=150,
    )
    plt.show()

    print(f"\nSaved model and results in: {OUTPUT_DIR}")
    print("Best model:", model_path)
    print("Grid results:", results_path)



# ============================================================
# 1. CONFIGURATION
# ============================================================
def finetune_random_forest():
    RANDOM_SEED = 42
    OUTPUT_DIR = "finetuning/randomforest_accuracy_grid_results"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    assert X_train.ndim == X_val.ndim == X_test.ndim == 2
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert set(np.unique(y_train)) == {0, 1}
    assert set(np.unique(y_val)) == {0, 1}
    assert set(np.unique(y_test)).issubset({0, 1})

    print("Dataset shapes:")
    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)
    print("Train positive rate:", y_train.mean())
    print("Validation positive rate:", y_val.mean())
    print("Test positive rate:", y_test.mean())


    # ============================================================
    # 2. HYPERPARAMETER GRID
    # ============================================================

    PARAM_GRID = {
        "n_estimators": [300, 600],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", 0.7, 1.0],
        "class_weight": [None, "balanced"],
    }

    # 2 * 3 * 3 * 4 * 3 * 2 = 432 configurations.
    # Set QUICK_SEARCH=True for a smaller initial search.
    QUICK_SEARCH = False

    if QUICK_SEARCH:
        PARAM_GRID = {
            "n_estimators": [300],
            "max_depth": [None, 15],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.7],
            "class_weight": [None, "balanced"],
        }

    # Tune thresholds for validation accuracy.
    THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.01), 2)


    def best_accuracy_threshold(y_true, probabilities):
        """Find the threshold with highest accuracy on validation data."""
        predictions = (
            probabilities[:, None] >= THRESHOLDS[None, :]
        ).astype(np.int8)

        accuracies = np.mean(
            predictions == y_true[:, None],
            axis=0,
        )

        max_accuracy = accuracies.max()

        # Break ties by choosing the threshold closest to 0.5.
        tied = np.flatnonzero(np.isclose(accuracies, max_accuracy))
        best_idx = tied[
            np.argmin(np.abs(THRESHOLDS[tied] - 0.5))
        ]

        return (
            float(THRESHOLDS[best_idx]),
            float(accuracies[best_idx]),
        )


    # ============================================================
    # 3. GRID SEARCH
    # ============================================================

    grid_keys = list(PARAM_GRID)
    all_combinations = list(
        product(*(PARAM_GRID[key] for key in grid_keys))
    )

    total_runs = len(all_combinations)
    print(f"\nConfigurations to evaluate: {total_runs}")

    search_results = []
    best_accuracy = -np.inf
    best_params = None
    best_threshold = 0.5
    best_model = None

    for run_number, values in enumerate(all_combinations, start=1):
        params = dict(zip(grid_keys, values))

        print(f"\n[{run_number}/{total_runs}] Testing: {params}")

        model = RandomForestClassifier(
            **params,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        val_probs = model.predict_proba(X_val)[:, 1]

        threshold, val_accuracy = best_accuracy_threshold(
            y_val, val_probs
        )

        val_preds = (val_probs >= threshold).astype(int)

        val_f1 = f1_score(
            y_val, val_preds, zero_division=0
        )
        val_ap = average_precision_score(y_val, val_probs)
        val_auc = roc_auc_score(y_val, val_probs)

        result = {
            **params,
            "validation_accuracy": val_accuracy,
            "validation_threshold": threshold,
            "validation_F1": val_f1,
            "validation_AP": val_ap,
            "validation_ROC_AUC": val_auc,
        }
        search_results.append(result)

        print(
            f"Accuracy={val_accuracy:.5f} | "
            f"Threshold={threshold:.2f} | "
            f"F1={val_f1:.5f} | "
            f"AP={val_ap:.5f} | "
            f"ROC-AUC={val_auc:.5f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = params.copy()
            best_threshold = threshold
            best_model = model

            print("  --> New best model by validation accuracy")


    # ============================================================
    # 4. SAVE GRID SEARCH RESULTS
    # ============================================================

    results_df = pd.DataFrame(search_results)
    results_df = results_df.sort_values(
        ["validation_accuracy", "validation_AP"],
        ascending=[False, False],
    ).reset_index(drop=True)

    results_path = os.path.join(
        OUTPUT_DIR, "grid_search_results.csv"
    )
    results_df.to_csv(results_path, index=False)

    print("\nTop 10 configurations:")
    print(results_df.head(10).to_string(index=False))

    print("\nBest hyperparameters:")
    print(json.dumps(best_params, indent=2))
    print(f"Best validation accuracy: {best_accuracy:.5f}")
    print(f"Selected threshold: {best_threshold:.2f}")


    # ============================================================
    # 5. REFIT SELECTED MODEL ON TRAINING DATA
    # ============================================================
    # Refit using the chosen configuration. The validation set is
    # not included in fitting, and the threshold remains validation-
    # selected.

    final_model = RandomForestClassifier(
        **best_params,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    final_model.fit(X_train, y_train)

    model_path = os.path.join(
        OUTPUT_DIR, "best_random_forest.joblib"
    )
    joblib.dump(final_model, model_path)


    # ============================================================
    # 6. VALIDATION METRICS AFTER REFIT
    # ============================================================

    val_probs = final_model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= best_threshold).astype(int)

    print("\nRefitted model validation metrics:")
    print(f"Accuracy: {accuracy_score(y_val, val_preds):.5f}")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"F1: {f1_score(y_val, val_preds, zero_division=0):.5f}")
    print(f"AP: {average_precision_score(y_val, val_probs):.5f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, val_probs):.5f}")


    # ============================================================
    # 7. FINAL TEST EVALUATION
    # ============================================================
    # Do not change hyperparameters or threshold based on test results.

    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_preds, labels=[0, 1]
    ).ravel()

    test_metrics = {
        "threshold": best_threshold,
        "accuracy": accuracy_score(y_test, test_preds),
        "balanced_accuracy": balanced_accuracy_score(y_test, test_preds),
        "precision": precision_score(y_test, test_preds, zero_division=0),
        "recall": recall_score(y_test, test_preds, zero_division=0),
        "f1": f1_score(y_test, test_preds, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "ROC_AUC": roc_auc_score(y_test, test_probs),
        "average_precision": average_precision_score(y_test, test_probs),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    print("\nFinal test metrics:")
    print(pd.Series(test_metrics).to_string())

    with open(
        os.path.join(OUTPUT_DIR, "test_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(test_metrics, f, indent=2)


    # ============================================================
    # 8. PRECISION-RECALL CURVE
    # ============================================================

    precision, recall, _ = precision_recall_curve(
        y_test, test_probs
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        label=f"Random Forest (AP={test_metrics['average_precision']:.3f})",
    )
    plt.axhline(
        y=np.mean(y_test),
        linestyle="--",
        label=f"Positive rate ({np.mean(y_test):.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Random Forest Precision-Recall Curve — Test Set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "precision_recall_curve.png"),
        dpi=150,
    )
    plt.show()


    # ============================================================
    # 9. FEATURE IMPORTANCE
    # ============================================================

    importances = final_model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "feature_index": np.arange(X_train.shape[1]),
        "importance": importances,
    }).sort_values("importance", ascending=False)

    feature_importance_df.to_csv(
        os.path.join(OUTPUT_DIR, "feature_importance.csv"),
        index=False,
    )

    plt.figure(figsize=(10, 6))
    top_features = feature_importance_df.head(20).sort_values(
        "importance", ascending=True
    )

    plt.barh(
        top_features["feature_index"].astype(str),
        top_features["importance"],
    )
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Index")
    plt.title("Top Random Forest Features")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "feature_importance.png"),
        dpi=150,
    )
    plt.show()

    print(f"\nSaved model and results in: {OUTPUT_DIR}")
    print("Best model:", model_path)
    print("Grid results:", results_path)


# ============================================================
# 1. CONFIGURATION
# ============================================================
def finetune_catboost():
    RANDOM_SEED = 42
    EARLY_STOPPING_ROUNDS = 100
    OUTPUT_DIR = "finetuning/catboost_accuracy_grid_results"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    assert X_train.ndim == X_val.ndim == X_test.ndim == 2
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert set(np.unique(y_train)) == {0, 1}
    assert set(np.unique(y_val)) == {0, 1}
    assert set(np.unique(y_test)).issubset({0, 1})

    print("Dataset shapes:")
    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)
    print("Train positive rate:", y_train.mean())
    print("Validation positive rate:", y_val.mean())
    print("Test positive rate:", y_test.mean())


    # ============================================================
    # 2. HYPERPARAMETER GRID
    # ============================================================

    PARAM_GRID = {
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.07],
        "l2_leaf_reg": [3, 7],
        "random_strength": [0.5, 1.0],
        "bagging_temperature": [0.0, 1.0],
    }

    ITERATION_OPTIONS = [500, 1000]

    # Search a broad range, including the default threshold.
    THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.01), 2)


    def best_accuracy_threshold(y_true, probabilities):
        """Maximize classification accuracy using validation labels only."""
        predictions = (
            probabilities[:, None] >= THRESHOLDS[None, :]
        ).astype(np.int8)

        accuracies = np.mean(
            predictions == y_true[:, None],
            axis=0,
        )

        max_accuracy = accuracies.max()

        # If tied, choose the threshold closest to 0.5.
        tied = np.flatnonzero(
            np.isclose(accuracies, max_accuracy)
        )
        best_idx = tied[
            np.argmin(np.abs(THRESHOLDS[tied] - 0.5))
        ]

        return (
            float(THRESHOLDS[best_idx]),
            float(accuracies[best_idx]),
        )


    # ============================================================
    # 3. GRID SEARCH
    # ============================================================

    grid_keys = list(PARAM_GRID)
    all_combinations = list(
        product(*(PARAM_GRID[key] for key in grid_keys))
    )

    total_runs = len(all_combinations) * len(ITERATION_OPTIONS)
    print(f"\nConfigurations to evaluate: {total_runs}")

    search_results = []
    best_accuracy = -np.inf
    best_model = None
    best_params = None
    best_threshold = 0.5

    run_number = 0

    for values in all_combinations:
        params = dict(zip(grid_keys, values))

        for iterations in ITERATION_OPTIONS:
            run_number += 1

            current_params = {
                **params,
                "iterations": iterations,
            }

            print(
                f"\n[{run_number}/{total_runs}] "
                f"Testing: {current_params}"
            )

            model = CatBoostClassifier(
                **current_params,
                loss_function="Logloss",
                eval_metric="Accuracy",
                random_seed=RANDOM_SEED,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )

            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False,
            )

            val_probs = model.predict_proba(X_val)[:, 1]

            # Tune threshold using validation data only.
            threshold, val_accuracy = best_accuracy_threshold(
                y_val, val_probs
            )

            val_preds = (val_probs >= threshold).astype(int)
            val_ap = average_precision_score(y_val, val_probs)
            val_auc = roc_auc_score(y_val, val_probs)
            val_f1 = f1_score(
                y_val, val_preds, zero_division=0
            )

            best_iteration = model.get_best_iteration()
            if best_iteration is None or best_iteration < 0:
                best_iteration = iterations
            else:
                best_iteration += 1

            result = {
                **current_params,
                "best_iteration": best_iteration,
                "validation_accuracy": val_accuracy,
                "validation_threshold": threshold,
                "validation_F1": val_f1,
                "validation_AP": val_ap,
                "validation_ROC_AUC": val_auc,
            }
            search_results.append(result)

            print(
                f"Accuracy={val_accuracy:.5f} | "
                f"Threshold={threshold:.2f} | "
                f"F1={val_f1:.5f} | "
                f"AP={val_ap:.5f} | "
                f"ROC-AUC={val_auc:.5f}"
            )

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_params = {
                    **params,
                    "iterations": best_iteration,
                }
                best_threshold = threshold
                best_model = model

                print("  --> New best model by validation accuracy")


    # ============================================================
    # 4. SAVE GRID SEARCH RESULTS
    # ============================================================

    results_df = pd.DataFrame(search_results)
    results_df = results_df.sort_values(
        ["validation_accuracy", "validation_AP"],
        ascending=[False, False],
    ).reset_index(drop=True)

    results_path = os.path.join(
        OUTPUT_DIR, "grid_search_results.csv"
    )
    results_df.to_csv(results_path, index=False)

    print("\nTop 10 configurations:")
    print(results_df.head(10).to_string(index=False))

    print("\nBest hyperparameters:")
    print(json.dumps(best_params, indent=2))
    print(f"Best validation accuracy: {best_accuracy:.5f}")
    print(f"Selected threshold: {best_threshold:.2f}")


    # ============================================================
    # 5. REFIT SELECTED MODEL ON TRAINING DATA
    # ============================================================
    # Keep the validation set out of model fitting.
    # Preserve the threshold selected during validation tuning.

    final_model = CatBoostClassifier(
        **best_params,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

    final_model.fit(X_train, y_train, verbose=False)

    model_path = os.path.join(
        OUTPUT_DIR, "best_catboost.cbm"
    )
    final_model.save_model(model_path)


    # ============================================================
    # 6. VALIDATION METRICS AFTER REFIT
    # ============================================================

    val_probs = final_model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= best_threshold).astype(int)

    print("\nRefitted model validation metrics:")
    print(f"Accuracy: {accuracy_score(y_val, val_preds):.5f}")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"F1: {f1_score(y_val, val_preds, zero_division=0):.5f}")
    print(f"AP: {average_precision_score(y_val, val_probs):.5f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, val_probs):.5f}")


    # ============================================================
    # 7. FINAL TEST EVALUATION — DO NOT USE FOR TUNING
    # ============================================================

    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_preds, labels=[0, 1]
    ).ravel()

    test_metrics = {
        "threshold": best_threshold,
        "accuracy": accuracy_score(y_test, test_preds),
        "balanced_accuracy": balanced_accuracy_score(y_test, test_preds),
        "precision": precision_score(y_test, test_preds, zero_division=0),
        "recall": recall_score(y_test, test_preds, zero_division=0),
        "f1": f1_score(y_test, test_preds, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "ROC_AUC": roc_auc_score(y_test, test_probs),
        "average_precision": average_precision_score(y_test, test_probs),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    print("\nFinal test metrics:")
    print(pd.Series(test_metrics).to_string())

    with open(
        os.path.join(OUTPUT_DIR, "test_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(test_metrics, f, indent=2)


    # ============================================================
    # 8. PRECISION-RECALL CURVE
    # ============================================================

    precision, recall, _ = precision_recall_curve(
        y_test, test_probs
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        label=(
            f"CatBoost "
            f"(AP={test_metrics['average_precision']:.3f})"
        ),
    )
    plt.axhline(
        y=np.mean(y_test),
        linestyle="--",
        label=f"Positive rate ({np.mean(y_test):.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("CatBoost Precision-Recall Curve — Test Set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "precision_recall_curve.png"),
        dpi=150,
    )
    plt.show()


    # ============================================================
    # 9. FEATURE IMPORTANCE
    # ============================================================

    importances = final_model.get_feature_importance()

    feature_importance_df = pd.DataFrame({
        "feature_index": np.arange(X_train.shape[1]),
        "importance": importances,
    }).sort_values("importance", ascending=False)

    feature_importance_df.to_csv(
        os.path.join(OUTPUT_DIR, "feature_importance.csv"),
        index=False,
    )

    plt.figure(figsize=(10, 6))
    top_features = feature_importance_df.head(20).sort_values(
        "importance", ascending=True
    )

    plt.barh(
        top_features["feature_index"].astype(str),
        top_features["importance"],
    )
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Index")
    plt.title("Top CatBoost Features")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "feature_importance.png"),
        dpi=150,
    )
    plt.show()

    print(f"\nSaved model and results in: {OUTPUT_DIR}")
    print("Best model:", model_path)
    print("Grid results:", results_path)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset = np.load("data/processed/link_prediction_dataset.npz")

X_train = np.asarray(dataset["X_train"], dtype=np.float32)
X_val   = np.asarray(dataset["X_val"], dtype=np.float32)
X_test  = np.asarray(dataset["X_test"], dtype=np.float32)

y_train = np.asarray(dataset["y_train"], dtype=np.int64)
y_val   = np.asarray(dataset["y_val"], dtype=np.int64)
y_test  = np.asarray(dataset["y_test"], dtype=np.int64)

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)
# For trusted files you created yourself.
# weights_only=False may be needed for PyG Data objects
# on newer PyTorch versions.
saved = torch.load(
    "data/processed/link_prediction_graphs.pt",
    map_location="cpu",
    weights_only=False,
)

x = saved["x"].to(device)
train_data = saved["train_data"].to(device)
val_data = saved["val_data"].to(device)
test_data = saved["test_data"].to(device)

print("Node features:", x.shape)
print("Train graph:", train_data.edge_index.shape)
print("Validation candidates:", val_data.edge_label_index.shape)
print("Test candidates:", test_data.edge_label_index.shape)

# finetune_gnn()
# finetune_logistic_regression()
# finetune_random_forest()
finetune_catboost()
# finetune_xgboost()