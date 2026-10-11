# ==========================================
# IMPORTS
# ==========================================

# Standard libraries
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Machine learning models
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc
)

plt.use("Agg")

# ==========================================
# TRAIN AND EVALUATE ALL LINK-PREDICTION MODELS
# ==========================================
# Assumes preprocessing has already created:
# train_data, val_data, test_data
# X_train, X_val, X_test
# and the 26 engineered edge features.
#
# Trains:
#   1. Random Forest
#   2. Logistic Regression
#   3. XGBoost
#   4. CatBoost
#   5. GNN
#
# Selects thresholds using VALIDATION data only.
# Evaluates final metrics on TEST data.
# ==========================================

import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
)

# ==========================================
# 1. REPRODUCIBILITY AND LABELS
# ==========================================


edges_path = 'data/input/edges_train.edgelist'
df_attributes = pd.read_csv('data/input/attributes.csv', index_col=0)
df_solution = pd.read_csv('data/input/solutionInput.csv', index_col=0)

import pandas as pd
import torch
import torch.nn.functional as F

# Print the first few rows to inspect what column holds the category
print(df_attributes.head())

# 2. Extract the column containing the category letters ('a', 'b', 'c', etc.)
# If your CSV has the category in the SECOND column (index 1), use iloc[:, 1]. 
# If it's the ONLY column, use iloc[:, 0].
category_col = df_attributes.iloc[:, 0]  # Adjust index if necessary!

# Check unique values to confirm it's just your 6 categories
print("Unique categories found:", category_col.unique())
# Expected output: ['b', 'a', 'c', 'e', 'd', 'f'] (or similar)

# 3. Map letters to integers (0 to 5)
category_mapping = {val: i for i, val in enumerate(sorted(category_col.unique()))}
numeric_cats = category_col.map(category_mapping).values

# Convert to tensor
cat_tensor = torch.tensor(numeric_cats, dtype=torch.long)

# 4. One-hot encode (num_classes should be 6, not 1800)
num_classes = len(category_mapping)
x = F.one_hot(cat_tensor, num_classes=num_classes).float()

print(f"Corrected node feature matrix shape: {x.shape}") 
# Should now correctly print: torch.Size([1800, 6])

import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# ==========================================
# 1. PREPROCESS NODE ATTRIBUTES
# ==========================================

category_col = df_attributes.iloc[:, 0]

category_mapping = {
    val: i
    for i, val in enumerate(sorted(category_col.dropna().unique()))
}

numeric_cats = category_col.map(category_mapping)

if numeric_cats.isna().any():
    raise ValueError("Missing or unmapped node categories found.")

cat_tensor = torch.tensor(
    numeric_cats.to_numpy(),
    dtype=torch.long
)

x = F.one_hot(
    cat_tensor,
    num_classes=len(category_mapping)
).float()

num_nodes = len(df_attributes)

print("Number of nodes:", num_nodes)
print("Feature matrix:", x.shape)

# ==========================================
# 2. LOAD EDGES AND INCLUDE ALL NODES
# ==========================================


G = nx.read_edgelist(
    edges_path,
    delimiter=",",
    nodetype=int,
    create_using=nx.Graph()
)

# IMPORTANT:
# Add every node, including nodes with no edges.
G.add_nodes_from(range(num_nodes))

print("Nodes in graph:", G.number_of_nodes())
print("Undirected edges:", G.number_of_edges())

# Check that node IDs correspond to feature rows 0 ... num_nodes-1.
if set(G.nodes()) != set(range(num_nodes)):
    raise ValueError(
        "Graph node IDs do not match feature row indices. "
        "Map the actual node IDs to the correct attribute rows."
    )

# Convert the graph to PyTorch Geometric
pyg_data = from_networkx(G)

data = Data(
    x=x,
    edge_index=pyg_data.edge_index,
    num_nodes=num_nodes
)

print(data)
print("Edge index shape:", data.edge_index.shape)

assert data.x.size(0) == data.num_nodes
assert data.edge_index.size(1) > 0

import scipy.sparse as sp
import networkx as nx

def compute_katz_scores(graph_data, beta=None, max_path_length=5):
    """
    Compute truncated Katz scores using only graph_data.edge_index.

    K = beta*A + beta^2*A^2 + ... + beta^L*A^L

    Returns:
        G: NetworkX graph used for the calculation
        katz: sparse matrix containing Katz scores
    """
    G = pyg_to_networkx(graph_data)
    num_nodes = graph_data.x.size(0)

    # Sparse adjacency matrix, with node order fixed as 0,...,N-1
    A = nx.to_scipy_sparse_array(
        G,
        nodelist=range(num_nodes),
        format="csr",
        dtype=np.float64
    )
    A = sp.csr_matrix(A)

    # Choose a conservative beta if one was not specified.
    # This is below 1 / maximum degree, which is a safe
    # upper bound for the spectral radius of this undirected graph.
    if beta is None:
        max_degree = max(dict(G.degree()).values(), default=1)
        beta = 0.9 / max(max_degree, 1)

    print(f"Katz beta: {beta:.6f}")

    # Accumulate beta^l * A^l
    power = A.copy()
    katz = beta * power

    for length in range(2, max_path_length + 1):
        power = power @ A
        power.eliminate_zeros()
        katz = katz + (beta ** length) * power

    katz.eliminate_zeros()

    return G, katz.tocsr()

SEED = 42
np.random.seed(SEED)


def pyg_to_networkx(graph_data):
    """Use only the message-passing edges visible in this split."""
    edge_index = graph_data.edge_index.detach().cpu()
    num_nodes = graph_data.x.size(0)

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(
        (int(u), int(v))
        for u, v in edge_index.t().tolist()
        if u != v
    )

    return G


# ==========================================
# 1. ENGINEER CANDIDATE EDGE FEATURES
# ==========================================

def create_edge_features(graph_data, katz):
    """
    Creates 29 symmetric edge features:
      - 18 node attribute features:
          * Sum of endpoint features: 6
          * Absolute difference: 6
          * Elementwise product: 6
      - 11 structural features:
          * Same category
          * Common neighbors
          * Jaccard similarity
          * Preferential attachment
          * Adamic-Adar
          * Resource allocation
          * Shortest-path distance
          * Katz score
          * Same community
          * Same-attribute common neighbors
          * Endpoint homophily
    """
    x = graph_data.x.detach().cpu().numpy().astype(np.float32)

    candidate_edges = (
        graph_data.edge_label_index.detach().cpu().t().tolist()
    )

    G = pyg_to_networkx(graph_data)

    # Category ID for each node, derived from its one-hot features.
    category_ids = np.argmax(x, axis=1)

    # ------------------------------------------
    # Community detection on this split's graph
    # ------------------------------------------
    from networkx.algorithms.community import (
        greedy_modularity_communities,
    )

    communities = list(greedy_modularity_communities(G))

    node_community = {}
    for community_id, community in enumerate(communities):
        for node in community:
            node_community[node] = community_id

    # Ensure any unassigned nodes have their own community.
    next_community_id = len(communities)
    for node in G.nodes():
        if node not in node_community:
            node_community[node] = next_community_id
            next_community_id += 1

    # ------------------------------------------
    # Node homophily on this split's graph
    # ------------------------------------------
    node_homophily = {}

    for node in G.nodes():
        node_neighbors = list(G.neighbors(node))

        if not node_neighbors:
            node_homophily[node] = 0.0
        else:
            matching_neighbors = sum(
                category_ids[neighbor] == category_ids[node]
                for neighbor in node_neighbors
            )

            node_homophily[node] = (
                matching_neighbors / len(node_neighbors)
            )

    # ------------------------------------------
    # Neighborhood and degree information
    # ------------------------------------------
    neighbors = {
        node: set(G.neighbors(node))
        for node in range(x.shape[0])
    }

    degrees = dict(G.degree())
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

    rows = []

    for u, v in candidate_edges:
        u, v = int(u), int(v)

        xu, xv = x[u], x[v]

        # 18 symmetric node-attribute features
        category_sum = xu + xv
        category_diff = np.abs(xu - xv)
        category_product = xu * xv

        same_category = float(
            category_ids[u] == category_ids[v]
        )

        # Structural features from the split-specific graph
        nu = neighbors[u]
        nv = neighbors[v]
        common = nu & nv

        common_count = len(common)
        union_count = len(nu | nv)

        jaccard = (
            common_count / union_count
            if union_count else 0.0
        )

        preferential_attachment = degrees[u] * degrees[v]

        adamic_adar = sum(
            1.0 / np.log(degrees[w])
            for w in common
            if degrees[w] > 1
        )

        resource_allocation = sum(
            1.0 / degrees[w]
            for w in common
            if degrees[w] > 0
        )

        distance = shortest_paths.get(u, {}).get(
            v, float("inf")
        )

        if np.isinf(distance):
            distance = -1

        katz_score = float(katz[u, v])

        # --------------------------------------
        # Three additional features
        # --------------------------------------

        # 1. Do the endpoints belong to the same community?
        same_community = float(
            node_community[u] == node_community[v]
        )

        # 2. Count shared neighbors with the same category
        # as both endpoints. Zero if endpoint categories differ.
        if same_category and common:
            same_attr_common_neighbors = sum(
                category_ids[w] == category_ids[u]
                for w in common
            )
        else:
            same_attr_common_neighbors = 0

        # 3. Average homophily of the two endpoints
        endpoint_homophily = (
            node_homophily[u] + node_homophily[v]
        ) / 2.0

        # Preserve the original 26 feature positions, then append
        # the three new features in the metadata order below.
        structural_features = np.array([
            same_category,
            common_count,
            jaccard,
            preferential_attachment,
            adamic_adar,
            resource_allocation,
            distance,
            katz_score,
            same_community,
            same_attr_common_neighbors,
            endpoint_homophily,
        ], dtype=np.float32)

        row = np.concatenate([
            category_sum,
            category_diff,
            category_product,
            structural_features,
        ])

        rows.append(row)

    return np.asarray(rows, dtype=np.float32)


# ==========================================
# 2. GNN WITH MLP EDGE DECODER
# ==========================================
class Net(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        out_channels=16,
        edge_feature_channels=26,
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

        edge_embedding = torch.cat([
            src + dst,
            torch.abs(src - dst),
            src * dst,
            edge_features,
        ], dim=-1)

        return self.decoder(edge_embedding).squeeze(-1)


# ==========================================
# 3. SPLIT EDGES INTO TRAIN / VAL / TEST
# ==========================================
torch.manual_seed(42)
np.random.seed(42)

splitter = RandomLinkSplit(
    num_val=0.10,
    num_test=0.10,
    is_undirected=True,
    add_negative_train_samples=True,
    neg_sampling_ratio=1.0,
    disjoint_train_ratio=0.1,
)


train_data, val_data, test_data = splitter(data)


def canonical_edges(edge_index):
    """Represent undirected edges as sorted node pairs."""
    edges = edge_index.detach().cpu().numpy().T
    return {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in edges
        if u != v
    }


def check_no_target_edge_leakage(split_data, split_name):
    graph_edges = canonical_edges(split_data.edge_index)

    labels = split_data.edge_label.detach().cpu().numpy().ravel()
    label_edges = split_data.edge_label_index

    positive_edges = canonical_edges(
        label_edges[:, labels == 1]
    )

    leaked_edges = graph_edges.intersection(positive_edges)

    assert not leaked_edges, (
        f"{split_name} leakage detected: "
        f"{len(leaked_edges)} positive target edges "
        "are present in edge_index. "
        f"Examples: {list(leaked_edges)[:5]}"
    )

    print(
        f"{split_name}: OK — "
        f"{len(positive_edges)} positive targets checked; "
        "none are in the message-passing graph."
    )


check_no_target_edge_leakage(train_data, "Train")
check_no_target_edge_leakage(val_data, "Validation")
check_no_target_edge_leakage(test_data, "Test")

print(
    "Training message-passing edges:",
    train_data.edge_index.size(1)
)
print(
    "Validation candidate edges:",
    val_data.edge_label_index.size(1)
)
print(
    "Test candidate edges:",
    test_data.edge_label_index.size(1)
)


# ==========================================
# 4. COMPUTE ENGINEERED EDGE FEATURES
# ==========================================
print("\nComputing Katz scores...")
G_train = pyg_to_networkx(train_data)
max_degree = max(dict(G_train.degree()).values(), default=1)
katz_beta = 0.9 / max(max_degree, 1)

G_train, katz_train = compute_katz_scores(
    train_data, beta=katz_beta
)
G_val, katz_val = compute_katz_scores(
    val_data, beta=katz_beta
)
G_test, katz_test = compute_katz_scores(
    test_data, beta=katz_beta
)

print("Katz matrices calculated.")

print("Creating edge features...")

X_train = create_edge_features(train_data, katz_train)
X_val = create_edge_features(val_data, katz_val)
X_test = create_edge_features(test_data, katz_test)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)


assert len(X_train) == train_data.edge_label.numel()
assert len(X_val) == val_data.edge_label.numel()
assert len(X_test) == test_data.edge_label.numel()


# ==========================================
# 5. INITIALIZE MODEL AND TENSORS
# ==========================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

y_train = (
    train_data.edge_label.detach().cpu().numpy().astype(int)
)
y_val = (
    val_data.edge_label.detach().cpu().numpy().astype(int)
)
y_test = (
    test_data.edge_label.detach().cpu().numpy().astype(int)
)


# Keep candidate-edge order identical to feature/label order.
assert len(X_train) == len(y_train)
assert len(X_val) == len(y_val)
assert len(X_test) == len(y_test)

assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]

print("Device:", device)
print("Training examples:", len(y_train))
print("Validation examples:", len(y_val))
print("Test examples:", len(y_test))
print("Features per edge:", X_train.shape[1])


# ==========================================
# 2. THRESHOLD SELECTION
# ==========================================

def find_best_f1_threshold(y_true, probabilities):
    """Choose a classification threshold using validation data."""
    thresholds = np.arange(0.05, 0.96, 0.01)

    scores = [
        f1_score(
            y_true,
            (probabilities >= threshold).astype(int),
            zero_division=0,
        )
        for threshold in thresholds
    ]

    best_idx = int(np.argmax(scores))

    return float(thresholds[best_idx]), float(scores[best_idx])

# ==========================================
# 6. SAVE REUSABLE DATASET FOR ALL MODELS
# ==========================================
import os
import json
import numpy as np
import torch

OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keep labels in CPU NumPy arrays
y_train = train_data.edge_label.detach().cpu().numpy().astype(np.int64)
y_val = val_data.edge_label.detach().cpu().numpy().astype(np.int64)
y_test = test_data.edge_label.detach().cpu().numpy().astype(np.int64)

# Save features and labels for classical ML models
np.savez_compressed(
    os.path.join(OUTPUT_DIR, "link_prediction_dataset.npz"),
    X_train=X_train.astype(np.float32),
    X_val=X_val.astype(np.float32),
    X_test=X_test.astype(np.float32),
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,
)

# Save graph information for GNN models.
# These objects are still on the CPU at this point.
torch.save(
    {
        "x": data.x.detach().cpu(),
        "train_data": train_data.cpu(),
        "val_data": val_data.cpu(),
        "test_data": test_data.cpu(),
    },
    os.path.join(OUTPUT_DIR, "link_prediction_graphs.pt"),
)

# Document the exact feature order

feature_names = (
    [f"category_sum_{c}" for c in "abcdef"]
    + [f"category_abs_diff_{c}" for c in "abcdef"]
    + [f"category_product_{c}" for c in "abcdef"]
    + [
        "same_category",
        "common_neighbors",
        "jaccard",
        "preferential_attachment",
        "adamic_adar",
        "resource_allocation",
        "shortest_path_distance",
        "katz_score",
        "same_community",
        "same_attr_common_neighbors",
        "endpoint_homophily",
    ]
)

assert len(feature_names) == 29

metadata = {
    "seed": SEED,
    "num_features": len(feature_names),
    "feature_names": feature_names,
    "split": {
        "train": 0.80,
        "validation": 0.10,
        "test": 0.10,
    },
    "split_method": "PyG RandomLinkSplit",
    "is_undirected": True,
    "disjoint_train_ratio": 0.1,
    "negative_sampling_ratio": 1.0,
    "katz_beta": float(katz_beta),
}

with open(
    "data/processed/feature_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=2)

print("Saved", len(feature_names), "feature names.")

print("\nDataset saved successfully:")
print(f"  {OUTPUT_DIR}/link_prediction_dataset.npz")
print(f"  {OUTPUT_DIR}/link_prediction_graphs.pt")
print(f"  {OUTPUT_DIR}/feature_metadata.json")

print("\nSaved array shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
print("X_test: ", X_test.shape, "y_test: ", y_test.shape)