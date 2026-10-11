"""Tells you WHY the node-id alignment check fails.

Run from the project root:  python diagnose_alignment.py
"""

import os
import time
from pathlib import Path

import networkx as nx
import torch

EDGES_PATH = Path("data/input/edges_train.edgelist")
GRAPHS_PATH = Path("data/processed/link_prediction_graphs.pt")


def canon(pairs):
    return {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in pairs
        if int(a) != int(b)
    }


saved = torch.load(GRAPHS_PATH, map_location="cpu", weights_only=False)
n = saved["x"].size(0)

print("Saved graph file:", GRAPHS_PATH.resolve())
print("  last written:  ", time.ctime(os.path.getmtime(GRAPHS_PATH)))
print("  current time:  ", time.ctime())

G = nx.read_edgelist(
    EDGES_PATH, delimiter=",", nodetype=int, create_using=nx.Graph()
)
G.add_nodes_from(range(n))
order = list(G.nodes())
print("\nFirst 10 nodes in the order they appear in the edge list:", order[:10])
print("Identity order would be:                                   ", list(range(10)))

original = canon(G.edges())

# What from_networkx() did with the OLD graph-building code
mapping = {node: i for i, node in enumerate(order)}
scrambled = canon((mapping[u], mapping[v]) for u, v in G.edges())

td = saved["test_data"]
labels = td.edge_label.detach().cpu().numpy().ravel()
pos = td.edge_label_index.detach().cpu()[:, torch.as_tensor(labels == 1)]
saved_edges = (
    canon(td.edge_index.detach().cpu().t().tolist()) | canon(pos.t().tolist())
)

identity_overlap = len(saved_edges & original) / len(original)
scrambled_overlap = len(saved_edges & scrambled) / len(original)

print(f"\nSaved graphs match original node ids:      {identity_overlap:.1%}")
print(f"Saved graphs match OLD scrambled numbering: {scrambled_overlap:.1%}")

if identity_overlap > 0.999:
    print("\nResult: saved graphs are correct. Re-run make_submission.py.")
elif scrambled_overlap > 0.999:
    print(
        "\nResult: the saved files were built with the OLD node order.\n"
        "Preprocessing has not been re-run with the fix, or the fix is not\n"
        "in effect, or it wrote to a different folder than the one above."
    )
else:
    print(
        "\nResult: matches neither numbering. The saved graphs come from a\n"
        "different edge list or a different pipeline than data/input/."
    )
