"""Feature code shared by preprocessing and submission.

These three functions are copied unchanged from the preprocessing script.
Import them there too, so training features and solution features always
come from one definition:

    from link_features import pyg_to_networkx, compute_katz_scores, create_edge_features
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized, svds
edges_path = 'data/input/edges_train.edgelist'

def pyg_to_networkx(graph_data):
    """Use only the message-passing edges visible in this split."""
    edge_index = graph_data.edge_index.detach().cpu()
    num_nodes = graph_data.x.size(0)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))        # fixes node order BEFORE edges are added
    G.add_edges_from(
        nx.read_edgelist(edges_path, delimiter=",", nodetype=int).edges()
    )

    # Guard against this coming back
    assert list(G.nodes()) == list(range(num_nodes)), "Node order is not 0..N-1"

    return G


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

    A = nx.to_scipy_sparse_array(
        G,
        nodelist=range(num_nodes),
        format="csr",
        dtype=np.float64
    )
    A = sp.csr_matrix(A)

    if beta is None:
        max_degree = max(dict(G.degree()).values(), default=1)
        beta = 0.9 / max(max_degree, 1)

    print(f"Katz beta: {beta:.6f}")

    power = A.copy()
    katz = beta * power

    for length in range(2, max_path_length + 1):
        power = power @ A
        power.eliminate_zeros()
        katz = katz + (beta ** length) * power

    katz.eliminate_zeros()

    return G, katz.tocsr()


def create_edge_features(graph_data, katz, simrank_iterations=8,
                         simrank_decay=0.8, low_rank_dim=8):
    """
    Creates 34 symmetric edge features.

    Original 34 features:
      - 18 endpoint attribute features
      - 11 structural features

    Added 5 features:
      - Rooted PageRank (symmetrized)
      - Hitting time (minimum of both directions)
      - Commute time
      - SimRank
      - Low-rank adjacency similarity

    All graph-derived features use graph_data.edge_index only.
    """

    x = graph_data.x.detach().cpu().numpy().astype(np.float32)
    n = x.shape[0]

    candidate_edges = (
        graph_data.edge_label_index.detach().cpu().t().tolist()
    )

    G = pyg_to_networkx(graph_data)
    G.add_nodes_from(range(n))
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.Graph(G)

    category_ids = np.argmax(x, axis=1)

    # --------------------------------------------------
    # Adjacency and transition matrices
    # --------------------------------------------------
    A = nx.to_scipy_sparse_array(
        G,
        nodelist=range(n),
        dtype=np.float64,
        format="csr",
    )
    A = sp.csr_matrix(A)

    degrees_array = np.asarray(A.sum(axis=1)).ravel()
    inverse_degrees = np.zeros(n, dtype=np.float64)
    nonzero = degrees_array > 0
    inverse_degrees[nonzero] = 1.0 / degrees_array[nonzero]

    # P[u, v] = probability of moving from u to v.
    P = sp.diags(inverse_degrees) @ A
    P = P.tocsr()

    # --------------------------------------------------
    # Existing feature: greedy modularity communities
    # --------------------------------------------------
    from networkx.algorithms.community import (
        greedy_modularity_communities,
    )

    node_community = {node: node for node in range(n)}

    if G.number_of_edges() > 0:
        communities = list(greedy_modularity_communities(G))
        node_community = {}

        for community_id, community in enumerate(communities):
            for node in community:
                node_community[node] = community_id

        for node in range(n):
            node_community.setdefault(node, len(node_community))

    # --------------------------------------------------
    # Existing feature: endpoint homophily
    # --------------------------------------------------
    neighbors = {
        node: set(G.neighbors(node))
        for node in range(n)
    }

    degrees = dict(G.degree())

    node_homophily = {}
    for node in range(n):
        node_neighbors = neighbors[node]

        if not node_neighbors:
            node_homophily[node] = 0.0
        else:
            node_homophily[node] = sum(
                category_ids[v] == category_ids[node]
                for v in node_neighbors
            ) / len(node_neighbors)

    # --------------------------------------------------
    # New feature 1: Rooted PageRank
    # --------------------------------------------------
    alpha = 0.85
    rooted_pagerank = {}

    candidate_nodes = {
        int(node)
        for edge in candidate_edges
        for node in edge
    }

    pagerank_system = (
        sp.eye(n, format="csc") - alpha * P.T.tocsc()
    )
    solve_pagerank = factorized(pagerank_system)

    for source in candidate_nodes:
        restart = np.zeros(n, dtype=np.float64)
        restart[source] = 1.0

        rooted_pagerank[source] = solve_pagerank(
            (1.0 - alpha) * restart
        )

    # --------------------------------------------------
    # New features 2 and 3: Hitting and commute times
    # --------------------------------------------------
    component_id = {}
    component_volume = {}
    laplacian_pinv = {}

    for cid, component in enumerate(nx.connected_components(G)):
        nodes = sorted(component)
        component_volume[cid] = sum(degrees[u] for u in nodes)

        for u in nodes:
            component_id[u] = cid

        if len(nodes) == 1:
            laplacian_pinv[cid] = (
                nodes,
                np.zeros((1, 1), dtype=np.float64),
            )
            continue

        subgraph = G.subgraph(nodes)
        L = nx.laplacian_matrix(
            subgraph, nodelist=nodes
        ).toarray().astype(np.float64)

        L_pinv = np.linalg.pinv(L)
        laplacian_pinv[cid] = (nodes, L_pinv)

    component_local_index = {
        cid: {node: i for i, node in enumerate(nodes)}
        for cid, (nodes, _) in laplacian_pinv.items()
    }

    def random_walk_times(u, v):
        """Return symmetric hitting time and commute time."""
        if component_id[u] != component_id[v]:
            return -1.0, -1.0

        cid = component_id[u]
        nodes, L_pinv = laplacian_pinv[cid]
        local = component_local_index[cid]

        i = local[u]
        j = local[v]
        volume = component_volume[cid]

        h_uv = volume * (L_pinv[j, j] - L_pinv[i, j])
        h_vu = volume * (L_pinv[i, i] - L_pinv[i, j])

        h_uv = max(float(h_uv), 0.0)
        h_vu = max(float(h_vu), 0.0)

        hitting_time = min(h_uv, h_vu)
        commute_time = h_uv + h_vu

        return hitting_time, commute_time

    # --------------------------------------------------
    # New feature 4: SimRank
    # --------------------------------------------------
    S = np.eye(n, dtype=np.float64)

    for _ in range(simrank_iterations):
        S_next = simrank_decay * (P @ S @ P.T)
        S_next = np.asarray(S_next)
        np.fill_diagonal(S_next, 1.0)
        S = S_next

    # --------------------------------------------------
    # New feature 5: Low-rank adjacency approximation
    # --------------------------------------------------
    if n > 2 and low_rank_dim >= 1:
        k = min(low_rank_dim, n - 1)

        U, singular_values, _ = svds(
            A.astype(np.float64),
            k=k,
            which="LM",
            v0=np.random.RandomState(42).normal(size=n),
        )

        order = np.argsort(singular_values)[::-1]
        U = U[:, order]
        singular_values = singular_values[order]

        low_rank_embedding = (
            U * np.sqrt(np.maximum(singular_values, 0.0))
        )
    else:
        low_rank_embedding = np.zeros((n, 1), dtype=np.float64)

    # --------------------------------------------------
    # Original feature engineering + new methods
    # --------------------------------------------------
    rows = []

    for u, v in candidate_edges:
        u, v = int(u), int(v)

        xu, xv = x[u], x[v]

        had_edge = G.has_edge(u, v)
        if had_edge:
            G.remove_edge(u, v)

        category_sum = xu + xv
        category_diff = np.abs(xu - xv)
        category_product = xu * xv

        same_category = float(category_ids[u] == category_ids[v])

        nu = set(G.neighbors(u))
        nv = set(G.neighbors(v))
        common = nu & nv

        common_count = len(common)
        union_count = len(nu | nv)

        jaccard = (
            common_count / union_count if union_count else 0.0
        )

        deg_u = G.degree(u)
        deg_v = G.degree(v)
        preferential_attachment = deg_u * deg_v

        adamic_adar = sum(
            1.0 / np.log(G.degree(w))
            for w in common
            if G.degree(w) > 1
        )

        resource_allocation = sum(
            1.0 / G.degree(w)
            for w in common
            if G.degree(w) > 0
        )

        try:
            distance = float(nx.shortest_path_length(G, u, v))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            distance = -1.0

        if had_edge:
            G.add_edge(u, v)

        katz_score = float(katz[u, v])

        same_community = float(
            node_community[u] == node_community[v]
        )

        same_attr_common_neighbors = (
            sum(category_ids[w] == category_ids[u] for w in common)
            if same_category else 0
        )

        endpoint_homophily = (
            node_homophily[u] + node_homophily[v]
        ) / 2.0

        original_structural = [
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
        ]

        rpr_uv = rooted_pagerank[u][v]
        rpr_vu = rooted_pagerank[v][u]
        rooted_pagerank_score = (rpr_uv + rpr_vu) / 2.0

        hitting_time, commute_time = random_walk_times(u, v)

        simrank_score = float(S[u, v])

        low_rank_score = float(
            np.dot(low_rank_embedding[u], low_rank_embedding[v])
        )

        new_structural = [
            rooted_pagerank_score,
            hitting_time,
            commute_time,
            simrank_score,
            low_rank_score,
        ]

        row = np.concatenate([
            category_sum,
            category_diff,
            category_product,
            np.asarray(original_structural, dtype=np.float32),
            np.asarray(new_structural, dtype=np.float32),
        ])

        rows.append(row)

    return np.asarray(rows, dtype=np.float32)
