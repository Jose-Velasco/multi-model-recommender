import torch
from torch_geometric.utils import to_undirected, degree, is_undirected, contains_self_loops
from networkx import Graph

def graph_info(data):
    print(f"Summary")
    print("-" * 40)

    # --- Node & edge basics ---
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges} "
          f"({'undirected' if data.is_undirected() else 'directed'})")

    # --- Feature info ---
    if getattr(data, "x", None) is not None:
        print(f"Node features: {data.x.shape[1]} dims")
    else:
        print("Node features: None")

    if getattr(data, "edge_attr", None) is not None:
        print(f"Edge features: {data.edge_attr.shape[1]} dims")

    if getattr(data, "y", None) is not None:
        print(f"Labels: {data.y.shape} | dtype={data.y.dtype}")

    # --- Self-loops, duplicates, isolated nodes ---
    print(f"Self-loops: {contains_self_loops(data.edge_index)}")
    if data.num_nodes < 1e6:  # avoid big tensors
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        isolated = (deg == 0).sum().item()
        print(f"Isolated nodes: {isolated} / {data.num_nodes}")
        print(f"Mean degree: {deg.mean():.2f} | "
              f"Max degree: {deg.max().item() if deg.numel() else 0}")

    # --- Density ---
    possible_edges = data.num_nodes * (data.num_nodes - 1)
    density = data.num_edges / possible_edges if possible_edges else 0
    print(f"Density: {density:.6f}")

    # --- Data types & device ---
    print(f"Data device: {data.edge_index.device}")
    print(f"Data types: x={data.x.dtype if getattr(data, 'x', None) is not None else None}, "
          f"edge_index={data.edge_index.dtype}")

    # --- Memory footprint estimate ---
    total_bytes = 0
    for k, v in data.items():
        if torch.is_tensor(v):
            total_bytes += v.numel() * v.element_size()
    print(f"Approx. tensor memory: {total_bytes / 1e6:.2f} MB")

    # --- Check for unbalanced labels (classification) ---
    if getattr(data, "y", None) is not None and data.y.ndim == 1:
        num_classes = int(data.y.max()) + 1 if data.y.numel() else 0
        class_counts = torch.bincount(data.y).tolist()
        print(f"Classes: {num_classes} | Label distribution: {class_counts}")

    print("-" * 40)

def get_item_user_node_id_inv_map(graph) -> tuple[dict[int, int], dict[int, int]]:
    user_node_id_inv_map = {v: k for k, v in graph.users_node_id_map.items()} # pyright: ignore[reportAttributeAccessIssue]
    item_node_id_inv_map = {v: k for k, v in graph.item_node_id_map.items()} # pyright: ignore[reportAttributeAccessIssue]
    return user_node_id_inv_map, item_node_id_inv_map

def generate_node_labels(graph: Graph, user_node_id_inv_map: dict[int, int], item_node_id_inv_map: dict[int, int]) -> dict[int, str]:
    # Label nodes so networkx can assign a node the type of node (User | Item) + their user/item_id (not node_id)
    labels: dict[int, str] = {}
    for node_id in graph.nodes():
        if node_id in user_node_id_inv_map:
            labels[node_id] = f"U{user_node_id_inv_map[node_id]}"
        elif node_id in item_node_id_inv_map:
            labels[node_id] = f"I{item_node_id_inv_map[node_id]}"
        else:
            labels[node_id] = str(node_id)
    return labels

def top_degree_subgraph(G: Graph, top_n: int = 1000):
    """
    sampling strategy used to extract the most connected (high-degree) nodes 
    from a large graph so that its small enough to visualize or analyze clearly.
    keeps only the most connected (“important”) nodes
    Preserves the most active users/items or central entities.
    """
    deg = dict(G.degree())
    top_nodes = [n for n,_ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    return G.subgraph(top_nodes).copy()