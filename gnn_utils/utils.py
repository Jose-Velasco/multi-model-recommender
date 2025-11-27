from pathlib import Path
import numpy as np
import torch
from torch_geometric.utils import degree, contains_self_loops
from networkx import Graph
from torch_geometric.data import Data
from typing import Literal, Optional
from torchmetrics import MetricCollection
import math, copy
from torchmetrics.retrieval import (
    RetrievalRecall,
    RetrievalPrecision,
    RetrievalMAP,
    RetrievalNormalizedDCG,
)
from collections import defaultdict
from dateutil import tz
import datetime

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
    print(f"Data types:")
    for k in data.keys():
        if torch.is_tensor(data[k]):
            print(f"    {k}={data[k].dtype}, shape={data[k].shape}")
        else:
            print(f"    {k}={type(data[k])}, len={len(data[k])}")
        # if getattr(data, k, None):

    # print(f"Data types: x={data.x.dtype if getattr(data, 'x', None) is not None else None}, "
        #   f"edge_index={data.edge_index.dtype}")

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
    """
    Gets mapping from node_id as key to its original entity_id as value

    ex. user_node_id_inv_map has key of node_id that returns the associated user_id
    """
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
    deg = dict(G.degree()) # pyright: ignore[reportCallIssue]
    top_nodes = [n for n,_ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    return G.subgraph(top_nodes).copy()

def dict_to_tensor(
    mapping: dict[int | str, float],
    num_nodes: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    default: float = 0.0,
) -> torch.Tensor:
    """
    Convert a node-indexed dictionary (e.g., NetworkX centrality output)
    into a dense torch tensor aligned by node index [0..num_nodes-1].

    Parameters
    ----------
    mapping : dict
        Dictionary where keys are node indices (int or str convertible to int),
        and values are scalars (float).
    num_nodes : int
        Total number of nodes (tensor length).
    dtype : torch.dtype, optional
        Tensor dtype (default: torch.float32).
    device : torch.device, optional
        Target device (default: CPU).
    default : float, optional
        Default value for missing nodes (default: 0.0).

    Returns
    -------
    torch.Tensor
        Tensor of shape [num_nodes], aligned to node IDs.
    """
    t = torch.full((num_nodes,), fill_value=default, dtype=dtype, device=device)
    for node_id, v in mapping.items():
        idx = int(node_id)
        if 0 <= idx < num_nodes:
            t[idx] = float(v)
    return t

def add_node_attr(graph: Data, features: torch.Tensor, attr_name: Optional[str] = None) -> Data:
    """
    if attr_name provided, adds features as a attribute to  the graph object with key=attr_name
    else
    
    concatenates the features to the node features So each node's feature vector just gets extended with the new features.
    """
    assert graph.x is not None
    if attr_name is None:
        x = graph.x
        x = torch.concat((x, features), dim=1)
        graph.x = x
    else:
        graph[attr_name] = features
    return graph

def add_edge_attr(graph: Data, features: torch.Tensor, attr_name: Optional[str] = None) -> Data:
    """
    Safely add or concatenate edge-level features to a PyG graph.

    if attr_name provided, adds features as a attribute to the graph object with key=attr_name
    else

    concatenates the features to the edge features So each edge's feature vector just gets extended with the new features.

    Parameters
    ----------
    graph : torch_geometric.data.Data
        The input graph.
    features : torch.Tensor
        Tensor of shape [num_edges, num_new_features].
    attr_name : str or None, optional
        - If None: concatenate features to existing graph.edge_attr (create if missing).
        - If str: add as a new attribute (graph[attr_name] = features).
    """
    num_edges = graph.edge_index.size(1) # pyright: ignore[reportOptionalMemberAccess]
    assert features.size(0) == num_edges, (
        f"Feature rows ({features.size(0)}) must match number of edges ({num_edges})."
    )

    # Match dtype/device with existing edge_attr (if any)
    if graph.edge_attr is not None:
        features = features.to(graph.edge_attr.device, dtype=graph.edge_attr.dtype)

    if attr_name is None:
        if graph.edge_attr is None:
            graph.edge_attr = features
        else:
            graph.edge_attr = torch.cat([graph.edge_attr, features], dim=1)
    else:
        graph[attr_name] = features
    return graph

def normalize_zscore(x: torch.Tensor, dim: int = 0, eps: float = 1e-12) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    # guard against zero std (constant feature)
    # if any feature column (or row) in x has zero variance (i.e. every value is identical),
    # then std == 0
    # chooses element-wise between a and b
    # It checks each entry of std:
    # If std[i] < eps (≈ 0, constant feature), replace it with 1.0.
    # Else, keep the computed std.
    std = torch.where(std < eps, torch.ones_like(std), std)
    return (x - mean) / (std + eps)

def log1p_standardize(x, dim=0, eps=1e-12) -> torch.Tensor:
    x = torch.clamp(x, min=0)
    x = torch.log1p(x)
    return normalize_zscore(x, dim=dim, eps=eps)

def drop_zero_columns(tensor: torch.Tensor, verbose: bool = True) -> torch.Tensor:
    """
    Removes columns from a 2D tensor that are entirely zero (sum == 0).

    Args:
        tensor (torch.Tensor): Input tensor of shape [N, D].
        verbose (bool): If True, prints how many columns were kept/dropped.

    Returns:
        torch.Tensor: Tensor with only non-zero columns retained.
    """
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tuple(tensor.shape)}")

    # Mask columns whose absolute-sum > 0
    mask = tensor.abs().sum(dim=0) > 0

    if not mask.any():
        if verbose:
            print("[drop_zero_columns] All columns are zero. Returning original tensor.")
        return tensor

    filtered = tensor[:, mask]

    if verbose:
        dropped = (~mask).sum().item()
        kept = mask.sum().item()
        print(f"[drop_zero_columns] Kept {kept} / {mask.numel()} columns; dropped {dropped}.")

    return filtered

def normalize_metrics_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            v = v.detach().cpu()
            if v.numel() == 1:
                out[k] = v.item()         # scalar
            else:
                out[k] = v.tolist()       # small vector
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out

def metrics_tracker_factory(top_k: int, aggregation: Literal['mean', 'median', 'min', 'max']) -> MetricCollection:
    metricCollection = MetricCollection([
        RetrievalRecall(top_k=top_k, aggregation=aggregation),
        RetrievalPrecision(top_k=top_k, aggregation=aggregation),
        RetrievalMAP(top_k=top_k, aggregation=aggregation),
        RetrievalNormalizedDCG(top_k=top_k, aggregation=aggregation),

    ])

    return metricCollection
def generate_now_timestamp_str(time_zone_info: datetime.tzinfo | None = tz.gettz('America/Los_Angeles')) -> str:
    # get the current time and set it up into a readable string for the file paths
    current_time = datetime.datetime.now(time_zone_info)
    date_string = f"{current_time.year}-{current_time.month}-{current_time.day}"
    time_string = f"{current_time.hour}-{current_time.minute}-{current_time.second}"
    date_time_path_str = date_string + "_" + time_string
    return date_time_path_str

# [1] Chatgpt was used to implement EarlyStopper for stopping training early if validation start increasing to much.
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0, mode="min", restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.best = math.inf if mode == "min" else -math.inf
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, metric, model=None):
        improved = metric < (self.best - self.min_delta) if self.mode=="min" else metric > (self.best + self.min_delta)
        if improved:
            self.best = metric
            self.counter = 0
            if model and self.restore_best:
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def load_best(self, model):
        if self.restore_best and self.best_state:
            model.load_state_dict(self.best_state)

# def add_edges_to_dict(data: Data, user_seen_items: defaultdict[int, set[int]]):
#     """
#     user_seen_items: key should be the user's node id and value is the set of items the
#     user has interacted with
#     """
#     users = data.edge_label_index[0].tolist()
#     items = data.edge_label_index[1].tolist()
#     for u, i in zip(users, items):
#         user_seen_items[int(u)].add(i)

def aggregate_items_user_has_interacted_with(src_nodes: torch.Tensor, dst_nodes: torch.Tensor, user_node_id_unique: set) ->defaultdict[int, set[int]]:
    """
    user_seen_items: key should be the user's node id and value is the set of items the
    user has interacted with
    """
    user_seen_items: defaultdict[int, set[int]] = defaultdict(set)
    for edge_idx, src_node in enumerate(src_nodes):
        if src_node.item() in user_node_id_unique:
            item_dst_node = int(dst_nodes[edge_idx])
            # means this user has interacted with this item node
            user_seen_items[int(src_node)].add(item_dst_node)
    return user_seen_items

# def get_save_allowed_items_per_user(
#         save_allowed_items_per_user_file_path: str,
#         train_data: Data,
#         val_data: Data,
#         test_data: Data,
#         item_node_id_map: dict[int, int]):
#     """
#     if allowed items per user is saved loads it else builds it and saves it to avoid
#     having to rebuild it every time.
#     """
#     if Path(save_allowed_items_per_user_file_path).exists():
#         print(f"Loading found saved allowed dict per user")
#         allowed_items_per_user = torch.load(save_allowed_items_per_user_file_path)
#     else:
#         print(f"Rebuilding: did not find precomputed saved allowed dict at {save_allowed_items_per_user_file_path}")
#         user_seen_items: defaultdict[int, set[int]] = defaultdict(set)
#         add_edges_to_dict(train_data, user_seen_items=user_seen_items)
#         add_edges_to_dict(val_data, user_seen_items=user_seen_items)
#         add_edges_to_dict(test_data, user_seen_items=user_seen_items)
#         all_items: set[int] = set(item_node_id_map.values())
#         allowed_items_per_user = build_allowed_items_per_user(
#             all_items=all_items,
#             user_seen_items=user_seen_items,
#         )
#         torch.save(allowed_items_per_user, save_allowed_items_per_user_file_path)
#     return allowed_items_per_user

def get_save_allowed_items_per_user(
        save_allowed_items_per_user_file_path: str,
        graph: Data,
        user_to_node_id_map: dict[int, int],
        item_node_id_map: dict[int, int]):
    """
    if allowed items per user is saved loads it else builds it and saves it to avoid
    having to rebuild it every time.

    Assumes bipartite graph of user and items nodes
    """
    if Path(save_allowed_items_per_user_file_path).exists():
        print(f"Loading found saved allowed dict per user")
        allowed_items_per_user = torch.load(save_allowed_items_per_user_file_path)
    else:
        print(f"Rebuilding: did not find precomputed saved allowed dict at {save_allowed_items_per_user_file_path}")
        assert graph.edge_index is not None, f"Error: graph did not have .edge_index attribute required but has {graph}"
        user_seen_items: defaultdict[int, set[int]] = aggregate_items_user_has_interacted_with(
            src_nodes=graph.edge_index[0],
            dst_nodes=graph.edge_index[1],
            user_node_id_unique=set(user_to_node_id_map.values())
        )
        
        # user_node_id_unique = set(user_to_node_id_map.values())
        # # user_node_id = torch.tensor(list(user_to_node_id_map.values()))
        # item_node_id_unique = set(item_node_id_map.values())

        # for col, src_node in enumerate(graph.edge_index[0].tolist()):
        #     if src_node in user_node_id_unique:
        #         item_dst_node = int(graph.edge_index[1])
        #         # means this user has interacted with this item node
        #         user_seen_items[src_node].add(item_dst_node)

        all_items_unique: set[int] = set(item_node_id_map.values())
        allowed_items_per_user = build_allowed_items_per_user(
            all_items=all_items_unique,
            user_seen_items=user_seen_items,
        )
        torch.save(allowed_items_per_user, save_allowed_items_per_user_file_path)
    return allowed_items_per_user

def build_allowed_items_per_user(
    all_items: set[int],
    user_seen_items: defaultdict[int, set[int]],
) -> dict[int, torch.Tensor]:
    """
    Precompute, for each user (global node ID), the list of items (global IDs)
    that are valid negatives (= all_items minus items they've interacted with).

    This is done ONCE before training/eval to avoid having to do this
    every batch during validation
    """
    allowed_items_per_user: dict[int, torch.Tensor] = {}

    all_items_list = list(all_items)
    all_items_set = set(all_items_list)

    for u, seen in user_seen_items.items():
        # seen is a set of item_global_ids
        # allowed = all items the user has NOT interacted with
        allowed = list(all_items_set - seen)
        if not allowed:
            # user has seen everything; just fall back
            # to all items so we can still sample something
            allowed = all_items_list

        allowed_items_per_user[u] = torch.tensor(
            allowed, dtype=torch.long
            # allowed, dtype=torch.int32 # trying to save disk space but needs to be casted back on loading
        )  # stays on CPU, which is fine

    return allowed_items_per_user
