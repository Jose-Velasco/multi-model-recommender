import torch
from typing import Any, Optional, Callable, cast, Literal, Iterable
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, AddRandomWalkPE, AddLaplacianEigenvectorPE
from torch_geometric.utils import to_networkx, degree
from .utils import add_node_attr, add_edge_attr, dict_to_tensor, drop_zero_columns
import networkx as nx
import numpy as np

class AddDegreeCentrality(BaseTransform):
    def __init__(self, normalize: Optional[Callable[..., Tensor]] = None, attr_name: Optional[str] = None):
        self.normalize = normalize
        # self.undirected = undirected
        self.attr_name = attr_name
    
    def forward(self, data: Data):
        assert data.edge_index is not None
        row = data.edge_index[0]
        # row is a list of source nodes thus is a node appears as a source node there exits an edge,
        # counts the number of time a node appears that is there degree 
        deg = degree(row, num_nodes=data.num_nodes).unsqueeze(1)
        if self.normalize:
           deg = self.normalize(deg)
        add_node_attr(data, features=deg, attr_name=self.attr_name)
        return data

class AddPageRank(BaseTransform):
    def __init__(self, normalize: Optional[Callable[..., Tensor]] = None, attr_name: Optional[str] = None, pagerank_alpha: float = 0.85):
        self.normalize = normalize
        self.attr_name = attr_name
        self.pagerank_alpha = pagerank_alpha
    
    def forward(self, data: Data):
        assert data.edge_index is not None
        assert data.num_nodes is not None
        
        graph_nx = to_networkx(
            data,
            node_attrs=["x"],
            edge_attrs=["edge_attr"] if getattr(data, "edge_attr", None) else None
        )
        pr_dict = nx.pagerank(graph_nx, alpha=self.pagerank_alpha)

        feature = dict_to_tensor(pr_dict, data.num_nodes).unsqueeze(1)
        # feature = torch.tensor(pr, dtype=torch.float32).unsqueeze(1)
        if self.normalize:
           feature = self.normalize(feature)
        add_node_attr(data, features=feature, attr_name=self.attr_name)
        return data

class AddClusteringCoefficient(BaseTransform):
    """
    In a strictly bipartite graph there are no triangles (odd cycles), so the standard NetworkX nx.clustering(G)
    (which measures triangle density around each node) is 0 for every node.

    Summing all node coefficients ⇒ 0.
    """
    def __init__(self, normalize: Optional[Callable[..., Tensor]] = None, attr_name: Optional[str] = None):
        self.normalize = normalize
        self.attr_name = attr_name
    
    def forward(self, data: Data):
        assert data.edge_index is not None
        assert data.num_nodes is not None

        graph_nx = to_networkx(
            data,
            node_attrs=["x"],
            edge_attrs=["edge_attr"] if getattr(data, "edge_attr", None) else None
        )
        clustering_dict = cast(dict[Any, float], nx.clustering(graph_nx))

        feature = dict_to_tensor(clustering_dict, data.num_nodes).unsqueeze(1)
        if self.normalize:
           feature = self.normalize(feature)
        add_node_attr(data, features=feature, attr_name=self.attr_name)
        return data

class AddBipartiteClusteringCoefficient(BaseTransform):
    """
    measures density of 4-cycles (squares), the bipartite analogue of triangles.
    
    networkx might not have gpu accelerated version might take very long (did not let it finish)
    """
    def __init__(self,
                 normalize: Optional[Callable[..., Tensor]] = None, 
                 attr_name: Optional[str] = None,
                 BCC_mode: Literal["dot", "max", "min"] = "dot"):
        self.normalize = normalize
        self.attr_name = attr_name
        self.BCC_mode = BCC_mode
    
    def forward(self, data: Data):
        assert data.edge_index is not None
        assert data.num_nodes is not None

        graph_nx = to_networkx(
            data,
            node_attrs=["x"],
            edge_attrs=["edge_attr"] if getattr(data, "edge_attr", None) else None
        )
        clustering_dict = nx.bipartite.clustering(graph_nx, mode=self.BCC_mode)

        feature = dict_to_tensor(clustering_dict, data.num_nodes).unsqueeze(1)
        if self.normalize:
           feature = self.normalize(feature)
        add_node_attr(data, features=feature, attr_name=self.attr_name)
        return data

class AddBirank(BaseTransform):
    """
    measures density of 4-cycles (squares), the bipartite analogue of triangles.
    """
    def __init__(self,
                 br_first_node_set: tuple[int, ...],
                 normalize: Optional[Callable[..., Tensor]] = None, 
                 attr_name: Optional[str] = None,
                 br_alpha: float = 0.8,
                 br_beta: float = 0.8,
                 br_top_personalization: Optional[dict] = None,
                 br_bottom_personalization: Optional[dict] = None,
                 br_max_iter: int = 100,
                 weight: Optional[str] = None):
        self.normalize = normalize
        self.attr_name = attr_name
        # "one side" of the bipartite graph eg. all user nodes
        self.br_first_node_set = br_first_node_set
        # interval [0,1]
        self.br_alpha = br_alpha
        # interval [0,1]
        self.br_beta = br_beta
        # Personalization values are used to encode a priori weights for a given node
        self.br_top_personalization = br_top_personalization
        # encode a priori weights for a given node
        self.br_bottom_personalization = br_bottom_personalization
        self.br_max_iter = br_max_iter
        # Edge data key to use as weight.
        self.weight = weight
    
    def forward(self, data: Data):
        assert data.edge_index is not None
        assert data.num_nodes is not None

        # Handle edge_attr existence + emptiness safely
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None and edge_attr.numel() > 0:
            edge_attrs = ["edge_attr"]
        else:
            edge_attrs = None

        graph_nx = to_networkx(
            data,
            node_attrs=["x"],
            edge_attrs=edge_attrs
        )
        bi_rank = nx.bipartite.birank(
            graph_nx,
            nodes=self.br_first_node_set,
            alpha=self.br_alpha,
            beta=self.br_beta,
            top_personalization=self.br_top_personalization,
            bottom_personalization=self.br_bottom_personalization,
            max_iter=self.br_max_iter,
            weight=self.weight
        )

        feature = dict_to_tensor(bi_rank, data.num_nodes).unsqueeze(1)
        if self.normalize:
           feature = self.normalize(feature)
        add_node_attr(data, features=feature, attr_name=self.attr_name)
        return data

class AddKatz(BaseTransform):
    def __init__(self, normalize: Optional[Callable[..., Tensor]] = None,
                 attr_name: Optional[str] = None, 
                 katz_alpha: float = 0.1, 
                 katz_beta: float = 1.0,
                 max_iter: int = 5000,
                 katz_normalize: bool = True,
                 katz_weight: Optional[str] = None):
        self.normalize = normalize
        self.attr_name = attr_name
        # must be < 1/lambda_max(A)
        self.katz_alpha = katz_alpha
        self.katz_beta = katz_beta
        self.katz_normalize = katz_normalize
        self.max_iter = max_iter
        # holds the name of the edge attribute used as weight
        # interpreted as the connection strength.
        self.katz_weight = katz_weight
    
    def forward(self, data: Data):
        assert data.edge_index is not None

        graph_nx = to_networkx(
            data,
            node_attrs=["x"],
            edge_attrs=["edge_attr"] if getattr(data, "edge_attr", None) else None
        )

        katz = nx.katz_centrality(
            graph_nx,
            alpha=self.katz_alpha,
            beta=self.katz_beta,
            max_iter=self.max_iter,
            normalized=self.katz_normalize,
            weight=self.katz_weight
        )
        feature = torch.tensor(katz.values(), dtype=torch.float32).unsqueeze(1)
        if self.normalize:
           feature = self.normalize(feature)
        add_node_attr(data, features=feature, attr_name=self.attr_name)
        return data

class AddClosenessCentralities(BaseTransform):
    def __init__(self, normalize: Optional[Callable[..., Tensor]] = None, attr_name: Optional[str] = None):
        self.normalize = normalize
        self.attr_name = attr_name
    
    def forward(self, data: Data):
        assert data.edge_index is not None
        assert data.num_nodes is not None

        graph_nx = to_networkx(
            data,
            node_attrs=["x"],
            edge_attrs=["edge_attr"] if getattr(data, "edge_attr", None) else None
        )
        closeness_dict = nx.closeness_centrality(graph_nx)
        feature = dict_to_tensor(closeness_dict, data.num_nodes).unsqueeze(1)
        # feature = torch.tensor(closeness.values(), dtype=torch.float32).unsqueeze(1)
        if self.normalize:
           feature = self.normalize(feature)
        add_node_attr(data, features=feature, attr_name=self.attr_name)
        return data

class AddBipartiteClosenessCentralities(BaseTransform):
    def __init__(self, bp_first_node_set: Iterable[int],
                 normalize: Optional[Callable[..., Tensor]] = None,
                 attr_name: Optional[str] = None,
                 closeness_normalized: bool = True):
        # "one side" of the bipartite graph eg. all user nodes or all items
        self.bp_first_node_set = bp_first_node_set
        self.closeness_normalized = closeness_normalized
        self.normalize = normalize
        self.attr_name = attr_name
    
    def forward(self, data: Data):
        assert data.edge_index is not None
        assert data.num_nodes is not None

        graph_nx = to_networkx(data, node_attrs=["x"], edge_attrs=getattr(data, "edge_attr", None), to_undirected=True)

        bp_closeness_dict = nx.bipartite.closeness_centrality(graph_nx, nodes=self.bp_first_node_set, normalized=self.closeness_normalized)
        feature = dict_to_tensor(bp_closeness_dict, data.num_nodes).unsqueeze(1)
        if self.normalize:
           feature = self.normalize(feature)
        add_node_attr(data, features=feature, attr_name=self.attr_name)
        return data

class AddBipartiteBetweennessCentralities(BaseTransform):
    def __init__(self, bp_first_node_set: Iterable[int],
                 normalize: Optional[Callable[..., Tensor]] = None,
                 attr_name: Optional[str] = None):
        # "one side" of the bipartite graph eg. all user nodes or all items
        self.bp_first_node_set = bp_first_node_set
        self.normalize = normalize
        self.attr_name = attr_name
    
    def forward(self, data: Data):
        assert data.edge_index is not None
        assert data.num_nodes is not None

        graph_nx = to_networkx(data, node_attrs=["x"], edge_attrs=getattr(data, "edge_attr", None), to_undirected=True)

        bp_betweenness_dict = nx.bipartite.betweenness_centrality(graph_nx, nodes=self.bp_first_node_set)
        feature = dict_to_tensor(bp_betweenness_dict, data.num_nodes).unsqueeze(1)
        if self.normalize:
           feature = self.normalize(feature)
        add_node_attr(data, features=feature, attr_name=self.attr_name)
        return data

class AddBetweennessCentralities(BaseTransform):
    def __init__(self, normalize: Optional[Callable[..., Tensor]] = None, attr_name: Optional[str] = None,
                 betweenness_normalize: bool = True,
                 betweenness_k: int=10,
                 rnd_seed: int = 101):
        self.normalize = normalize
        self.attr_name = attr_name
        self.betweenness_normalize = betweenness_normalize
        self.betweenness_k = betweenness_k
        self.rnd_seed = rnd_seed
    
    def forward(self, data: Data):
        assert data.edge_index is not None
        assert data.num_nodes is not None

        # Handle edge_attr existence + emptiness safely
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None and edge_attr.numel() > 0:
            edge_attrs = ["edge_attr"]
        else:
            edge_attrs = None
        # graph_nx = to_networkx(data, node_attrs=["x"], edge_attrs=["edge_attr"])
        graph_nx = to_networkx(
            data,
            node_attrs=["x"],
            edge_attrs=edge_attrs
        )
        betweenness_dict = nx.betweenness_centrality(graph_nx, k=self.betweenness_k, normalized=self.betweenness_normalize, seed=self.rnd_seed)
        feature = dict_to_tensor(betweenness_dict, data.num_nodes).unsqueeze(1)
        # feature = torch.tensor(betweenness.values(), dtype=torch.float32).unsqueeze(1)
        if self.normalize:
           feature = self.normalize(feature)
        add_node_attr(data, features=feature, attr_name=self.attr_name)
        return data

class AddConstraintStructuralHoles(BaseTransform):
    def __init__(self, normalize: Optional[Callable[..., Tensor]] = None, attr_name: Optional[str] = None):
        self.normalize = normalize
        self.attr_name = attr_name
    
    def forward(self, data: Data):
        assert data.edge_index is not None
        assert data.num_nodes is not None

        graph_nx = to_networkx(data, node_attrs=["x"], edge_attrs=getattr(data, "edge_attr", None))
        constraint_dict = nx.constraint(graph_nx)
        feature = dict_to_tensor(constraint_dict, data.num_nodes).unsqueeze(1)
        constraint_arr = np.array(feature, dtype=np.float32)
        # replace nodes with nan constraint values in the graph with the max constraint + mean constraint
        # this way nodes with that have undefined constraint AKA. nan values will have the highest constraint
        # in the graph but since these nodes with nan constraint are more constrained that the node(s)
        # with this highest constraint we add more their constant relative to the graph using the mean constraint
        max_constraint = np.nanmax(constraint_arr)
        mean_constraint = np.nanmean(constraint_arr)
        constraint_arr[np.isnan(constraint_arr)] = max_constraint + mean_constraint

        feature = torch.tensor(constraint_arr, dtype=torch.float32)
        if self.normalize:
           feature = self.normalize(feature)
        add_node_attr(data, features=feature, attr_name=self.attr_name)
        return data

class AddAdamicAdar(BaseTransform):
    """
    weighted overlap favoring rare neighbors: richer but unbounded.
    does not work really on bipartite graphs since there are not common neighbors between user and items
    this is because user only connect to items and items only connect to users. a neighbor of a user is an item
    but and item cannot be connected to an item thus not common neightbors.
    """
    def __init__(self, normalize: Optional[Callable[..., Tensor]] = None, attr_name: Optional[str] = None):
        self.normalize = normalize
        self.attr_name = attr_name
    
    def forward(self, data: Data):
        assert data.edge_index is not None
        source, target = data.edge_index
        source = source.tolist()
        target = target.tolist()

        graph_nx = to_networkx(data, node_attrs=["x"], edge_attrs=getattr(data, "edge_attr", None), to_undirected=True)

        adamic_adar_iterator = nx.adamic_adar_index(graph_nx, ebunch=list(zip(source, target)))
        
        # convert to tensors aligned with edge_index features
        aa_scores = {(u,v): p for u, v, p in adamic_adar_iterator}

        feature = torch.tensor([aa_scores[(u,v)] for u, v in zip(source, target)], dtype=torch.float32).unsqueeze(1)
        # Optional normalization (must preserve [E, k] shape)
        if self.normalize:
           feature = self.normalize(feature)
        add_edge_attr(data, features=feature, attr_name=self.attr_name)
        return data

class AddBipartiteOverlapWeighted(BaseTransform):
    """
    apply_jaccard (bool): if True uses jaccard similarly between node neighbors else "fraction of common neighbors by minimum of both nodes degree"
    """
    def __init__(self, bp_first_node_set: Iterable[int], normalize: Optional[Callable[..., Tensor]] = None, attr_name: Optional[str] = None,
                 apply_jaccard: bool = True):
        # "one side" of the bipartite graph eg. all user nodes or all items
        self.bp_first_node_set = bp_first_node_set
        self.apply_jaccard = apply_jaccard
        self.normalize = normalize
        self.attr_name = attr_name
    
    def forward(self, data: Data):
        assert data.edge_index is not None
        assert data.num_nodes is not None

        graph_nx = to_networkx(data, node_attrs=["x"], edge_attrs=getattr(data, "edge_attr", None), to_undirected=True)

        bi_overlap_weighted: nx.Graph = nx.bipartite.overlap_weighted_projected_graph(graph_nx, nodes=self.bp_first_node_set, jaccard=self.apply_jaccard)

        # convert to tensors aligned with edge_index features
        edge_weights = [bi_overlap_weighted[int(u)][int(v)]["weight"] for u, v in zip(data.edge_index[0], data.edge_index[1])]

        feature = torch.as_tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
        # Optional normalization (must preserve [E, k] shape)
        if self.normalize:
           feature = self.normalize(feature)
        add_edge_attr(data, features=feature, attr_name=self.attr_name)
        return data

class AddRandomWalkRelativeDistance(BaseTransform):
    """
    Pair-wise distance from: Random Walks

    Build edge features from node RWPE:
      edge_rw(e = u->v) = reduce(rw[u] - rw[v])
    Default: absolute difference (size [E, k]).
    """
    def __init__(self, normalize: Optional[Callable[..., Tensor]] = None, attr_name: Optional[str] = None,
                 pre_computed_rw_attr_name: Optional[str] = "random_walk_pe",
                 walk_length: int = 3,
                 drop_all_zero_columns: bool = True):
        self.normalize = normalize
        self.attr_name = attr_name
        self.pre_computed_rw_attr_name = pre_computed_rw_attr_name
        # used if RW is not precomputed
        self.walk_length = walk_length
        self.drop_all_zero_columns = drop_all_zero_columns
    
    def forward(self, data: Data):
        assert data.edge_index is not None
        source, target = data.edge_index # [E], [E]
        # source/target: [E]
        # edge_rw: [E,k]

        # Ensure node-level RWPE is present
        if self.pre_computed_rw_attr_name and hasattr(data, self.pre_computed_rw_attr_name):
            # Construct edge-level differences: [E, k] k: features
            edge_rw = torch.abs(data[self.pre_computed_rw_attr_name][source] - data[self.pre_computed_rw_attr_name][target])
        else:
            print(f"Precomputed RW was not found as attribute to graph names {self.pre_computed_rw_attr_name}.")
            print(f"Computing RW with walk_length of {self.walk_length}.")
            rw_transformer = AddRandomWalkPE(walk_length=self.walk_length) # adds data.random_walk_pe [N, k]
            data = rw_transformer(data)

            # Construct edge-level differences: [E, k] k: features
            edge_rw = torch.abs(data.random_walk_pe[source] - data.random_walk_pe[target])

            # cleaning up
            del data.random_walk_pe
        
        if self.drop_all_zero_columns:
            edge_rw = drop_zero_columns(edge_rw)

        # Optional normalization (must preserve [E, k] shape)
        if self.normalize:
           edge_rw = self.normalize(edge_rw)
        
        # Attach to graph
        add_edge_attr(data, features=edge_rw, attr_name=self.attr_name)
        return data

class AddLaplacianEigenvectorRelativeDistance(BaseTransform):
    """
    Pair-wise distance from: Laplacian Eigenvector

    lp_k (int): The number of non-trivial eigenvectors to consider.
    
    """
    def __init__(self, normalize: Optional[Callable[..., Tensor]] = None, attr_name: Optional[str] = None,
                 pre_computed_lp_attr_name: Optional[str] = "laplacian_eigenvector_pe",
                 lpe_k: int = 16,
                 lpe_is_undirected: bool = False):
        self.normalize = normalize
        self.attr_name = attr_name
        self.pre_computed_lp_attr_name = pre_computed_lp_attr_name
        self.lpe_k = lpe_k
        self.lpe_is_undirected = lpe_is_undirected
    
    def forward(self, data: Data):
        assert data.edge_index is not None
        source, target = data.edge_index
        if self.pre_computed_lp_attr_name and hasattr(data, self.pre_computed_lp_attr_name):
            # [E, k] k: features
            edge_lp = torch.abs(data[self.pre_computed_lp_attr_name][source] - data[self.pre_computed_lp_attr_name][target])
        else:
            print(f"Precomputed LapEV was not found as attribute to graph names {self.pre_computed_lp_attr_name}.")
            print(f"Computing LapEV with k of {self.lpe_k}.")
            lp_transformer = AddLaplacianEigenvectorPE(k=self.lpe_k, is_undirected=self.lpe_is_undirected)
            data = lp_transformer(data)
            # [E, k] k: features
            edge_lp = torch.abs(data.laplacian_eigenvector_pe[source] - data.laplacian_eigenvector_pe[target])
            # cleaning up
            del data.laplacian_eigenvector_pe

        # Optional normalization (must preserve [E, k] shape)
        if self.normalize:
           edge_lp = self.normalize(edge_lp)
        add_edge_attr(data, features=edge_lp, attr_name=self.attr_name)
        return data
