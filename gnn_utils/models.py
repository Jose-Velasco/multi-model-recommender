import torch
from typing import Any, Dict, Optional
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn.attention import PerformerAttention
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ModuleDict,
    ReLU,
    GELU,
    Sequential,
    Dropout
)

class EdgeMLPClassifier(torch.nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, dropout=0.3):
        super().__init__()
        # 2 * because you concatenate user and music node embeddings
        self.net = Sequential(
            Linear(2 * input_channels, hidden_channels),
            # ReLU(),
            GELU(),
            Dropout(dropout),
            Linear(hidden_channels, 1)
        )
        self.dropout = dropout

    def forward(self, x_user: torch.Tensor, x_items: torch.Tensor):
        z = torch.cat([x_user, x_items], dim=-1)
        # shape [batch] a score for each link: real number (higher = more likely interaction)
        return self.net(z).squeeze(-1)

class GPS(torch.nn.Module):
    # def __init__(self, channels: int, pe_keys: list[str], pe_dims: list[int], num_layers: int,
    def __init__(self, channels: int, pe_parameters: list[tuple[str, int]], # pe_out_dim: int
                 num_layers: int,
                 attn_type: str, attn_kwargs: Dict[str, Any], attentions_heads: int,
                 mpnn: list[MessagePassing], classifier: torch.nn.Module,
                 num_of_nodes: int, dropout_prob: float, act_func: str):
        super().__init__()

        # tuple first element is pe key and second element is the associated pe's dim
        self.dropout_prob = dropout_prob
        self.act_func = act_func
        self.pe_parameters = pe_parameters
        # output dim of pe transformation
        # self.pe_out_dim = pe_out_dim
        
        # self.pe_transformations: dict[str, Sequential] = {}
        self.pe_transformations = ModuleDict()
        total_pe_dims_size: int = 0
        for pe_key, pe_dim in self.pe_parameters:
            self.pe_transformations[pe_key] = Sequential(
                BatchNorm1d(pe_dim),
                Linear(pe_dim, pe_dim)
            )
            total_pe_dims_size += pe_dim

        assert (channels - total_pe_dims_size) > 1, "Need at least 1 learned node embeddings after removing pe features dim"

        self.node_emb = Embedding(num_of_nodes, channels - total_pe_dims_size)

        self.convs = ModuleList()
        for layer_index in range(num_layers):
            layer_mpnn = mpnn[layer_index]
            conv = GPSConv(channels, layer_mpnn, heads=attentions_heads,
                           dropout=self.dropout_prob, act=act_func,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.classifier = classifier
        self.redraw_projection: RedrawProjection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    # def forward(self, x, pe: torch.Tensor, edge_index, edge_attr, batch):
    def forward(self, x, edge_index, edge_attr, batch: Data):
        all_pe =  self._process_all_positional_encodings(batch)
        # only gets the node embeddings for nodes in the current batch and concat their pe embeddings 
        x = torch.cat((self.node_emb(batch.n_id), all_pe), 1)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.dropout(x, self.dropout_prob, training=self.training)

        return x
    
    def _process_all_positional_encodings(self, batch: Data) -> torch.Tensor:
        pe_to_combine: list[torch.Tensor] = []
        for pe in self.pe_transformations.keys():
            if pe in batch:
                # we only want PEs for nodes in the batch/subgraph and not all PEs for all nodes in thw whole graph
                # nodes_in_subgraph = batch.n_id
                # pe_of_nodes_in_subgraph = batch[pe][nodes_in_subgraph]
                pe_of_nodes_in_subgraph = batch[pe]
                pe_transformed = self.pe_transformations[pe](pe_of_nodes_in_subgraph)
                pe_to_combine.append(pe_transformed)
            else:
                raise KeyError(f"found pe that has not projection: {pe}")
        return torch.cat(pe_to_combine, dim=-1)
        # x_pe = torch.cat([batch.pe for pe in self.pe_transformations.keys() if pe in batch], dim=-1)
    
    def get_embeddings(self, node_id: Optional[torch.Tensor]) -> torch.Tensor:
        """
        node_id (torch.Tensor): The indices of the nodes of the embeddings
                that will be returned.
                If set to :obj:`None`, all nodes will be used.
        """
        emb = self.node_emb.weight
        return emb if node_id is None else emb[node_id]
         

# code from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py
class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1
