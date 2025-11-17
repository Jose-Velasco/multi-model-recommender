# %%
from gnn_utils.datasets import PositiveIterationGraph
from torch_geometric.data import Data
from gnn_utils.train_utils import train
from gnn_utils.utils import (
    metrics_tracker_factory, normalize_zscore, log1p_standardize, graph_info,
    EarlyStopper, get_save_allowed_items_per_user
)

from torch_geometric.transforms import Compose, AddRandomWalkPE, AddLaplacianEigenvectorPE, RandomLinkSplit
from torch_geometric.sampler import NegativeSampling
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GINEConv, summary
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.lightgcn import BPRLoss
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial
from typing import cast
from gnn_utils.transforms import (
    AddDegreeCentrality, AddPageRank, AddBirank,
    AddBetweennessCentralities, AddConstraintStructuralHoles,
    AddLaplacianEigenvectorRelativeDistance,
    AddRandomWalkRelativeDistance, BuildRankingCandidatesTransform,
    KeepUserToItemSupervisionEdges
)
from gnn_utils.models import GPS, EdgeMLPClassifier
from collections import defaultdict
from functools import partial

# from torch_geometric.metrics import (
#     LinkPredMAP,
#     LinkPredMetricCollection,
#     LinkPredPrecision,
#     LinkPredRecall,
# )
# %%
ROOT_DATA_DIR = "./data"
GRAPH_DATA_DIR = f"{ROOT_DATA_DIR}/graph_data"
SAVE_ALLOWED_ITEMS_PER_USER = f"{GRAPH_DATA_DIR}/allowed_items_per_user.pt"
RNG_SEED = 101
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 30% of edges for supervision and 70% are edges for message passing if set to 0.3
NUM_VAL = 0.1
NUM_TEST = 0.1
DISJOINT_TRAIN_RATIO = 0.4
NEGATIVE_SAMPLE_RATIO = 3
EVALUATION_NUM_NEGATIVE = 50
NUM_NEIGHBORS = [5, 5, 5]
NUM_WORKERS = 8
# (PIN_MEMORY, NON_BLOCKING) should increase training performance with GPU as long as dataset fits in memory(ram memory)
PIN_MEMORY = True
NON_BLOCKING = True

NUM_EPOCHS: int = 2
# BATCH_SIZE: int = 1024
TRAIN_BATCH_SIZE: int = 256
EVALUATION_BATCH_SIZE: int = 512
OPTIMIZER_LEARNING_RATE: float = 0.001
WEIGHT_DECAY: float = 0.01

GNN_HIDDEN_CHANNELS: int = 128
NUM_LAYERS: int = 2
CLASSIFIER_HIDDEN_CHANNELS: int = 1024
DROPOUT: float = 0.4
# PE_OUT_DIM = 16
ATTN_KWARGS = {'dropout': DROPOUT}
ATTENTIONS_HEADS = 4
GINE_MLP_CHANNEL_LIST = [GNN_HIDDEN_CHANNELS, 3 * GNN_HIDDEN_CHANNELS, GNN_HIDDEN_CHANNELS]

TOP_K = 20
METRIC_AGGREGATION = "mean"

torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
# %%
dataset_transformations = Compose(
    [
        AddBetweennessCentralities(normalize=log1p_standardize, rnd_seed=RNG_SEED, betweenness_k=700),
        AddDegreeCentrality(normalize=log1p_standardize),
        AddPageRank(normalize=log1p_standardize),
        AddConstraintStructuralHoles(normalize=log1p_standardize),

        AddLaplacianEigenvectorPE(k=32, is_undirected=True),
        # need to manually removed zero coumns
        AddRandomWalkPE(walk_length=2),
        AddLaplacianEigenvectorRelativeDistance(normalize=normalize_zscore, lpe_is_undirected=True, lpe_k=32),
        AddRandomWalkRelativeDistance(normalize=normalize_zscore, walk_length=2),
    ]
)

# These need the set of nodes from "one side" of the bipartite graph
bipartite_pre_transform = [
        partial(AddBirank, normalize=normalize_zscore)
    ]

# Remember to drop RW all zero columns
# the first node feature column is just a place holder drop it
dataset = PositiveIterationGraph(
    root=GRAPH_DATA_DIR,
    add_user_item_node_mapping=True,
    is_undirected=True,
    pre_transform=dataset_transformations,
    bipartite_pre_transform=bipartite_pre_transform,
)
# %%
graph = dataset[0]
graph_info(graph)
item_node_id_map: dict[int, int] = graph.item_node_id_map # pyright: ignore[reportAttributeAccessIssue]
del graph.item_node_id_map # pyright: ignore[reportAttributeAccessIssue]

users_node_id_map: dict[int, int] = graph.users_node_id_map # pyright: ignore[reportAttributeAccessIssue]
del graph.users_node_id_map # pyright: ignore[reportAttributeAccessIssue]
# %%
dataset_split_transform = RandomLinkSplit(
    num_val=NUM_VAL,
    num_test=NUM_TEST,
    is_undirected=True,
    add_negative_train_samples=False,
    disjoint_train_ratio=DISJOINT_TRAIN_RATIO
)
train_data, val_data, test_data = cast(tuple[Data, Data, Data], dataset_split_transform(graph))

user_seen_items: defaultdict[int, set[int]] = defaultdict(set)
all_items: set[int] = set()

allowed_items_per_user = get_save_allowed_items_per_user(
    save_allowed_items_per_user_file_path=SAVE_ALLOWED_ITEMS_PER_USER,
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    item_node_id_map=item_node_id_map)

train_supervision_edge_keep_transform = KeepUserToItemSupervisionEdges(
    attach_node_type_vector=False,
    num_nodes=graph.num_nodes, # type: ignore
    item_node_ids=[item_node_id for item_node_id in item_node_id_map.values()]
)
eval_supervision_edge_keep_transform = KeepUserToItemSupervisionEdges(
    attach_node_type_vector=True,
    num_nodes=graph.num_nodes, # type: ignore
    item_node_ids=train_supervision_edge_keep_transform.item_node_ids
)

train_data = train_supervision_edge_keep_transform(train_data)
val_data = eval_supervision_edge_keep_transform(val_data)
test_data = eval_supervision_edge_keep_transform(test_data)

negative_sampler = NegativeSampling(
    mode="triplet",
    amount=NEGATIVE_SAMPLE_RATIO
)

val_transform = BuildRankingCandidatesTransform(
    allowed_items_per_user=allowed_items_per_user,  # dict[int, Tensor of global item ids]
    num_neg_eval=EVALUATION_NUM_NEGATIVE,
    item_type=eval_supervision_edge_keep_transform.item_type,                 # whatever code you use for item nodes
    allow_batch_fallback=True,
)
test_transform = BuildRankingCandidatesTransform(
    allowed_items_per_user=allowed_items_per_user,  # dict[int, Tensor of global item ids]
    num_neg_eval=EVALUATION_NUM_NEGATIVE,
    item_type=eval_supervision_edge_keep_transform.item_type,                 # whatever code you use for item nodes
    allow_batch_fallback=True,
)

# train_data.edge_index = train message passing graph
# train_data.edge_label_index = train supervision edges

# batch.src_index – source nodes (users in this case, if edges are user→item)  (local ids)
# batch.dst_pos_index - the positive items (local ids)
# batch.dst_neg_index - the negative items sampled for those users (local ids)
"""
from docs
edge_label_index (Tensor or EdgeType or Tuple[EdgeType, Tensor]) - The edge indices for which neighbors are sampled to create mini-batches.

so basically these are the supervision edges or samples we are computing loss to perform gradient decent
"""

train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=NUM_NEIGHBORS,
    edge_label_index=train_data.edge_label_index, # user→item supervision edges
    batch_size=TRAIN_BATCH_SIZE,
    neg_sampling=negative_sampler,
    edge_label=None,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=NUM_NEIGHBORS,
    edge_label_index=val_data.edge_label_index, # user→item supervision edges
    batch_size=EVALUATION_BATCH_SIZE,
    neg_sampling=None, # no sampler-based negatives in eval
    transform=val_transform, # attaches candidate negative items and targets
    edge_label=None,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=NUM_NEIGHBORS,
    edge_label_index=test_data.edge_label_index, # user→item supervision edges
    batch_size=EVALUATION_BATCH_SIZE,
    edge_label=None,
    neg_sampling=None, # no sampler-based negatives in eval
    transform=test_transform, # attaches candidate negative items and targets
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

# eval_negative_sampler = partial(sample_negatives, num_neg=EVALUATION_NUM_NEGATIVE, all_items=all_items, user_seen_items=user_seen_items)

# Metrics to track
# RetrievalRecall(k=20)
# RetrievalPrecision(k=20)
# RetrievalMAP(k=20)
# RetrievalNormalizedDCG(k=20) ← use this to pick best model
# %%
classifier = EdgeMLPClassifier(
    input_channels=GNN_HIDDEN_CHANNELS,
    hidden_channels=CLASSIFIER_HIDDEN_CHANNELS,
    dropout=DROPOUT
)
pe_parameters: list[tuple[str, int]] = [
    ("random_walk_pe", graph.random_walk_pe.shape[1]), # pyright: ignore[reportAttributeAccessIssue]
    ("laplacian_eigenvector_pe", graph.laplacian_eigenvector_pe.shape[1]) # pyright: ignore[reportAttributeAccessIssue]
]
mess_pass_nn: list[MessagePassing] = []
for _ in range(NUM_LAYERS):
    nn = MLP(
        channel_list=GINE_MLP_CHANNEL_LIST,
        dropout=DROPOUT,
    )
    mess_pass_nn.append(GINEConv(nn=nn, edge_dim=graph.num_edge_features))

model = GPS(
    channels=GNN_HIDDEN_CHANNELS,
    pe_parameters=pe_parameters,
    num_layers=NUM_LAYERS,
    attn_type="performer",
    attn_kwargs=ATTN_KWARGS,
    attentions_heads=ATTENTIONS_HEADS,
    mpnn=mess_pass_nn,
    classifier=classifier,
    num_of_nodes=graph.num_nodes, # type: ignore
    dropout_prob=DROPOUT,
    act_func="gelu"
).to(DEVICE)

test_batch = next(iter(train_loader)).to(DEVICE)
summary(
    model=model,
    x=test_batch.x,
    edge_index=test_batch.edge_index,
    edge_attr=test_batch.edge_attr,
    batch=test_batch)
# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=OPTIMIZER_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3,
    min_lr=0.00001
)
criterion = BPRLoss()
early_stopper = EarlyStopper(patience=3, mode="max")
metrics = metrics_tracker_factory(top_k=TOP_K, aggregation=METRIC_AGGREGATION).to(DEVICE)
# %%
train(
    model=model,
    train_dataloader=train_loader,
    test_dataloader=val_loader,
    optimizer=optimizer,
    loss_fn=criterion,
    learning_rate_scheduler=scheduler,
    early_stopper=early_stopper,
    epochs=NUM_EPOCHS,
    device=DEVICE,
    non_blocking=NON_BLOCKING,
    metricsTrackers=metrics
)
# %%
test_batch = next(iter(train_loader)).to(DEVICE)
summary_str = summary(
    model=model,
    x=test_batch.x,
    edge_index=test_batch.edge_index,
    edge_attr=test_batch.edge_attr,
    batch=test_batch)
print(summary_str)
# %%
# TODO: get test func working with eval_negative_sampler
#       test training func then test and overall train
#       use NDCG@20 for LR scheduler and earlystopping once working move to
#       ray tune
