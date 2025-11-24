# %%
from functools import partial
from typing import cast
import numpy as np
from gnn_utils.datasets import PositiveIterationGraph
from gnn_utils.utils import drop_zero_columns, get_save_allowed_items_per_user, graph_info, log1p_standardize, normalize_zscore
from gnn_utils.ray_utils import (
    get_single_dataset_loader, generate_dataset_loaders, GPSConfig, GINEConfig,
    EdgeMLPClassifierConfig, BaseConfig, LearningRateSchedulerConfigs, gps_model_factory
    )
from gnn_utils.models import GPS
import torch
from torch_geometric.transforms import Compose, AddRandomWalkPE, AddLaplacianEigenvectorPE, RandomLinkSplit
from gnn_utils.transforms import (
    AddDegreeCentrality, AddPageRank, AddBirank,
    AddBetweennessCentralities, AddConstraintStructuralHoles,
    AddLaplacianEigenvectorRelativeDistance,
    AddRandomWalkRelativeDistance
)
from torch_geometric.data import Data
from ray import tune
import random

# %%
ROOT_DATA_DIR = "./data"
GRAPH_DATA_DIR = f"{ROOT_DATA_DIR}/graph_data"
SAVE_ALLOWED_ITEMS_PER_USER = f"{GRAPH_DATA_DIR}/allowed_items_per_user.pt"
# SAVE_MODEL_PATH = f"{GRAPH_DATA_DIR}/best_model_checkpoint.pt"
RNG_SEED = 101
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MLFLOW_URI = "http://localhost:8080"
# %%
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
graph: Data = dataset[0] # pyright: ignore[reportAssignmentType]
graph_info(graph)

# these are guaranteed to be applied during graph first built but manually checked
# to let type checking know they are there
assert graph.x is not None and graph.edge_index is not None and graph.edge_attr is not None
# %%
# preprocessing random walk PE to remove first columns since it currently always 0
# think it because unless a node has a self loop at one hop away the probability of returning back
# is 0, and in this bipartite graph there are no self loops
graph.random_walk_pe = drop_zero_columns(graph.random_walk_pe)
graph.x = graph.x[:, 1:]

# %%
item_node_id_map: dict[int, int] = graph.item_node_id_map # pyright: ignore[reportAttributeAccessIssue]
del graph.item_node_id_map # pyright: ignore[reportAttributeAccessIssue]
print(f"Number of item nodes: {len(item_node_id_map)}")

users_node_id_map: dict[int, int] = graph.users_node_id_map # pyright: ignore[reportAttributeAccessIssue]
del graph.users_node_id_map # pyright: ignore[reportAttributeAccessIssue]

# %%
allowed_items_per_user = get_save_allowed_items_per_user(
    save_allowed_items_per_user_file_path=SAVE_ALLOWED_ITEMS_PER_USER,
    graph=graph, # pyright: ignore[reportArgumentType]
    user_to_node_id_map=users_node_id_map,
    item_node_id_map=item_node_id_map)
# %%
dataset_split_transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=False,
    disjoint_train_ratio=0.4
)
train_data, val_data, test_data = cast(tuple[Data, Data, Data], dataset_split_transform(graph))
# %%
test_loader = get_single_dataset_loader(
    graph=test_data,
    is_evaluating=False,
    allowed_items_per_user=None,
    item_node_ids=[item_node_id for item_node_id in item_node_id_map.values()],
    batch_size=64,
    neg_sampling_ratio=3.0,
    evaluation_num_negative=13,
    num_neighbors=[2,2,2],
    num_workers=2,
    pin_memory=True
)
# %%
train_loader, val_loader, test_loader = generate_dataset_loaders(
    graph=graph,
    allowed_items_per_user=allowed_items_per_user,
    item_node_ids=[item_node_id for item_node_id in item_node_id_map.values()],
    val_data_size = 0.1,
    test_data_size = 0.1,
    train_batch_size = 128,
    evaluation_batch_size= 512,
    is_undirected = True,
    disjoint_train_ratio = 0.4,
    neg_sampling_ratio = 3.0,
    evaluation_num_negative=50,
    num_neighbors = [5, 5, 5],
    num_workers = 3,
    pin_memory = True
)
# %%

# lr_scheduler_config = LearningRateSchedulerConfigs(
#     mode="max",
#     factor=0.5,
#     patience=3,
#     min_lr=0.00001
# )

lr_scheduler_config_ray = {
    "mode":"max",
    "factor":0.5,
    "patience":3,
    "min_lr":0.00001
}

# base_config = BaseConfig(
#     max_epoch=5,
#     device="cuda",
#     num_workers=2,
#     train_batch_size=64,
#     evaluation_batch_size=128,
#     optimizer_learning_rate=0.001,
#     weight_decay=0.01,
#     learning_rate_scheduler_configs=lr_scheduler_config,
#     val_data_size=0.1,
#     test_data_size=0.1,
#     is_undirected=True,
#     disjoint_train_ratio=0.4,
#     neg_sampling_ratio=3,
#     evaluation_num_negative=50,
#     num_neighbors=[3,3],
#     pin_memory=True,
#     non_blocking=True,
#     BPR_loss_lambda=0.1,
#     metrics_top_k=20,
#     metrics_aggregation="mean"
# )

base_config_ray = {
    "max_epoch": 20,
    "device": "cuda",
    "num_workers": 2,
    "train_batch_size": tune.choice([128, 256, 512, 1024]),
    "evaluation_batch_size": tune.choice([512, 1024, 1048]),
    "optimizer_learning_rate": tune.choice([0.01, 0.001, 0.0001]),
    "weight_decay": tune.choice([0.01, 0.001, 0.0001, 0.00001]),
    "learning_rate_scheduler_configs": lr_scheduler_config_ray,
    "val_data_size": 0.1,
    "test_data_size": 0.1,
    "is_undirected": True,
    "disjoint_train_ratio": tune.choice([0.3, 0.4, 0.5, 0.6]),
    "neg_sampling_ratio": tune.choice([2, 3, 4, 5, 6]),
    "evaluation_num_negative": tune.choice([25, 50, 75, 100]),
    "num_neighbors":tune.choice([tune.sample_from(lambda config: [tune.qrandint(lower=5, upper=15, q=1).sample() for num_nodes_to_sample in range(config["num_layers"])])]),
    "pin_memory": True,
    "non_blocking": True,
    "BPR_loss_lambda": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
    "metrics_top_k": 20,
    "metrics_aggregation": "mean"
}

# classifier_config = EdgeMLPClassifierConfig(
#     input_channels=128,
#     hidden_channels=128,
#     dropout=0.4
# )

classifier_config_ray = {
    "input_channels": tune.sample_from(lambda config: config["channels"]),
    "hidden_channels": tune.choice([512, 1024, 2048, 4096]),
    "dropout": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
}

# mpnn_config = GINEConfig(
#     num_layers=2,
#     mlp_channels=[128,128,128],
#     dropout=0.4,
#     edge_dim=graph.num_edge_features
# )

mpnn_config_ray = {
    "num_layers": tune.choice([2, 3, 4]),
    "mlp_channels": tune.sample_from(
        lambda spec: [
            random.choice([64, 128, 256, 512]) for _ in range(random.randint(1, 4))
        ]
    ),
    "dropout": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
    "edge_dim": graph.num_edge_features,
}

# gps_model_config = GPSConfig(
#    channels=128,
#    num_layers=2,
#    attn_type="performer",
#    attn_kwargs={'dropout': 0.4},
#    attentions_heads=4,
#    mpnn=mpnn_config,
#    classifier=classifier_config,
#    num_of_nodes=graph.num_nodes, # pyright: ignore[reportArgumentType]
#    num_initial_node_features=graph.num_node_features,
#    dropout_prob=0.4,
#    act_func="gelu",
#    training_configs=base_config
# )

gps_model_config_ray = {
    "channels": tune.choice([128, 256, 512, 1024]),
    "num_layers": tune.choice([2, 3, 4]),
    "attn_type": "performer",
    "attn_kwargs": tune.sample_from(lambda spec: {'dropout': random.choice([0.1, 0.2, 0.3, 0.4, 0.5])}),
    "attentions_heads": tune.choice([2, 4, 8]),
    "mpnn": mpnn_config_ray,
    "classifier": classifier_config_ray,
    "num_of_nodes": graph.num_nodes,
    "num_initial_node_features": graph.num_node_features,
    "dropout_prob": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
    "act_func": tune.choice(["gelu", "relu"]),
    "training_configs": base_config_ray,
}

# %%
# model = gps_model_factory(
#     config=gps_model_config,
#     graph=graph,
#     model_class=GPS
# )
# TODO: test out ray tune on and then send to colab with stronger GPU