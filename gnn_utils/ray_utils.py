import copy
from dataclasses import dataclass
import os
import random
from typing import Any, Literal, Optional, Type, cast
import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from gnn_utils.transforms import BuildRankingCandidatesTransform, KeepUserToItemSupervisionEdges
from gnn_utils.utils import generate_now_timestamp_str, metrics_tracker_factory, normalize_metrics_dict
from gnn_utils.train_utils import test_step, train_step
from gnn_utils.models import GPS, EdgeMLPClassifier
from torch_geometric.nn import GINEConv
from torch_geometric.nn.models import MLP
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
import tempfile
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.sampler import NegativeSampling
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn.models.lightgcn import BPRLoss
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.logger import JsonLoggerCallback, CSVLoggerCallback
from ray.tune import JupyterNotebookReporter, CLIReporter
from ray.tune.progress_reporter import TuneReporterBase
from ray.widgets.util import in_notebook

@dataclass
class LearningRateSchedulerConfigs:
    mode: Literal["min", "max"]
    factor: float
    patience: int
    min_lr: float

@dataclass
class BaseConfig:
    max_epoch: int
    device: str
    num_workers: int
    train_batch_size: int
    evaluation_batch_size: int
    optimizer_learning_rate: float
    weight_decay: float
    learning_rate_scheduler_configs: LearningRateSchedulerConfigs
    val_data_size: float
    test_data_size: float
    is_undirected: bool
    disjoint_train_ratio: float
    neg_sampling_ratio: float
    evaluation_num_negative: int
    num_neighbors: list[int]
    pin_memory: bool
    non_blocking: bool
    BPR_loss_lambda: float
    metrics_top_k: int
    metrics_aggregation: Literal['mean', 'median', 'min', 'max']

@dataclass
class EdgeMLPClassifierConfig:
    input_channels: int
    hidden_channels: int
    dropout: float

@dataclass
class GINEConfig:
    num_layers: int
    mlp_channels: list[int]
    dropout: float
    edge_dim: float

@dataclass
class GPSConfig:
    channels: int
    # pe_parameters: list[tuple[str, int]]
    num_layers: int
    attn_type: str 
    attn_kwargs: dict[str, Any]
    attentions_heads: int
    # mpnn: list[MessagePassing]
    mpnn: GINEConfig
    classifier: EdgeMLPClassifierConfig
    num_of_nodes: int
    num_initial_node_features: int
    dropout_prob: float
    act_func: str
    training_configs: BaseConfig

def convert_dict_params_to_dataclass(hyper_param_config: dict, model_config_dataclass: Type[GPSConfig]) -> GPSConfig:
    # Deep copy the hyperparameter configuration to avoid mutation
    copied_hyper_param_config = copy.deepcopy(hyper_param_config)
    copied_hyper_param_config["training_configs"]["learning_rate_scheduler_configs"] = LearningRateSchedulerConfigs(**copied_hyper_param_config["training_configs"]["learning_rate_scheduler_configs"])
    copied_hyper_param_config["training_configs"] = BaseConfig(**copied_hyper_param_config["training_configs"])
    copied_hyper_param_config["classifier"] = EdgeMLPClassifierConfig(**copied_hyper_param_config["classifier"])
    copied_hyper_param_config["mpnn"] = GINEConfig(**copied_hyper_param_config["mpnn"])
    
    model_config = model_config_dataclass(**copied_hyper_param_config)
    return model_config

def gps_model_factory(config: GPSConfig, graph: Data, model_class: Type[GPS]):
    classifier = EdgeMLPClassifier(
        input_channels=config.classifier.input_channels,
        hidden_channels=config.classifier.hidden_channels,
        dropout=config.classifier.dropout
    )

    pe_parameters: list[tuple[str, int]] = [
        ("random_walk_pe", graph.random_walk_pe.shape[1]), # pyright: ignore[reportAttributeAccessIssue]
        ("laplacian_eigenvector_pe", graph.laplacian_eigenvector_pe.shape[1]) # pyright: ignore[reportAttributeAccessIssue]
    ]

    mess_pass_nn: list[MessagePassing] = []
    for _ in range(config.mpnn.num_layers):
        nn = MLP(
            channel_list=config.mpnn.mlp_channels,
            dropout=config.mpnn.dropout,
        )
        mess_pass_nn.append(GINEConv(nn=nn, edge_dim=graph.num_edge_features))

    model = model_class(
        channels=config.channels,
        pe_parameters=pe_parameters,
        num_layers=config.num_layers,
        attn_type=config.attn_type,
        attn_kwargs=config.attn_kwargs,
        attentions_heads=config.attentions_heads,
        mpnn=mess_pass_nn,
        classifier=classifier,
        num_of_nodes=graph.num_nodes, # type: ignore
        num_initial_node_features=graph.num_node_features,
        dropout_prob=config.dropout_prob,
        act_func=config.act_func
    )
    return model

def get_single_dataset_loader(graph: Data, is_evaluating: bool, allowed_items_per_user: Optional[dict[int, torch.Tensor]],
                             item_node_ids: list[int], batch_size = 128,
                             neg_sampling_ratio = 3.0, evaluation_num_negative=50,
                             num_neighbors = [5, 5, 5], num_workers: int = 3, pin_memory: bool = True):
    """
    prerequisite need to have already applied RandomLinkSplit as edge_label_index attribute is required
    """
    assert (not is_evaluating) or (is_evaluating and allowed_items_per_user is not None), f"If training, loader allowed_items_per_user is not needed, if is_evaluating then allowed_items_per_user is needed to pair users with items they have not seen."
    supervision_edge_keep_transform = KeepUserToItemSupervisionEdges(
        attach_node_type_vector=is_evaluating,
        num_nodes=graph.num_nodes, # type: ignore
        item_node_ids=item_node_ids
    )

    graph = supervision_edge_keep_transform(graph)

    negative_sampler: NegativeSampling | None = None
    evaluation_transform: BuildRankingCandidatesTransform | None = None
    
    if not is_evaluating:
        negative_sampler = NegativeSampling(
            mode="triplet",
            amount=neg_sampling_ratio
        )
    else:
        assert allowed_items_per_user is not None, "allowed_items_per_user is needed in evaluation to pair users with items they have not interacted with."
        evaluation_transform = BuildRankingCandidatesTransform(
            allowed_items_per_user=allowed_items_per_user,  # dict[int, Tensor of global item ids]
            num_neg_eval=evaluation_num_negative,
            item_type=supervision_edge_keep_transform.item_type,
            allow_batch_fallback=True,
        )

    data_loader = LinkNeighborLoader(
        data=graph,
        num_neighbors=num_neighbors,
        edge_label_index=graph.edge_label_index, # user→item supervision edges
        batch_size=batch_size,
        # no sampler-based negatives in eval
        neg_sampling=negative_sampler,
        # attaches candidate negative items and targets if is_evaluating
        transform=evaluation_transform,
        edge_label=None,
        shuffle= not is_evaluating,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return data_loader

def _add_recommender_metrics_to_ray_hyper_param_search(metric: str, metric_mode: str) -> TuneReporterBase:
    if in_notebook():
        reporter = JupyterNotebookReporter(
            sort_by_metric=True,
            metric=metric,
            mode=metric_mode,
        )
    else:
        reporter = CLIReporter(
            sort_by_metric=True,
            metric=metric,
            mode=metric_mode,
        )

    reporter.add_metric_column("RetrievalMAP")
    reporter.add_metric_column("RetrievalNormalizedDCG")
    reporter.add_metric_column("RetrievalPrecision")
    reporter.add_metric_column("RetrievalRecall")
    return reporter


def generate_dataset_loaders(graph, allowed_items_per_user: dict[int, torch.Tensor], item_node_ids: list[int],
                             val_data_size = 0.1, test_data_size = 0.1, train_batch_size = 128,
                             evaluation_batch_size= 512, is_undirected = True,
                             disjoint_train_ratio = 0.4, neg_sampling_ratio = 3.0, evaluation_num_negative=50,
                             num_neighbors = [5, 5, 5], num_workers: int = 3, pin_memory: bool = True):
    graph_split_transform = RandomLinkSplit(
        num_val=val_data_size,
        num_test=test_data_size,
        is_undirected=is_undirected,
        add_negative_train_samples=False,
        disjoint_train_ratio=disjoint_train_ratio,
    )

    train_data, val_data, test_data = cast(tuple[Data, Data, Data], graph_split_transform(graph))

    train_loader = get_single_dataset_loader(
        graph=train_data,
        is_evaluating=False,
        allowed_items_per_user=None,
        item_node_ids=item_node_ids,
        batch_size=train_batch_size,
        neg_sampling_ratio=neg_sampling_ratio,
        evaluation_num_negative=-1,
        num_neighbors=num_neighbors,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = get_single_dataset_loader(
        graph=val_data,
        is_evaluating=True,
        allowed_items_per_user=allowed_items_per_user,
        item_node_ids=item_node_ids,
        batch_size=evaluation_batch_size,
        neg_sampling_ratio=-1,
        evaluation_num_negative=-evaluation_num_negative,
        num_neighbors=num_neighbors,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = get_single_dataset_loader(
        graph=test_data,
        is_evaluating=True,
        allowed_items_per_user=allowed_items_per_user,
        item_node_ids=item_node_ids,
        batch_size=evaluation_batch_size,
        neg_sampling_ratio=-1,
        evaluation_num_negative=-evaluation_num_negative,
        num_neighbors=num_neighbors,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # edge_label_index = train_data["user", "interacts", "music"].edge_label_index
    # edge_label = train_data["user", "interacts", "music"].edge_label

    # train_supervision_edge_keep_transform = KeepUserToItemSupervisionEdges(
    #     attach_node_type_vector=False,
    #     num_nodes=graph.num_nodes, # type: ignore
    #     item_node_ids=item_node_ids
    # )
    # eval_supervision_edge_keep_transform = KeepUserToItemSupervisionEdges(
    #     attach_node_type_vector=True,
    #     num_nodes=graph.num_nodes, # type: ignore
    #     item_node_ids=item_node_ids
    # )

    # train_data = train_supervision_edge_keep_transform(train_data)
    # val_data = eval_supervision_edge_keep_transform(val_data)
    # test_data = eval_supervision_edge_keep_transform(test_data)
    
    # negative_sampler = NegativeSampling(
    #     mode="triplet",
    #     amount=neg_sampling_ratio
    # )
    
    # val_transform = BuildRankingCandidatesTransform(
    #     allowed_items_per_user=allowed_items_per_user,  # dict[int, Tensor of global item ids]
    #     num_neg_eval=evaluation_num_negative,
    #     item_type=eval_supervision_edge_keep_transform.item_type,
    #     allow_batch_fallback=True,
    # )

    # test_transform = BuildRankingCandidatesTransform(
    #     allowed_items_per_user=allowed_items_per_user,  # dict[int, Tensor of global item ids]
    #     num_neg_eval=evaluation_num_negative,
    #     item_type=eval_supervision_edge_keep_transform.item_type,
    #     allow_batch_fallback=True,
    # )

    # train_loader = LinkNeighborLoader(
    #     data=train_data,
    #     num_neighbors=num_neighbors,
    #     edge_label_index=train_data.edge_label_index, # user→item supervision edges
    #     batch_size=train_batch_size,
    #     neg_sampling=negative_sampler,
    #     edge_label=None,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory
    # )

    # val_loader = LinkNeighborLoader(
    #     data=val_data,
    #     num_neighbors=num_neighbors,
    #     edge_label_index=val_data.edge_label_index, # user→item supervision edges
    #     batch_size=evaluation_batch_size,
    #     neg_sampling=None, # no sampler-based negatives in eval
    #     transform=val_transform, # attaches candidate negative items and targets
    #     edge_label=None,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory
    # )

    # test_loader = LinkNeighborLoader(
    #     data=test_data,
    #     num_neighbors=num_neighbors,
    #     edge_label_index=test_data.edge_label_index, # user→item supervision edges
    #     batch_size=evaluation_batch_size,
    #     edge_label=None,
    #     neg_sampling=None, # no sampler-based negatives in eval
    #     transform=test_transform, # attaches candidate negative items and targets
    #     shuffle=False,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory
    # )

    return train_loader, val_loader, test_loader

def report_with_optional_checkpoint(
    epoch: int,
    model,
    optimizer,
    metrics: dict,
    checkpoint_interval: int = 5,
):
    """
    Report metrics to Ray Tune and optionally include a checkpoint.

    Args:
        epoch (int): Current epoch index (0-based).
        model (nn.Module): The model being trained.
        optimizer (Optimizer): The optimizer for saving state.
        metrics (dict): Metrics to report.
        checkpoint_interval (int): How often to checkpoint (every N epochs).
    """
    should_checkpoint = (epoch + 1) % checkpoint_interval == 0

    if should_checkpoint:
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")

            # Build checkpoint state
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

            torch.save(state, path)

            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            tune.report(metrics=metrics, checkpoint=checkpoint)
    else:
        # Fast path: metrics only
        tune.report(metrics=metrics)


def ray_tune_train_gnn(
        config: dict[str, Any],
        train_dataset: Data,
        val_dataset: Data,
        graph: Data,
        item_node_ids: list[int],
        allowed_items_per_user: dict[int, torch.Tensor],
        model_class: Type[GPS]):

    # train_loader, val_loader, test_loader = generate_dataset_loaders(
    #     graph=graph_ref,
    #     batch_size=config["batch_size"],
    #     neg_sampling_ratio=config["neg_sampling_ratio"],
    #     num_neighbors=config["num_neighbors"]
    # )
    gps_config = convert_dict_params_to_dataclass(config, GPSConfig)

    train_loader = get_single_dataset_loader(
        graph=train_dataset,
        is_evaluating=False,
        allowed_items_per_user=None,
        item_node_ids=item_node_ids,
        batch_size=gps_config.training_configs.train_batch_size,
        neg_sampling_ratio=gps_config.training_configs.neg_sampling_ratio,
        evaluation_num_negative=-1,
        num_neighbors=gps_config.training_configs.num_neighbors,
        num_workers=gps_config.training_configs.num_workers,
        pin_memory=gps_config.training_configs.pin_memory
    )

    val_loader = get_single_dataset_loader(
        graph=val_dataset,
        is_evaluating=True,
        allowed_items_per_user=allowed_items_per_user,
        item_node_ids=item_node_ids,
        batch_size=gps_config.training_configs.evaluation_batch_size,
        neg_sampling_ratio=-1,
        evaluation_num_negative=gps_config.training_configs.evaluation_num_negative,
        num_neighbors=gps_config.training_configs.num_neighbors,
        num_workers=gps_config.training_configs.num_workers,
        pin_memory=gps_config.training_configs.pin_memory
    )

    model = gps_model_factory(
        config=gps_config,
        graph=graph,
        model_class=model_class
    ).to(gps_config.training_configs.device)

    # GNN_model.to(config["device"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=gps_config.training_configs.optimizer_learning_rate,
        weight_decay=gps_config.training_configs.weight_decay
    )

    learning_rate_scheduler = ReduceLROnPlateau(
        optimizer,
        mode=gps_config.training_configs.learning_rate_scheduler_configs.mode,
        factor=gps_config.training_configs.learning_rate_scheduler_configs.factor,
        patience=gps_config.training_configs.learning_rate_scheduler_configs.patience,
        min_lr=gps_config.training_configs.learning_rate_scheduler_configs.min_lr
    )

    criterion = BPRLoss(gps_config.training_configs.BPR_loss_lambda)

    # ray hyper param search scheduler should handle this
    # early_stopper = EarlyStopper(patience=3, mode="max")

    metrics_tracker = metrics_tracker_factory(
        top_k=gps_config.training_configs.metrics_top_k,
        aggregation=gps_config.training_configs.metrics_aggregation
    ).to(gps_config.training_configs.device)

    for epoch in range(gps_config.training_configs.max_epoch):
        train_loss = train_step(
            model=model,
            dataloader=train_loader,
            loss_fn=criterion,
            optimizer=optimizer,
            device=torch.device(gps_config.training_configs.device),
            non_blocking=gps_config.training_configs.non_blocking
        )

        # during evaluation and inference we do apply reg as its used in training the model
        # to prevent overfitting. Thus, loss during eval/inference evaluation is on pure ranking scores
        # with no penalties. The way BPRLoss is implemented if we set .lambda_reg to zero it wont apply
        # reg. if we do not change it to zero then it will apply reg but we are not providing its .forward
        # method the user embeddings, thus it crashes, so set it to zero then change it back so its applied
        # during training when we do provide it the user embeddings
        save_train_lambda: float = criterion.lambda_reg
        criterion.lambda_reg = 0.0

        test_loss = test_step(
            model=model,
            dataloader=val_loader,
            loss_fn=criterion,
            device=torch.device(gps_config.training_configs.device),
            non_blocking=gps_config.training_configs.non_blocking,
            metricsTrackers=metrics_tracker,
        )
        criterion.lambda_reg = save_train_lambda

        raw_epoch_metrics = metrics_tracker.compute()
        metrics_tracker.reset()

        epoch_additional_metrics_res = normalize_metrics_dict(raw_epoch_metrics)

        epoch_logs: dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
        }

        learning_rate_scheduler.step(epoch_additional_metrics_res['RetrievalNormalizedDCG'])


        report_with_optional_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            metrics=epoch_logs | epoch_additional_metrics_res,
            checkpoint_interval=5,  # every 5 epochs
        )
        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        #     # checkpoint = None
        #     path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
        #     # if (epoch + 1) % 5 == 0:
        #     # This saves the model to the trial directory
        #     torch.save(
        #         (model.state_dict(), optimizer.state_dict()), path
        #     )
        #     checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

        #     # Send the current training result back to Tune
        #     tune.report(
        #         metrics=epoch_logs | epoch_additional_metrics_res,
        #         checkpoint=checkpoint
        #     )

# @remote(num_gpus=1)
def test_best_model(
        best_result,
        graph_ref,
        test_dataset: Data,
        item_node_ids: list[int],
        allowed_items_per_user: dict[int, torch.Tensor],
        model_class: Type[GPS] = GPS,):
    gps_config: GPSConfig = convert_dict_params_to_dataclass(best_result.config, GPSConfig)
    device = gps_config.training_configs.device

    best_trained_model = gps_model_factory(
        config=gps_config,
        graph=ray.get(graph_ref),
        model_class=model_class
    )

    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    # model_state = torch.load(checkpoint_path, map_location=device)
    model_state, optimizer_state, epoch = torch.load(checkpoint_path, map_location=device)
    del optimizer_state
    best_trained_model.load_state_dict(model_state)

    # train_loader, val_loader, test_loader = generate_dataset_loaders(
    #     graph=ray.get(graph_ref),
    #     batch_size=best_result.config["batch_size"],
    #     neg_sampling_ratio=best_result.config["neg_sampling_ratio"],
    #     num_neighbors=best_result.config["num_neighbors"]
    # )

    criterion = BPRLoss(gps_config.training_configs.BPR_loss_lambda)

    # during evaluation and inference we do apply reg as its used in training the model
    # to prevent overfitting. Thus, loss during eval/inference evaluation is on pure ranking scores
    # with no penalties. The way BPRLoss is implemented if we set .lambda_reg to zero it wont apply
    # reg. if we do not change it to zero then it will apply reg but we are not providing its .forward
    # method the user embeddings, thus it crashes, so set it to zero then change it back so its applied
    # during training when we do provide it the user embeddings
    criterion.lambda_reg = 0.0

    metrics_tracker = metrics_tracker_factory(
        top_k=gps_config.training_configs.metrics_top_k,
        aggregation=gps_config.training_configs.metrics_aggregation
    ).to(device)

    test_loader = get_single_dataset_loader(
        graph=test_dataset,
        is_evaluating=True,
        allowed_items_per_user=allowed_items_per_user,
        # item_node_ids=[item_node_id for item_node_id in item_node_id_map.values()],
        item_node_ids=item_node_ids,
        batch_size=gps_config.training_configs.evaluation_batch_size,
        neg_sampling_ratio=-1,
        evaluation_num_negative=gps_config.training_configs.evaluation_num_negative,
        num_neighbors=gps_config.training_configs.num_neighbors,
        num_workers=gps_config.training_configs.num_workers,
        pin_memory=gps_config.training_configs.pin_memory
    )

    test_loss = test_step(
        model=best_trained_model,
        dataloader=test_loader,
        loss_fn=criterion,
        device=device, # pyright: ignore[reportArgumentType]
        non_blocking=gps_config.training_configs.non_blocking,
        metricsTrackers=metrics_tracker
    )

    test_metrics = metrics_tracker.compute()
    test_metrics["test_loss"] = test_loss

    print("Best trial evaluation on test set results:")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

def hyper_param_search_driver(config: dict[str, Any], graph: Data,
                             train_dataset: Data, validation_dataset: Data, test_dataset: Data,
                             item_node_ids: list[int], allowed_items_per_user: dict[int, torch.Tensor],
                             num_samples:int = 5, model_class: Type[GPS] = GPS,
                             max_num_epochs: int = 5, cpu_per_trial: int = 2, gpus_per_trial: float = 0.2,
                             grace_period: int = 1, reduction_factor: int = 2, metric: str = "RetrievalNormalizedDCG", metric_mode: str = "max",
                             mlflow_tracking_uri: str = "http://localhost:8080", scope: Literal["all", "last", "avg", "last-5-avg", "last-10-avg"] = "avg") -> dict[str, Any]:
    reporter = _add_recommender_metrics_to_ray_hyper_param_search(metric, metric_mode)

    experiment_name = f"gps_rec_sys_{generate_now_timestamp_str()}"
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=grace_period,
        reduction_factor=reduction_factor
    )

    shared_graph_ref = ray.put(graph)
    train_dataset_ref = ray.put(train_dataset)
    validation_dataset_ref = ray.put(validation_dataset)
    item_node_ids_ref = ray.put(item_node_ids)
    allowed_items_per_user_ref = ray.put(allowed_items_per_user)
    model_class_ref = ray.put(model_class)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                ray_tune_train_gnn,
                train_dataset=train_dataset_ref,
                val_dataset=validation_dataset_ref,
                graph=shared_graph_ref,
                item_node_ids=item_node_ids_ref,
                allowed_items_per_user=allowed_items_per_user_ref,
                model_class=model_class_ref
            ),
            resources={"cpu": cpu_per_trial, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=metric_mode,
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=tune.RunConfig(
            name=experiment_name,
            progress_reporter=reporter,
            # storage_path=root_storage_output_dir,
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=mlflow_tracking_uri,
                    experiment_name=experiment_name,
                    save_artifact=True,
                    log_params_on_trial_end=True
                ),
                JsonLoggerCallback(),
                CSVLoggerCallback()
            ]
            )
    )
    results = tuner.fit()

    best_result = results.get_best_result(metric, metric_mode, scope=scope)

    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics['test_loss'] if best_result.metrics and 'test_loss' in best_result.metrics else 'error'}")
    print(f"Best trial final RetrievalNormalizedDCG: {best_result.metrics['RetrievalNormalizedDCG'] if best_result.metrics and 'RetrievalNormalizedDCG' in best_result.metrics else 'error'}")

    # Call the test_best_model function asynchronously
    if best_result.config:
        test_accuracy_future = test_best_model(
            best_result=best_result,
            graph_ref=shared_graph_ref,
            test_dataset=test_dataset,
            item_node_ids=item_node_ids,
            allowed_items_per_user=allowed_items_per_user,
            model_class=model_class
        )
        return best_result.config

    print(f"model testing Failed: {best_result.config = }")
    raise RuntimeError(f"Error: No best model results found {best_result = }")

def align_mpnn_to_node_embedding_size(spec: dict) -> list[int]:
    # mlp_num_layers = random.randint(1, 4)
    # each GT layer has one GINE GNN with an MLP thus we need number of num_layers of MLP per GPS layer
    mlp_num_layers = spec["num_layers"]
    node_embedding_size: int = spec["channels"]
    mlp_channels_per_layer: list[int] = [node_embedding_size]

    while len(mlp_channels_per_layer) < mlp_num_layers:
        mlp_channels_per_layer.append(random.choice([64, 128, 256, 512]))

    # mlp_channels_per_layer[mlp_num_layers - 1] = node_embedding_size
    mlp_channels_per_layer.append(node_embedding_size)
    return mlp_channels_per_layer