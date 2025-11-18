import torch
from torch_geometric.data import Data
from typing import Optional
from tqdm.auto import tqdm
from torchmetrics import MetricCollection
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


from gnn_utils.utils import EarlyStopper

def build_ranking_candidates_with_allowed_per_user(
    batch,
    num_neg_eval: int,
    allowed_items_per_user: dict[int, torch.Tensor],  # global item IDs (CPU is fine)
    item_type: int = 1,
    allow_batch_fallback: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build ranking candidates for a batch using precomputed per-user allowed items.

    For each supervision edge (user -> pos_item) in this batch:
      1) Start from the user's allowed unseen items (global IDs).
      2) Restrict to items that actually appear in this batch subgraph.
      3) Try to sample `num_neg_eval` negatives from that set.
      4) If there aren't enough and `allow_batch_fallback=True`, fill the
         remaining negatives by sampling from *any* item in this batch
         (may include seen items, but keeps things simple and fast).
      5) If we still can't reach `num_neg_eval`, skip that user.

    Returns:
        cand_items_local:      [N] local item indices (for z[...] indexing)
        targets:               [N] 1 (positive) or 0 (negative)
        user_local_for_cands:  [N] local user indices
        user_global_for_cands: [N] global user IDs (for metrics grouping)
    """
    device = batch.x.device

    # Global -> local mapping for nodes in this subgraph
    # batch.n_id: [num_local_nodes], global node IDs
    n_id = batch.n_id
    global_to_local = {int(g): i for i, g in enumerate(n_id.tolist())}

    # Get node types for this subgraph
    # batch.node_type is global-length; select types of nodes in this batch:
    global_node_type = batch.node_type              # [num_global_nodes]
    batch_node_type = global_node_type[n_id]        # [num_local_nodes]

    # Collect *global* IDs of items present in this batch subgraph
    batch_items_mask = (batch_node_type == item_type)        # [num_local_nodes]
    batch_items_global = n_id[batch_items_mask].tolist()     # list[int]
    if not batch_items_global:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty, empty

    # For fast membership test + sampling
    batch_items_set = set(batch_items_global)
    batch_items_list = list(batch_items_set)
    num_batch_items = len(batch_items_list)

    # Supervision edges in THIS batch (local indices)
    users_local = batch.edge_label_index[0]      # [B]
    pos_items_local = batch.edge_label_index[1]  # [B]
    users_global = n_id[users_local]             # [B], global node ids

    cand_items_local_list = []
    targets_list = []
    user_local_list = []
    user_global_list = []

    for u_loc, u_glob, pos_loc in zip(users_local, users_global, pos_items_local):
        u_loc = int(u_loc)
        u_glob = int(u_glob)

        # 1) Get this user's allowed unseen items (global IDs)
        allowed = allowed_items_per_user.get(u_glob)
        if allowed is not None and allowed.numel() > 0:
            allowed_list = allowed.tolist()
            # 2) Restrict to items present in this batch
            allowed_in_batch = [g for g in allowed_list if g in batch_items_set]
        else:
            allowed_in_batch = []

        neg_globals: list[int] = []

        # 3) Use as many true unseen-in-batch items as possible
        if allowed_in_batch:
            if len(allowed_in_batch) >= num_neg_eval:
                # sample K from allowed_in_batch (with replacement)
                idx = torch.randint(0, len(allowed_in_batch),
                                    (num_neg_eval,))
                neg_globals.extend(allowed_in_batch[i] for i in idx.tolist())
            else:
                # take all allowed_in_batch and remember we still need more
                neg_globals.extend(allowed_in_batch)

        # 4) If not enough and fallback is allowed, fill from any batch item
        if len(neg_globals) < num_neg_eval and allow_batch_fallback:
            remaining = num_neg_eval - len(neg_globals)
            idx_fb = torch.randint(0, num_batch_items, (remaining,))
            neg_globals.extend(batch_items_list[i] for i in idx_fb.tolist())

        # If we STILL don't have enough, skip this user
        if len(neg_globals) < num_neg_eval:
            continue

        # Just in case we overshoot (shouldn't happen), truncate
        neg_globals = neg_globals[:num_neg_eval]

        # Map global -> local node ids; if something is missing, skip user
        try:
            neg_locals = [global_to_local[int(g)] for g in neg_globals]
        except KeyError:
            # Some sampled item not in this subgraph mapping; skip this user
            continue

        neg_locals_tensor = torch.tensor(
            neg_locals, dtype=torch.long, device=device
        )

        # candidates: 1 positive + K negatives (K = num_neg_eval)
        items_local = torch.cat([
            pos_loc.unsqueeze(0).to(device),  # [1]
            neg_locals_tensor,               # [K]
        ])

        cand_items_local_list.append(items_local)

        # targets: [1, 0, ..., 0]
        tgt = torch.zeros(1 + num_neg_eval, dtype=torch.long, device=device)
        tgt[0] = 1
        targets_list.append(tgt)

        u_loc_vec = torch.full(
            (1 + num_neg_eval,), u_loc, dtype=torch.long, device=device
        )
        u_glob_vec = torch.full(
            (1 + num_neg_eval,), int(u_glob), dtype=torch.long, device=device
        )
        user_local_list.append(u_loc_vec)
        user_global_list.append(u_glob_vec)

    # Edge case: nothing to evaluate in this batch
    if not cand_items_local_list:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty, empty

    cand_items_local      = torch.cat(cand_items_local_list)
    targets               = torch.cat(targets_list)
    user_local_for_cands  = torch.cat(user_local_list)
    user_global_for_cands = torch.cat(user_global_list)

    # Safety: all indices must be valid local node ids
    num_local_nodes = n_id.size(0)
    assert cand_items_local.max().item() < num_local_nodes
    assert cand_items_local.min().item() >= 0
    assert user_local_for_cands.max().item() < num_local_nodes
    assert user_local_for_cands.min().item() >= 0

    return cand_items_local, targets, user_local_for_cands, user_global_for_cands

def build_bpr_batch(
    model,
    batch:Data,
    z: torch.Tensor,
    items_neg_vector: Optional[torch.Tensor] = None,
    items_positive_vector: Optional[torch.Tensor] = None,
    K: Optional[int] = None
):
    """
    Constructs all tensors needed for BPR loss from a LinkNeighborLoader batch.

    Args:
        model: Your GraphGPS model (must have .classifier(...) and .get_embeddings(...)).
        batch: A mini-batch from LinkNeighborLoader (contains src_index, dst_pos_index, dst_neg_index).
        z: Node embeddings from the encoder for all nodes in the batch
           (shape: [num_batch_nodes, embedding_dim]).

    Returns:
        pos_scores_expanded: 1D tensor of shape [B * K]
        neg_scores:          1D tensor of shape [B * K]
        node_embedding_parameters:  Tensor for L2 reg
        B: number of positive examples
        K: number of negatives per positive
    """
    # Parameters for L2 regularization
    node_embedding_parameters = model.get_embeddings(batch.n_id)

    # Extract triplet components
    users = batch.src_index              # [B]
    # items_pos = batch.dst_pos_index      # [B]
    
    # [B] final size regardless of items_positive_vector provided
    items_pos = batch.dst_pos_index if items_positive_vector is None else items_positive_vector
    # items_neg = batch.dst_neg_index      # [B, K]
    if items_neg_vector is None:
        items_neg: torch.Tensor = batch.dst_neg_index # [B, K]
    else:
        assert K != None, "items_neg_vector is provided K is needed to know how many times more there are negative items than positive items"
        items_neg = items_neg_vector #[B*k]

    if K is None:
        K = items_neg.shape[1]
    B = items_neg.shape[0]
    # Flatten negatives to [B*K]
    # items_neg_flat = items_neg.reshape(-1)  # [B*K]
    items_neg_flat = items_neg.reshape(-1)  if items_neg_vector is None else items_neg_vector

    # Gather embeddings
    z_users = z[users]                    # [B, d]
    z_pos   = z[items_pos]                # [B, d]
    z_neg   = z[items_neg_flat]           # [B*K, d]

    # Positive pair scores
    pos_scores = model.classifier(z_users, z_pos).view(-1)  # [B]

    # Expand positives to shape [B*K]
    pos_scores_expanded = (
        pos_scores
        .unsqueeze(1)                     # [B, 1]
        .expand(-1, K)                    # [B, K]
        .reshape(-1)                      # [B*K]
    )

    # Negative pair scores
    d = z_users.size(1)
    z_users_expanded = (
        z_users
        .unsqueeze(1)                     # [B, 1, d]
        .expand(-1, K, -1)                # [B, K, d]
        .reshape(-1, d)                   # [B*K, d]
    )

    # neg_scores = model.classifier(z_users_expanded, z_neg).view(-1)  # [B*K]
    neg_scores = model.classifier(z_users_expanded, z_neg)  # [B*K]

    return pos_scores_expanded, neg_scores, node_embedding_parameters, B, K

def build_bpr_from_candidate_scores(
    scores: torch.Tensor,
    batch: Data,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Given flat candidate scores of shape [B * (1 + K)] produced from
    build_ranking_candidates(), reconstruct BPR-style positive and negative
    score tensors.

    Assumes:
      - For each user u, scores are ordered as:
            [pos_item_score, neg1, neg2, ..., negK]
      - batch.edge_label_index[0] has length B (one positive per user).

    Returns:
        pos_scores_expanded: Tensor [B*K], positive scores tiled K times
        neg_scores_flat:     Tensor [B*K], negative scores
        B:                   number of users in this batch
        K:                   number of negatives per user
    """
    # Number of users in this batch (one positive edge per user)
    B = batch.edge_label_index.shape[1]

    N = scores.numel()
    # N = B * (1 + K)  →  K = N/B - 1
    K = N // B - 1

    scores_2d = scores.view(B, 1 + K)  # [B, 1+K]

    pos_scores = scores_2d[:, 0]       # [B]
    neg_scores = scores_2d[:, 1:]      # [B, K]

    # Flatten negatives [B*K]
    neg_scores_flat = neg_scores.reshape(-1)

    # Expand positives [B] → [B*K]
    pos_scores_expanded = (
        pos_scores
        .unsqueeze(1)                 # [B, 1]
        .expand(-1, K)                # [B, K]
        .reshape(-1)                  # [B*K]
    )

    return pos_scores_expanded, neg_scores_flat, B, K

def train_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device,
              non_blocking: bool) -> float:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss: float = 0.0
    # total_samples: int = 0
    num_batches: int = 0

    # Loop through data loader data batches
    for batch in tqdm(dataloader):
        # Send data to target device
        batch = batch.to(device, non_blocking=non_blocking)
        optimizer.zero_grad()  # Clear gradients.

        # used by performer linear attention that GraphGPS uses
        model.redraw_projection.redraw_projections() # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]

        # 1. Forward pass
        # GraphGPS encoder, returns embeddings for all local nodes
        z: torch.Tensor = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch) # [B, d]

        # # Parameters to regularize (e.g., user+item embedding tables)
        # node_embedding_parameters: torch.Tensor = model.get_embeddings(batch.n_id) # pyright: ignore[reportCallIssue]

        # # Triplet components (local node indices)
        # users = batch.src_index              # shape [B] users (local ids)
        # items_positive = batch.dst_pos_index # shape [B] positive items (local ids)
        # items_negative = batch.dst_neg_index # [B, amount] negative items (local ids)

        # B, K = items_negative.shape
        
        # # Flatten negatives to [B*K]
        # items_negative = items_negative.reshape(-1) # [B * K] negative items (local ids)
        # # items_negative_amount = batch.dst_neg_index.shape[1]

        # # Gather embeddings
        # z_users = z[users]              # [B, d]
        # z_positive = z[items_positive]  # [B, d]
        # z_negative = z[items_negative]  # [B * K, d]

        # pos_scores: torch.Tensor = model.classifier(z_users, z_positive) # [B] # pyright: ignore[reportCallIssue]
        # # Positive scores: [B] -> expand to [B*K]
        # pos_scores_exp: torch.Tensor = (
        #     pos_scores
        #     .unsqueeze(1)
        #     .expand(-1, K)
        #     .reshape(-1)  # [B * K]
        # )

        # d = z_users.shape[1]
        # # Expand user embeddings to match negatives: [B, d] -> [B*K, d]
        # z_users_expanded = (
        #     z_users
        #     .unsqueeze(1) # [B, 1, d]
        #     .expand(-1, K, -1) # [B, K, d]
        #     .reshape(-1, d) # [B * K, d]
        # )
        # print(z_users_expanded.shape)
        # neg_scores  = model.classifier(z_users_expanded, z_negative) # pyright: ignore[reportCallIssue] [B * K]

        # Build BPR batch
        pos_scores_exp, neg_scores, node_embedding_parameters, B, K = build_bpr_batch(
            model=model,
            batch=batch,
            z=z
        )

        # 2. Calculate  and accumulate loss
        loss = loss_fn(pos_scores_exp, neg_scores , node_embedding_parameters) # [B * amount]
        train_loss += loss.item()
        num_batches += 1
        # total_samples += loss.numel()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Adjust metrics to get average loss
    # train_loss = train_loss / total_samples
    # Average loss per batch
    train_loss = train_loss / num_batches
    return train_loss

@torch.no_grad()
def test_step(model: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          device: torch.device,
          non_blocking: bool,
          metricsTrackers: MetricCollection) -> float:
    """Tests a PyTorch model for a single epoch.

    This is mostly useful for debugging or tracking the training objective,
    NOT for real recommendation quality (use ranking metrics for that).

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss: float =  0.0
    # total_samples: int = 0
    num_batches: int = 0
    # fallback_stats: dict = {}

    # Loop through DataLoader batches
    for batch in tqdm(dataloader):
        # Send data to target device
        batch = batch.to(device, non_blocking=non_blocking)
        cand_items_local      = batch.cand_items_local      # [N]
        targets               = batch.targets               # [N]
        user_local_for_cands  = batch.user_local_for_cands  # [N]
        user_global_for_cands = batch.user_global_for_cands # [N]

        # candidate_items, targets, indexes, K = build_ranking_candidates(
        #     batch,
        #     eval_negative_sampler,
        #     TorchDevice("cpu")
        # )

        # 1) Build candidate sets with global negatives restricted to this batch
        # (
        #     cand_items_local,
        #     targets,
        #     user_local_for_cands,
        #     user_global_for_cands,
        # ) = build_ranking_candidates_global_batch(
        #     batch=batch,
        #     num_neg_eval=num_neg_eval,
        #     all_items=all_items,
        #     user_seen_items=user_seen_items,
        # )
        # (
        #     cand_items_local,
        #     targets,
        #     user_local_for_cands,
        #     user_global_for_cands,
        # ) = build_ranking_candidates_with_allowed_per_user(
        #     batch=batch,
        #     num_neg_eval=num_neg_eval,
        #     allowed_items_per_user=allowed_items_per_user,
        #     allow_batch_fallback= True
        # )

        # Edge case: nothing to evaluate in this batch
        if cand_items_local.numel() == 0:
            print("nothing to evaluate in this batch, unable to find global negatives that are in this batch")
            continue

        # node_embedding_parameters = model.get_embeddings(batch.n_id) # pyright: ignore[reportCallIssue]
        # positive_items = candidate_items[targets == 1] #[B]
        # negative_items = candidate_items[targets == 0] #[B * (k-1)]
        # print(f"Test_step: {positive_items.shape = }")
        # print(f"Test_step: {negative_items.shape = }")

        # IMPORTANT: val_loader should be built with neg_sampling=None
        # and edge_label_index = user→item supervision edges.
        z: torch.Tensor = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch,
        )
        # 3) Local indexing into z
        z_users = z[user_local_for_cands]   # [N, d]
        z_items = z[cand_items_local]       # [N, d]
        # z_users = z[indexes] # [B*(1+K), d]
        # z_items = z[candidate_items] # [B*(1+K), d]

        # Scores for candidates
        scores = model.classifier(z_users, z_items).view(-1)  # pyright: ignore[reportCallIssue] # [B*(1+K)]

        # pos_scores_exp, neg_scores, node_embedding_parameters, B, K = build_bpr_batch(
        #     model=model,
        #     batch=batch,
        #     z=z,
        #     items_neg_vector=negative_items,
        #     items_positive_vector=positive_items,
        #     K=K
        # )

        #  BPR-style loss from candidate scores
        pos_scores_exp, neg_scores_flat, B, K = build_bpr_from_candidate_scores(
            scores,
            batch,
        )

        # eval loss *without* extra L2 reg on node parameters since there is no backpropagation
        loss = loss_fn(pos_scores_exp, neg_scores_flat, None)

        test_loss += loss.item()
        # total_samples += loss.numel()
        num_batches += 1

        # update running metrics
        # metricsTrackers.update(scores, targets, indexes)
                # 5) Ranking metrics: use *global* user IDs for grouping
        metricsTrackers.update(
            scores,
            targets,
            user_global_for_cands,
        )

    # Adjust metrics to get average loss and accuracy per batch
    # test_loss = test_loss / total_samples
    # Average loss per batch
    test_loss = test_loss / num_batches
    return test_loss

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          learning_rate_scheduler: LRScheduler,
          loss_fn: torch.nn.Module,
          early_stopper: EarlyStopper,
          epochs: int,
          device: torch.device,
          non_blocking: bool,
          metricsTrackers: MetricCollection) -> dict[str, list]:
    """
    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]}
    """
    # Create empty results dictionary
    results: dict[str, list] = {metric_key: [] for metric_key in metricsTrackers.keys()} # pyright: ignore[reportAssignmentType]
    results |= {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": [],
               "lr": []
    }

    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        train_loss = train_step(model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            non_blocking=non_blocking
        )
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(f"starting train_step")
        test_loss = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            non_blocking=non_blocking,
            metricsTrackers=metricsTrackers,
        )

        epoch_additional_metrics_res = metricsTrackers.compute()
        metricsTrackers.reset()
        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"RetrievalNormalizedDCG: {epoch_additional_metrics_res['RetrievalNormalizedDCG']:.4f}| "
          f"RetrievalPrecision: {epoch_additional_metrics_res['RetrievalPrecision']:.4f}| "
          f"RetrievalRecall: {epoch_additional_metrics_res['RetrievalRecall']:.4f}| "
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["lr"].append(optimizer.param_groups[0]["lr"])
        for metric_key, metric_result in epoch_additional_metrics_res.items():
          results[metric_key].append(metric_result)

        if isinstance(learning_rate_scheduler, ReduceLROnPlateau):
            learning_rate_scheduler.step(epoch_additional_metrics_res['RetrievalNormalizedDCG'])
        else:
            learning_rate_scheduler.step()
            
        early_stopper.step(epoch_additional_metrics_res['RetrievalNormalizedDCG'], model=model)
        if early_stopper.should_stop:
            print(f"Early stopping at epoch {epoch+1}. Best validation loss: {early_stopper.best:.4f}")
            break

    # Return the filled results at the end of the epochs
    return results