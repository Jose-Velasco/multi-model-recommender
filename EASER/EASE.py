"""
Adapted from the EASE model by Harald Steck (WWW 2019) and
the TorchEASE open-source implementation (Jay Franck, GitHub).https://github.com/franckjay/TorchEASE/blob/main/src/main/EASE.py
Extended with preprocessing, stratified splits, NDCG@20 eval, and λ grid search.
"""

import sys
import os
import logging
from typing import Optional, Dict, List, Tuple, Iterable
import random
import math

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

logger = logging.getLogger("TorchEASE_pipeline")
if not logger.handlers:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

# =========================
# Utilities
# =========================

def save_recommendations(recs: np.ndarray, path: str) -> None:
    """
    Save recommendations without user IDs.
    One line per user, k item IDs separated by spaces.
    """
    with open(path, "w") as f:
        for row in recs:
            f.write(" ".join(str(int(it)) for it in row) + "\n")


def load_train_txt_as_dataframe(
    path: str,
    user_col: str = "user",
    item_col: str = "item",
) -> pd.DataFrame:
    """
    Each line in the file: user_id item_1 item_2 ... item_n
    becomes one row per (user, item) pair.
    """
    users: List[int] = []
    items: List[int] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u = int(parts[0])
            seen = set()
            for it in parts[1:]:
                it_int = int(it)
                if it_int in seen:
                    continue  # dedupe per user
                seen.add(it_int)
                users.append(u)
                items.append(it_int)

    df = pd.DataFrame({user_col: users, item_col: items})
    return df


def kcore_filter(df: pd.DataFrame, user_col: str, item_col: str,
                 user_min: int = 1, item_min: int = 3) -> pd.DataFrame:
    """
    Iterative k-core filter (default: items seen by >=3 users; keep all users).
    """
    cur = df.copy()
    changed = True
    while changed:
        changed = False
        # filter users
        if user_min > 1:
            user_counts = cur[user_col].value_counts()
            good_users = set(user_counts[user_counts >= user_min].index)
            new = cur[cur[user_col].isin(good_users)]
            if len(new) != len(cur):
                cur = new
                changed = True
        # filter items
        item_counts = cur[item_col].value_counts()
        good_items = set(item_counts[item_counts >= item_min].index) if item_min > 1 else set(item_counts.index)
        new = cur[cur[item_col].isin(good_items)]
        if len(new) != len(cur):
            cur = new
            changed = True
    return cur


def stratified_user_split(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    eval_ratio: float = 0.2,
    seed: int = 42,
    warm_eval: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split of each user's items into TRAIN and EVAL.

    - For each user, randomly shuffle their items.
    - Select ~eval_ratio of items (at least 1 if possible) as eval.
    - Remaining items go to train.
    - Optionally require eval items to be "warm" (appear at least twice overall).

    """
    rng = random.Random(seed)

    # global item counts, used for warm_eval
    global_counts = df[item_col].value_counts().to_dict()

    train_rows: List[Tuple[int, int]] = []
    eval_rows: List[Tuple[int, int]] = []

    grouped = df.groupby(user_col)[item_col].apply(list)

    for u, items in grouped.items():
        items = list(items)
        rng.shuffle(items)

        n = len(items)
        if n == 0:
            continue

        # how many eval items for this user
        # at least 1 if user has >= 2 items; otherwise all go to train
        if n == 1:
            eval_size = 0
        else:
            eval_size = int(round(n * eval_ratio))
            eval_size = max(1, eval_size)
            eval_size = min(eval_size, n - 1)  # keep at least 1 train item

        if eval_size == 0:
            # everything to train
            train_rows.extend((u, i) for i in items)
            continue

        if warm_eval:
            # warm candidates = items that appear at least twice globally
            warm_items = [i for i in items if global_counts.get(i, 0) > 1]
        else:
            warm_items = items[:]

        # if not enough warm items, just fall back to shuffled list
        if len(warm_items) < eval_size:
            candidates = items
        else:
            candidates = warm_items

        # choose eval_size items from candidates
        chosen_eval = set(candidates[:eval_size])

        # remaining items are train
        train_items = [i for i in items if i not in chosen_eval]
        eval_items  = list(chosen_eval)

        # safety: if somehow no train items left, move one eval back to train
        if not train_items and eval_items:
            train_items.append(eval_items.pop())

        train_rows.extend((u, i) for i in train_items)
        eval_rows.extend((u, i) for i in eval_items)

    train_df = pd.DataFrame(train_rows, columns=[user_col, item_col])
    eval_df  = pd.DataFrame(eval_rows,  columns=[user_col, item_col])
    return train_df, eval_df


def cap_heavy_users(df: pd.DataFrame, user_col: str, item_col: str,
                    cap: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Randomly keep at most `cap` interactions per user (for training only).
    """
    rng = random.Random(seed)
    parts = []
    for u, grp in df.groupby(user_col):
        if len(grp) <= cap:
            parts.append(grp)
        else:
            parts.append(grp.sample(n=cap, random_state=rng.randint(0, 10**9)))
    return pd.concat(parts, axis=0, ignore_index=True)


def ndcg_at_k(recommended: List[int], ground_truth: Iterable[int], k: int = 20) -> float:
    """
    Binary relevance NDCG@k for one user.
    """
    gt = set(ground_truth)
    if not gt:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in gt:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(k, len(gt))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# =========================
# EASE model
# =========================
class TorchEASE:
    """
    Minimal EASE with optional user-row normalization and IDF weighting.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        user_col: str = "user",
        item_col: str = "item",
        score_col: Optional[str] = None,
        regularization: float = 500.0,
        user_row_norm: bool = True,
        idf_weight: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.df = df.copy()
        self.user_col = user_col
        self.item_col = item_col
        self.score_col = score_col
        self.regularization = float(regularization)
        self.user_row_norm = bool(user_row_norm)
        self.idf_weight = bool(idf_weight)
        self.logger = logger or logging.getLogger("TorchEASE_pipeline")

        # Filled in during fit()
        self.user2idx: Dict[int, int] = {}
        self.idx2user: np.ndarray = np.array([])
        self.item2idx: Dict[int, int] = {}
        self.idx2item: np.ndarray = np.array([])
        self.n_users: int = 0
        self.n_items: int = 0

        # Model parameters
        self.B: Optional[torch.Tensor] = None              # item-item weight matrix
        self.user_items: Optional[List[List[int]]] = None  # items seen by each user (internal ids)
        self.popular_items: Optional[torch.Tensor] = None  # global popularity (internal ids)

    # ----------------- helpers -----------------
    def _build_mappings(self) -> None:
        users = self.df[self.user_col].values
        items = self.df[self.item_col].values
        user2idx: Dict[int, int] = {}
        item2idx: Dict[int, int] = {}
        user_list: List[int] = []
        item_list: List[int] = []

        for u in users:
            if u not in user2idx:
                user2idx[u] = len(user_list)
                user_list.append(u)
        for it in items:
            if it not in item2idx:
                item2idx[it] = len(item_list)
                item_list.append(it)

        self.user2idx = user2idx
        self.idx2user = np.array(user_list, dtype=np.int64)
        self.item2idx = item2idx
        self.idx2item = np.array(item_list, dtype=np.int64)
        self.n_users = len(user_list)
        self.n_items = len(item_list)
        self.logger.info(f"Users: {self.n_users}, Items: {self.n_items}")

    # ----------------- fit -----------------
    def fit(self) -> None:
        """
        Fit EASE using the closed-form solution:

            G = X^T X + λI
            P = G^{-1}
            B = -P / diag(P), with diag(B) = 0
        """
        self._build_mappings()

        df = self.df
        u_idx = df[self.user_col].map(self.user2idx).values.astype(np.int64)
        i_idx = df[self.item_col].map(self.item2idx).values.astype(np.int64)

        # Base data (binary)
        if self.score_col is not None and self.score_col in df.columns:
            base = df[self.score_col].astype(np.float32).values
        else:
            base = np.ones(len(df), dtype=np.float32)

        # Optional user-row normalization
        if self.user_row_norm:
            counts = df.groupby(self.user_col)[self.item_col].count().to_dict()
            u_weights = np.array(
                [1.0 / math.sqrt(counts[int(u_ext)]) for u_ext in df[self.user_col].values],
                dtype=np.float32,
            )
        else:
            u_weights = np.ones_like(base, dtype=np.float32)

        # Optional IDF weight (by item df)
        if self.idf_weight:
            df_item = df.groupby(self.item_col)[self.user_col].nunique().to_dict()
            idf = {
                i_ext: math.log((1.0 + self.n_users) / (1.0 + df_item[i_ext]))
                for i_ext in df_item
            }
            i_weights = np.array(
                [idf[int(i_ext)] for i_ext in df[self.item_col].values],
                dtype=np.float32,
            )
        else:
            i_weights = np.ones_like(base, dtype=np.float32)

        data = (base * u_weights * i_weights).astype(np.float32)

        # Items per user (for masking and scoring)
        self.logger.info("Collecting items per user")
        user_items: List[List[int]] = [[] for _ in range(self.n_users)]
        for u, it in zip(u_idx, i_idx):
            user_items[u].append(int(it))
        self.user_items = user_items

        # Simple popularity ranking (for cold-start)
        item_counts = np.bincount(i_idx, minlength=self.n_items)
        popular_np = np.argsort(-item_counts)
        self.popular_items = torch.from_numpy(popular_np)

        # Sparse user–item matrix X (CPU)
        self.logger.info("Building sparse user-item matrix")
        indices_np = np.vstack([u_idx, i_idx])  # shape (2, nnz)
        indices = torch.from_numpy(indices_np).long()
        values = torch.from_numpy(data).float()

        X = torch.sparse_coo_tensor(
            indices, values, size=(self.n_users, self.n_items), dtype=torch.float32
        ).coalesce()

        # G = X^T X (dense)
        self.logger.info("Computing G = X^T X (dense)")
        G = torch.sparse.mm(X.t(), X).to_dense()

        # Add λ on the diagonal
        self.logger.info("Adding regularization to the diagonal")
        diag_indices = torch.arange(self.n_items)
        G[diag_indices, diag_indices] += self.regularization

        # Invert G
        self.logger.info("Inverting Gram matrix (this is the slow step)")
        P = torch.linalg.inv(G)

        # Compute B and zero out the diagonal
        self.logger.info("Computing item-item weight matrix B")
        diagP = torch.diag(P)
        B = -P / diagP.unsqueeze(0)
        B[diag_indices, diag_indices] = 0.0

        self.B = B

        # Free big intermediates
        del P, G, X
        self.logger.info("Fit finished")

    # ----------------- recommend -----------------
    def _recommend_for_user_idx(self, u_idx: int, k: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.B is None:
            raise RuntimeError("Call fit() before recommending.")
        assert self.user_items is not None
        assert self.popular_items is not None

        seen_list = self.user_items[u_idx]
        if len(seen_list) == 0:
            top_internal = self.popular_items[:k]
            scores = torch.zeros_like(top_internal, dtype=torch.float32)
            return top_internal, scores

        seen_items = torch.tensor(seen_list, dtype=torch.long)
        scores = self.B[seen_items].sum(dim=0)  # (n_items,)
        scores[seen_items] = -1e9

        if k >= scores.numel():
            top_scores, top_internal = torch.sort(scores, descending=True)
            top_scores = top_scores[:k]
            top_internal = top_internal[:k]
        else:
            top_scores, top_internal = torch.topk(scores, k)
        return top_internal, top_scores

    def recommend_all_users(self, k: int = 20) -> np.ndarray:
        if self.B is None:
            raise RuntimeError("Call fit() before recommending.")
        recs = np.zeros((self.n_users, k), dtype=np.int64)
        idx2item_t = torch.from_numpy(self.idx2item)
        for u_idx in range(self.n_users):
            top_internal, _ = self._recommend_for_user_idx(u_idx, k=k)
            top_items_ext = idx2item_t[top_internal].numpy()
            recs[u_idx, :len(top_items_ext)] = top_items_ext[:k]
        return recs

    def recommend_topk_for_external_user(self, u_external: int, k: int = 20) -> List[int]:
        """
        Convenience for evaluation on a split: get top-k item IDs for a given external user.
        """
        if u_external not in self.user2idx:
            # unknown user in this train split -> fall back to popularity
            assert self.popular_items is not None
            idx2item_t = torch.from_numpy(self.idx2item)
            return list(idx2item_t[self.popular_items[:k]].numpy())
        idx = self.user2idx[u_external]
        idx2item_t = torch.from_numpy(self.idx2item)
        top_internal, _ = self._recommend_for_user_idx(idx, k=k)
        return list(idx2item_t[top_internal].numpy())


# =========================
# Grid search + Pipeline
# =========================
def evaluate_ndcg(model: TorchEASE, eval_df: pd.DataFrame,
                  user_col: str, item_col: str, k: int = 20) -> float:
    """
    Mean NDCG@k over users present in eval_df.
    """
    per_user_gt = eval_df.groupby(user_col)[item_col].apply(list)
    scores = []
    for u, gt_items in per_user_gt.items():
        recs = model.recommend_topk_for_external_user(u, k=k)
        scores.append(ndcg_at_k(recs, gt_items, k))
    return float(np.mean(scores)) if scores else 0.0


def grid_search_lambda(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    user_col: str,
    item_col: str,
    lambdas: List[float],
    user_row_norm: bool = True,
    idf_weight: bool = False,
) -> Tuple[float, float, List[Tuple[float, float]]]:
    """
    Fit on train_df for each λ, evaluate NDCG@20 on val_df.

    Returns
    -------
    best_lambda : float
    best_ndcg   : float
    history     : list of (lambda, ndcg) pairs
    """
    best_lmbd, best_score = None, -1.0
    history: List[Tuple[float, float]] = []

    for l in lambdas:
        logger.info(f"[Grid] Training EASE with λ={l}")
        te = TorchEASE(
            train_df,
            user_col=user_col,
            item_col=item_col,
            regularization=l,
            user_row_norm=user_row_norm,
            idf_weight=idf_weight,
        )
        te.fit()
        ndcg = evaluate_ndcg(te, val_df, user_col, item_col, k=20)
        logger.info(f"[Grid] λ={l} -> NDCG@20={ndcg:.5f}")

        history.append((l, ndcg))

        if ndcg > best_score:
            best_score = ndcg
            best_lmbd = l

    return float(best_lmbd), float(best_score), history

def plot_lambda_curve(
    history: List[Tuple[float, float]],
    out_path: str = "ease_lambda_ndcg.png",
) -> None:
    """
    Save a simple plot of λ vs NDCG@20 to a PNG file.
    """
    if not history:
        logger.warning("No λ history to plot.")
        return

    lambdas = [h[0] for h in history]
    ndcgs   = [h[1] for h in history]

    plt.figure()
    plt.plot(lambdas, ndcgs, marker="o")
    plt.xscale("log")  # λ usually tuned on log scale
    plt.xlabel("lambda (log scale)")
    plt.ylabel("NDCG@20 (validation)")
    plt.title("EASE NDCG@20 vs lambda")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    logger.info(f"Saved λ vs NDCG@20 plot to: {out_path}")

def run_pipeline() -> None:
    """
    Full pipeline with fixed configuration, but ONLY for:
    - k-core preprocessing
    - train/val split
    - λ grid search with NDCG@20 on validation
    - plotting λ vs NDCG@20
    - optional test NDCG@20 evaluation
    """

    # ----------------- CONFIG -----------------
    train_path = "../train-1.txt"

    user_min       = 15       # users already have >=16 interactions
    item_min       = 3       # important for warm-start (items seen by >=3 users)
    eval_ratio = 0.2
    warm_eval      = True
    cap_per_user   = 0       # 0 = no cap (or e.g. 300 to cap heavy users)
    seed           = 42

    lambdas        = [100.0, 200.0, 500.0, 1000.0, 2000.0]
    user_row_norm  = True
    idf_weight     = True
    # ------------------------------------------

    # 1) load
    logger.info(f"Loading data: {train_path}")
    raw_df = load_train_txt_as_dataframe(train_path, user_col="user", item_col="item")

    # 2) k-core (keeps evaluation feasible for EASE)
    logger.info(f"Applying k-core filter: user_min={user_min}, item_min={item_min}")
    df = kcore_filter(raw_df, "user", "item", user_min=user_min, item_min=item_min)

    # 3) stratified split per user into TRAIN / EVAL
    logger.info(
        f"Stratified split per user (eval_ratio={eval_ratio}, warm_eval={warm_eval})"
    )
    train_df, eval_df = stratified_user_split(
        df,
        user_col="user",
        item_col="item",
        eval_ratio=eval_ratio,
        seed=seed,
        warm_eval=warm_eval,
    )

    # 4) heavy-user handling on TRAIN ONLY
    if cap_per_user > 0:
        logger.info(f"Capping heavy users in TRAIN to ≤{cap_per_user} interactions")
        train_df = cap_heavy_users(train_df, "user", "item", cap=cap_per_user, seed=seed)

    # 5) grid search λ on validation
    best_lmbd, best_val, history = grid_search_lambda(
        train_df,
        eval_df,
        "user",
        "item",
        lambdas=lambdas,
        user_row_norm=user_row_norm,
        idf_weight=idf_weight,
    )
    logger.info(f"Best λ={best_lmbd} with NDCG@20={best_val:.5f}")

    # Save λ vs NDCG@20 plot for the report
    plot_lambda_curve(history, out_path="../images/ease_lambda_ndcg.png")
    logger.info("Tuning pipeline finished (no submission written).")


if __name__ == "__main__":
    run_pipeline()