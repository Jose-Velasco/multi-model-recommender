"""
BPR-MF pipeline on train-1.txt

- k-core preprocessing (item_min=3)
- warm eval (test items appear at least twice globally)
- optional heavy-user cap
- optional IDF weighting of positives
- grid search over lambda (L2 reg) with NDCG@20
- records training loss per epoch for each lambda and plots curves
"""
# Portions of this file (initial skeleton and some functions) were generated with assistance from OpenAI's ChatGPT (GPT-5.1 Thinking) and then modified and debugged by the authors.

import sys
import os
import logging
from typing import Optional, Dict, List, Tuple, Iterable
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger("BPRMF_pipeline")
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


def save_recommendations(recs: np.ndarray, path: str) -> None:
    """
    Save recommendations without user IDs.
    One line per user, k item IDs separated by spaces.
    """
    with open(path, "w") as f:
        for row in recs:
            f.write(" ".join(str(int(it)) for it in row) + "\n")


def kcore_filter(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    user_min: int = 1,
    item_min: int = 3,
) -> pd.DataFrame:
    """
    Iterative k-core filter.
    Default here: keep items that appear in >=3 interactions, all users.
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
        if item_min > 1:
            item_counts = cur[item_col].value_counts()
            good_items = set(item_counts[item_counts >= item_min].index)
        else:
            item_counts = cur[item_col].value_counts()
            good_items = set(item_counts.index)
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
    - If warm_eval=True: eval items must appear at least twice globally
      (so they are "warm" targets).
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
        if n == 1:
            eval_size = 0
        else:
            eval_size = int(round(n * eval_ratio))
            eval_size = max(1, eval_size)
            eval_size = min(eval_size, n - 1)  # keep at least 1 train item

        if eval_size == 0:
            train_rows.extend((u, i) for i in items)
            continue

        if warm_eval:
            # warm candidates = items that appear at least twice globally
            warm_items = [i for i in items if global_counts.get(i, 0) > 1]
        else:
            warm_items = items[:]

        if len(warm_items) < eval_size:
            candidates = items
        else:
            candidates = warm_items

        chosen_eval = set(candidates[:eval_size])
        train_items = [i for i in items if i not in chosen_eval]
        eval_items = list(chosen_eval)

        # safety: ensure at least one train item
        if not train_items and eval_items:
            train_items.append(eval_items.pop())

        train_rows.extend((u, i) for i in train_items)
        eval_rows.extend((u, i) for i in eval_items)

    train_df = pd.DataFrame(train_rows, columns=[user_col, item_col])
    eval_df = pd.DataFrame(eval_rows, columns=[user_col, item_col])

    # ---- Ensure every eval item appears at least once in TRAIN ----
    train_items = set(train_df[item_col].unique())
    mask_unknown = ~eval_df[item_col].isin(train_items)

    if mask_unknown.any():
        moved = eval_df[mask_unknown]
        print(f"Moving {len(moved)} eval interactions with unseen items back to TRAIN")
        train_df = pd.concat([train_df, moved], ignore_index=True)
        eval_df = eval_df[~mask_unknown].reset_index(drop=True)
    # -------------------------------------------------------------------
    return train_df, eval_df


def cap_heavy_users(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    cap: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Randomly keep at most `cap` interactions per user (for training only).
    Helps prevent heavy users from dominating BPR updates.
    """
    rng = random.Random(seed)
    parts = []
    for u, grp in df.groupby(user_col):
        if len(grp) <= cap:
            parts.append(grp)
        else:
            # random sample of this user's interactions
            parts.append(grp.sample(
                n=cap,
                random_state=rng.randint(0, 10**9),
            ))
    return pd.concat(parts, axis=0, ignore_index=True)


def ndcg_at_k(
    recommended: List[int],
    ground_truth: Iterable[int],
    k: int = 20,
) -> float:
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
# BPR-MF model
# =========================

class BPRMF:
    """
    BPR-MF model with:
      - user & item embeddings
      - SGD over (u, i+, j-) triples
      - optional IDF-weighted positive sampling
      - logs training loss per epoch
    """

    def __init__(
        self,
        n_factors: int = 128,
        lr: float = 1e-3,
        reg: float = 1e-4,
        n_epochs: int = 50,
        n_neg: int = 1,
        use_idf_weight: bool = True,
        seed: int = 42,
    ):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_neg = n_neg
        self.use_idf_weight = use_idf_weight
        self.seed = seed

        # mappings
        self.user2idx: Dict[int, int] = {}
        self.idx2user: np.ndarray = np.array([], dtype=np.int64)
        self.item2idx: Dict[int, int] = {}
        self.idx2item: np.ndarray = np.array([], dtype=np.int64)

        # params
        self.P: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None

        # per-user positives
        self.user_pos: Optional[List[np.ndarray]] = None
        self.user_pos_sets: Optional[List[set]] = None
        self.user_pos_probs: Optional[List[Optional[np.ndarray]]] = None

        # simple popularity fallback
        self.popular_items: Optional[np.ndarray] = None

        # training loss per epoch
        self.loss_history: List[float] = []
        self.val_ndcg_history: List[Optional[float]] = []

    # ---- helpers ----

    def _build_mappings(self, df: pd.DataFrame, user_col: str, item_col: str) -> Tuple[np.ndarray, np.ndarray]:
        users = df[user_col].values
        items = df[item_col].values
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

        u_idx = df[user_col].map(self.user2idx).values.astype(np.int64)
        i_idx = df[item_col].map(self.item2idx).values.astype(np.int64)
        return u_idx, i_idx

    # ---- training ----

    def fit(
            self,
            train_df: pd.DataFrame,
            user_col: str = "user",
            item_col: str = "item",
            item_idf: Optional[Dict[int, float]] = None,
            val_df: Optional[pd.DataFrame] = None,
            eval_every: int = 5,
            k_eval: int = 20,
    ) -> None:
        """
        Train BPR-MF on implicit interactions from train_df.
        item_idf: external item_id -> idf weight; used when use_idf_weight=True.
        """
        rng = np.random.RandomState(self.seed)

        # mappings
        u_idx, i_idx = self._build_mappings(train_df, user_col, item_col)
        n_users = len(self.idx2user)
        n_items = len(self.idx2item)
        F = self.n_factors

        logger.info(f"[BPRMF] Users={n_users}, Items={n_items}, Factors={F}")

        # init embeddings
        scale = 1.0 / math.sqrt(F)
        self.P = rng.normal(0.0, scale, size=(n_users, F)).astype(np.float32)
        self.Q = rng.normal(0.0, scale, size=(n_items, F)).astype(np.float32)

        # build per-user positives
        user_pos: List[List[int]] = [[] for _ in range(n_users)]
        for uu, ii in zip(u_idx, i_idx):
            user_pos[uu].append(int(ii))
        self.user_pos = [np.array(lst, dtype=np.int64) for lst in user_pos]
        self.user_pos_sets = [set(lst) for lst in user_pos]

        # popularity (for cold-start / eval safety)
        item_counts = np.bincount(i_idx, minlength=n_items)
        self.popular_items = np.argsort(-item_counts)

        # optional IDF weights in internal index space
        if self.use_idf_weight and item_idf is not None:
            logger.info("[BPRMF] Using IDF-weighted positive sampling.")
            idf_internal = np.zeros(n_items, dtype=np.float32)
            for ext_id, w in item_idf.items():
                if ext_id in self.item2idx:
                    idf_internal[self.item2idx[ext_id]] = float(w)
            user_pos_probs: List[Optional[np.ndarray]] = []
            for pos_items in self.user_pos:
                if pos_items.size == 0:
                    user_pos_probs.append(None)
                    continue
                weights = idf_internal[pos_items]
                s = weights.sum()
                if s <= 0:
                    user_pos_probs.append(None)
                else:
                    user_pos_probs.append(weights / s)
            self.user_pos_probs = user_pos_probs
        else:
            logger.info("[BPRMF] Using uniform positive sampling.")
            self.user_pos_probs = [None for _ in range(n_users)]

        # SGD
        self.loss_history = []
        self.val_ndcg_history = []
        lr = self.lr
        reg = self.reg

        logger.info(f"[BPRMF] Starting training for {self.n_epochs} epochs, reg={reg}")

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            triple_count = 0

            user_order = rng.permutation(n_users)
            for u in user_order:
                pos_items = self.user_pos[u]
                if pos_items.size == 0:
                    continue

                # iterate each positive once per epoch
                for _ in pos_items:
                    i = self._sample_positive(u, rng)
                    for _ in range(self.n_neg):
                        j = self._sample_negative(u, rng, n_items)
                        loss_ij = self._update_triplet(u, i, j, lr, reg)
                        epoch_loss += loss_ij
                        triple_count += 1

            avg_loss = epoch_loss / max(1, triple_count)
            self.loss_history.append(float(avg_loss))

            val_ndcg = None
            if val_df is not None and eval_every is not None and eval_every > 0:
                if (epoch + 1) % eval_every == 0:
                    val_ndcg = self._eval_ndcg(val_df, user_col, item_col, k_eval)
                    logger.info(
                        f"[BPRMF] Epoch {epoch + 1}/{self.n_epochs} "
                        f"- avg loss={avg_loss:.6f}, val NDCG@{k_eval}={val_ndcg:.5f}"
                    )
                else:
                    logger.info(
                        f"[BPRMF] Epoch {epoch + 1}/{self.n_epochs} "
                        f"- avg loss={avg_loss:.6f}"
                    )
            else:
                logger.info(
                    f"[BPRMF] Epoch {epoch + 1}/{self.n_epochs} "
                    f"- avg loss={avg_loss:.6f}"
                )

            self.val_ndcg_history.append(
                float(val_ndcg) if val_ndcg is not None else None
            )

    def _sample_positive(self, u: int, rng: np.random.RandomState) -> int:
        pos_items = self.user_pos[u]
        probs = self.user_pos_probs[u]
        if probs is not None:
            return int(rng.choice(pos_items, p=probs))
        else:
            return int(rng.choice(pos_items))

    def _sample_negative(self, u: int, rng: np.random.RandomState, n_items: int) -> int:
        pos_set = self.user_pos_sets[u]
        while True:
            j = int(rng.randint(n_items))
            if j not in pos_set:
                return j

    def _update_triplet(
        self,
        u: int,
        i: int,
        j: int,
        lr: float,
        reg: float,
    ) -> float:
        """
        One SGD step for (u, i+, j-).
        Returns approx loss: -log sigma(x_ui - x_uj).
        """
        pu = self.P[u].copy()
        qi = self.Q[i].copy()
        qj = self.Q[j].copy()

        x_ui = float(np.dot(pu, qi))
        x_uj = float(np.dot(pu, qj))
        x_uij = x_ui - x_uj

        sigm = 1.0 / (1.0 + math.exp(-x_uij))
        grad = 1.0 - sigm  # = sigmoid(-x_uij)

        # gradient ascent on log-likelihood
        self.P[u] += lr * (grad * (qi - qj) - reg * pu)
        self.Q[i] += lr * (grad * pu - reg * qi)
        self.Q[j] += lr * (-grad * pu - reg * qj)

        # approx per-triple loss (without explicit reg term)
        loss = -math.log(sigm + 1e-10)
        return loss

    # ---- prediction / recommend ----

    def _scores_for_user_idx(self, u_idx: int) -> np.ndarray:
        """
        Raw scores for all items (internal indices) for a given user index.
        """
        if self.P is None or self.Q is None:
            raise RuntimeError("Model not fitted.")
        scores = self.P[u_idx] @ self.Q.T
        # do not recommend seen items
        seen = self.user_pos[u_idx]
        if seen.size > 0:
            scores[seen] = -1e9
        return scores

    def recommend_topk_for_external_user(self, u_external: int, k: int = 20) -> List[int]:
        """
        Get top-k item IDs for a given external user ID.
        """
        if self.popular_items is None:
            raise RuntimeError("Model not fitted.")
        if u_external not in self.user2idx:
            # unknown user -> popularity
            top_internal = self.popular_items[:k]
            return [int(self.idx2item[i]) for i in top_internal]

        u_idx = self.user2idx[u_external]
        scores = self._scores_for_user_idx(u_idx)
        if k >= scores.size:
            top_idx = np.argsort(-scores)[:k]
        else:
            top_idx = np.argpartition(-scores, k)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [int(self.idx2item[i]) for i in top_idx]

    def recommend_all_users(self, k: int = 20) -> np.ndarray:
        """
        Recommend top-k items for all users in internal index order.
        """
        if self.P is None or self.Q is None or self.popular_items is None:
            raise RuntimeError("Model not fitted.")
        n_users = self.P.shape[0]
        recs = np.zeros((n_users, k), dtype=np.int64)
        for u_idx in range(n_users):
            scores = self._scores_for_user_idx(u_idx)
            if k >= scores.size:
                top_idx = np.argsort(-scores)[:k]
            else:
                top_idx = np.argpartition(-scores, k)[:k]
                top_idx = top_idx[np.argsort(-scores[top_idx])]
            recs[u_idx] = self.idx2item[top_idx]
        return recs

    def _eval_ndcg(
        self,
        val_df: pd.DataFrame,
        user_col: str,
        item_col: str,
        k: int = 20,
    ) -> float:
        """
        Compute mean NDCG@k on val_df using current model parameters.
        """
        per_user_gt = val_df.groupby(user_col)[item_col].apply(list)
        scores = []
        for u, gt_items in per_user_gt.items():
            recs = self.recommend_topk_for_external_user(int(u), k=k)
            scores.append(ndcg_at_k(recs, gt_items, k))
        return float(np.mean(scores)) if scores else 0.0



# =========================
# Evaluation helpers
# =========================

def evaluate_ndcg_bpr(
    model: BPRMF,
    eval_df: pd.DataFrame,
    user_col: str,
    item_col: str,
    k: int = 20,
) -> float:
    """
    Mean NDCG@k over users in eval_df.
    """
    per_user_gt = eval_df.groupby(user_col)[item_col].apply(list)
    scores: List[float] = []
    for u, gt_items in per_user_gt.items():
        recs = model.recommend_topk_for_external_user(int(u), k=k)
        scores.append(ndcg_at_k(recs, gt_items, k))
    return float(np.mean(scores)) if scores else 0.0


def grid_search_bpr(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    user_col: str,
    item_col: str,
    n_factors_grid: List[int],
    lambdas: List[float],
    lr: float,
    n_epochs: int,
    n_neg: int,
    use_idf_weight: bool,
    seed: int = 42,
):
    """
    Grid search over (n_factors, lambda) for BPRMF.

    Returns
    -------
    best_k : int
    best_lambda : float
    best_ndcg : float
    ndcg_history : List[Tuple[int, float, float]]
        Each element: (k, lambda, ndcg@20)
    loss_history_per_combo : List[Tuple[Tuple[int, float], List[float]]]
        Each element: ((k, lambda), [loss_epoch1, ..., loss_epochN])
    """
    # Precompute IDF on train_df
    if use_idf_weight:
        df_item = train_df.groupby(item_col)[user_col].nunique().to_dict()
        n_users = train_df[user_col].nunique()
        item_idf = {
            int(i_ext): math.log((1.0 + n_users) / (1.0 + df_item[i_ext]))
            for i_ext in df_item
        }
    else:
        item_idf = None

    best_k: Optional[int] = None
    best_lmbd: Optional[float] = None
    best_score = -1.0
    best_loss_hist: Optional[List[float]] = None
    best_val_hist: Optional[List[Optional[float]]] = None


    ndcg_history: List[Tuple[int, float, float]] = []
    loss_history_per_combo: List[Tuple[Tuple[int, float], List[float]]] = []

    for k in n_factors_grid:
        for l in lambdas:
            logger.info(f"[Grid] Training BPRMF with n_factors={k}, lambda={l}")
            model = BPRMF(
                n_factors=k,
                lr=lr,
                reg=l,
                n_epochs=n_epochs,
                n_neg=n_neg,
                use_idf_weight=use_idf_weight,
                seed=seed,
            )
            model.fit(
                train_df,
                user_col=user_col,
                item_col=item_col,
                item_idf=item_idf,
                val_df=val_df,
                eval_every=5,  # validate every 5 epochs
                k_eval=20,
            )

            ndcg = evaluate_ndcg_bpr(model, val_df, user_col, item_col, k=20)
            logger.info(f"[Grid] k={k}, lambda={l} -> NDCG@20={ndcg:.5f}")

            ndcg_history.append((k, l, ndcg))
            loss_history_per_combo.append(((k, l), model.loss_history))

            if ndcg > best_score:
                best_score = ndcg
                best_k = k
                best_lmbd = l
                best_loss_hist = list(model.loss_history)
                best_val_hist = list(model.val_ndcg_history)

    assert best_k is not None and best_lmbd is not None
    return (
        best_k,
        best_lmbd,
        best_score,
        ndcg_history,
        loss_history_per_combo,
        best_loss_hist,
        best_val_hist,
    )


def plot_ndcg_by_k_lambda(
    ndcg_history: List[Tuple[int, float, float]],
    out_path: str = "../images/bpr_ndcg_k_lambda.png",
) -> None:
    """
    Plot lambda vs NDCG@20 for each n_factors as separate curves.
    """
    if not ndcg_history:
        logger.warning("No NDCG history to plot.")
        return

    # group by k
    k_values = sorted(set(k for (k, _, _) in ndcg_history))

    plt.figure()
    for k in k_values:
        xs = []
        ys = []
        for kk, lam, ndcg in ndcg_history:
            if kk == k:
                xs.append(lam)
                ys.append(ndcg)
        if not xs:
            continue
        xs, ys = zip(*sorted(zip(xs, ys), key=lambda t: t[0]))
        plt.plot(xs, ys, marker="o", label=f"k={k}")

    plt.xscale("log")
    plt.xlabel("lambda (reg, log scale)")
    plt.ylabel("NDCG@20 (validation)")
    plt.title("BPRMF NDCG@20 vs lambda for different n_factors")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved NDCG vs lambda plot to: {out_path}")


def plot_lambda_curve(
    history: List[Tuple[float, float]],
    out_path: str = "../images/bpr_lambda_ndcg.png",
) -> None:
    """
    Save a simple plot of lambda vs NDCG@20 to a PNG file.
    """
    if not history:
        logger.warning("No lambda history to plot.")
        return

    lambdas = [h[0] for h in history]
    ndcgs = [h[1] for h in history]

    plt.figure()
    plt.plot(lambdas, ndcgs, marker="o")
    plt.xscale("log")
    plt.xlabel("lambda (reg, log scale)")
    plt.ylabel("NDCG@20 (validation)")
    plt.title("BPRMF NDCG@20 vs lambda")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved lambda vs NDCG plot to: {out_path}")


def plot_loss_curves(
    loss_history_per_combo: List[Tuple[Tuple[int, float], List[float]]],
    out_path: str = "../images/bpr_loss_k_lambda.png",
) -> None:
    """
    Plot training loss vs epoch for each (k, lambda) combination.
    """
    if not loss_history_per_combo:
        logger.warning("No loss histories to plot.")
        return

    plt.figure()
    for (k, lam), losses in loss_history_per_combo:
        epochs = list(range(1, len(losses) + 1))
        label = f"k={k}, λ={lam}"
        plt.plot(epochs, losses, marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Average train loss (-log sigma)")
    plt.title("BPRMF training loss vs epoch (k, lambda grid)")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved BPRMF loss curves plot to: {out_path}")

def plot_best_train_val_curve(
    train_loss: List[float],
    val_ndcg: List[Optional[float]],
    eval_every: int,
    out_path: str = "../images/bpr_best_train_val.png",
    k_eval: int = 20,
) -> None:
    """
    Plot training loss and validation NDCG@k on the same figure
    (two y-axes) for the best (k, lambda) model.
    """
    if not train_loss:
        logger.warning("No train loss history to plot.")
        return

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, train_loss, marker="o", label="Train loss (-log σ)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg train loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # validation points only where we actually computed NDCG
    val_epochs = [e for e, v in zip(epochs, val_ndcg) if v is not None]
    val_values = [v for v in val_ndcg if v is not None]

    ax2 = ax1.twinx()
    if val_epochs:
        ax2.plot(
            val_epochs,
            val_values,
            marker="s",
            linestyle="--",
            color="tab:orange",
            label=f"Val NDCG@{k_eval}",
        )
    ax2.set_ylabel(f"Validation NDCG@{k_eval}", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("BPRMF training loss and validation NDCG")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved best train+val curve to: {out_path}")

# =========================
# Main pipeline
# =========================

def run_pipeline() -> None:
    """
    Full BPRMF pipeline:
    - load data
    - k-core (user_min, item_min)
    - stratified warm eval split
    - optional heavy-user cap on train
    - grid search lambda
    - save plots for report
    """
    # ---- CONFIG ----
    train_path = "../train-1.txt"

    user_min = 1       # dataset already has multi-interaction users
    item_min = 3       # ensures evaluated items are learnable
    eval_ratio = 0.2
    warm_eval = True
    cap_per_user = 300  # 0 = no cap; 300 recommended for heavy users
    seed = 42

    # BPR hyperparams to keep fixed during lambda search
    # n_factors_grid = [96, 128, 160, 192, 224]
    n_factors_grid = [128]
    lr = 2e-3
    n_epochs = 30
    n_neg = 1
    use_idf_weight = True

    # lambda grid (L2 reg strength)
    # lambdas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3] [5e-5, 1e-4, 2e-4, 3e-4, 5e-4]
    lambdas = [1e-4]
    # -------------------------------------------------------------------

    logger.info(f"Loading data from: {train_path}")
    raw_df = load_train_txt_as_dataframe(train_path, user_col="user", item_col="item")
    logger.info(f"Raw interactions: {len(raw_df)}")

    logger.info(f"Applying k-core: user_min={user_min}, item_min={item_min}")
    df = kcore_filter(raw_df, "user", "item", user_min=user_min, item_min=item_min)
    logger.info(f"After k-core interactions: {len(df)}")

    logger.info(f"Stratified user split (eval_ratio={eval_ratio}, warm_eval={warm_eval})")
    train_df, eval_df = stratified_user_split(
        df,
        user_col="user",
        item_col="item",
        eval_ratio=eval_ratio,
        seed=seed,
        warm_eval=warm_eval,
    )
    logger.info(f"Train interactions: {len(train_df)}, Eval interactions: {len(eval_df)}")

    if cap_per_user > 0:
        logger.info(f"Capping heavy users in TRAIN to ≤{cap_per_user} interactions")
        train_df = cap_heavy_users(train_df, "user", "item", cap=cap_per_user, seed=seed)
        logger.info(f"After capping train interactions: {len(train_df)}")

    (
        best_k,
        best_lmbd,
        best_val,
        ndcg_history,
        loss_history_per_combo,
        best_loss_hist,
        best_val_hist,
    ) = grid_search_bpr(
        train_df,
        eval_df,
        user_col="user",
        item_col="item",
        n_factors_grid=n_factors_grid,
        lambdas=lambdas,
        lr=lr,
        n_epochs=n_epochs,
        n_neg=n_neg,
        use_idf_weight=use_idf_weight,
        seed=seed,
    )

    logger.info(
        f"[Result] Best n_factors={best_k}, lambda={best_lmbd} "
        f"with NDCG@20={best_val:.5f}"
    )

    # plots for report
    plot_ndcg_by_k_lambda(
        ndcg_history, out_path="../images/bpr_ndcg_k_lambda.png"
    )
    plot_loss_curves(
        loss_history_per_combo, out_path="../images/bpr_loss_k_lambda.png"
    )

    # train vs val curve for best model
    plot_best_train_val_curve(
        best_loss_hist,
        best_val_hist,
        eval_every=5,
        out_path="../images/bpr_best_train_val.png",
        k_eval=20,
    )

    logger.info("BPRMF tuning pipeline finished (no submission file written).")


if __name__ == "__main__":
    run_pipeline()
