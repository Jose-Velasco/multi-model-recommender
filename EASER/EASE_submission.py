import sys
import logging
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# ----------------- logging -----------------
logger = logging.getLogger("TorchEASE_submit")
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
    num_users, k = recs.shape
    with open(path, "w") as f:
        for u in range(num_users):
            items_str = " ".join(str(int(it)) for it in recs[u])
            f.write(items_str + "\n")


def load_train_txt_as_dataframe(
    path: str,
    user_col: str = "user",
    item_col: str = "item",
) -> pd.DataFrame:

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


# =========================
# EASE model (PyTorch, CPU)
# =========================

class TorchEASE:

    def __init__(
        self,
        df: pd.DataFrame,
        user_col: str = "user",
        item_col: str = "item",
        score_col: Optional[str] = None,
        regularization: float = 500.0,
        user_row_norm: bool = True,
        idf_weight: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.df = df.copy()
        self.user_col = user_col
        self.item_col = item_col
        self.score_col = score_col
        self.regularization = float(regularization)
        self.user_row_norm = bool(user_row_norm)
        self.idf_weight = bool(idf_weight)
        self.logger = logger or logging.getLogger("TorchEASE_submit")

        # CPU only for simplicity & stability
        self.device = torch.device("cpu")

        # Filled in during fit()
        self.user2idx: Dict[int, int] = {}
        self.idx2user: np.ndarray = np.array([], dtype=np.int64)
        self.item2idx: Dict[int, int] = {}
        self.idx2item: np.ndarray = np.array([], dtype=np.int64)
        self.n_users: int = 0
        self.n_items: int = 0

        # Model parameters
        self.B: Optional[torch.Tensor] = None              # item-item weight matrix (CPU)
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
        Fit the EASE model in PyTorch (CPU).

        Closed-form solution (Steck, 2019):
            G = X^T X + lambda * I
            P = G^{-1}
            B = -P / diag(P), with diag(B) = 0
        """
        self._build_mappings()

        df = self.df
        # Map external ids to internal contiguous indices
        u_idx = df[self.user_col].map(self.user2idx).values.astype(np.int64)
        i_idx = df[self.item_col].map(self.item2idx).values.astype(np.int64)

        # Base data (binary or from score_col)
        if self.score_col is not None and self.score_col in df.columns:
            base = df[self.score_col].astype(np.float32).values
        else:
            base = np.ones(len(df), dtype=np.float32)

        # Optional user-row normalization
        if self.user_row_norm:
            counts = df.groupby(self.user_col)[self.item_col].count().to_dict()
            u_weights = np.array(
                [1.0 / max(1.0, np.sqrt(float(counts[int(u_ext)])))
                 for u_ext in df[self.user_col].values],
                dtype=np.float32,
            )
        else:
            u_weights = np.ones_like(base, dtype=np.float32)

        # Optional IDF weight (by item df)
        if self.idf_weight:
            df_item = df.groupby(self.item_col)[self.user_col].nunique().to_dict()
            idf = {
                i_ext: np.log((1.0 + self.n_users) / (1.0 + df_item[i_ext]))
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
        popular_np = np.argsort(-item_counts)  # descending
        self.popular_items = torch.from_numpy(popular_np)  # CPU

        # Sparse user–item matrix X (CPU)
        self.logger.info("Building sparse user-item matrix (CPU)")
        indices_np = np.vstack([u_idx, i_idx])  # shape (2, nnz)
        indices = torch.from_numpy(indices_np).long()
        values = torch.from_numpy(data).float()

        X = torch.sparse_coo_tensor(
            indices,
            values,
            size=(self.n_users, self.n_items),
            dtype=torch.float32,
        ).coalesce()  # CPU sparse tensor

        # Compute Gram matrix G = X^T X (dense, CPU)
        self.logger.info("Computing Gram matrix G = X^T X (dense, CPU)")
        G = torch.sparse.mm(X.t(), X).to_dense()  # (n_items, n_items) on CPU

        # Add lambda * I on the diagonal
        self.logger.info("Adding regularization to the diagonal")
        diag_indices = torch.arange(self.n_items)
        G[diag_indices, diag_indices] += self.regularization

        # Invert G on CPU
        self.logger.info("Inverting Gram matrix on CPU (this is the heavy step)")
        P = torch.linalg.inv(G)

        # Compute B and zero out the diagonal
        self.logger.info("Computing item-item weight matrix B")
        diagP = torch.diag(P)  # shape (n_items,)
        B = -P / diagP.unsqueeze(0)
        B[diag_indices, diag_indices] = 0.0

        # Store model parameters
        self.B = B  # B is on CPU

        # Free large temporaries
        del P, G, X
        self.logger.info("Fit finished")

    # ----------------- recommend -----------------
    def _recommend_for_user_idx(
        self,
        u_idx: int,
        k: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.B is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        assert self.user_items is not None
        assert self.popular_items is not None

        seen_list = self.user_items[u_idx]
        if len(seen_list) == 0:
            # Cold-start: fall back to globally popular items
            top_internal = self.popular_items[:k]
            scores = torch.zeros_like(top_internal, dtype=torch.float32)
            return top_internal, scores

        seen_items = torch.tensor(seen_list, dtype=torch.long)

        # Score each item by summing corresponding rows of B
        # B[seen_items] has shape (num_seen, n_items)
        scores = self.B[seen_items].sum(dim=0)  # shape: (n_items,)

        # Do not recommend items the user has already interacted with
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
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        recs = np.zeros((self.n_users, k), dtype=np.int64)

        # idx2item as torch tensor for easy indexing
        idx2item_t = torch.from_numpy(self.idx2item)

        for u_idx in range(self.n_users):
            top_internal, _ = self._recommend_for_user_idx(u_idx, k=k)
            top_items_ext = idx2item_t[top_internal].numpy()
            num_write = min(k, len(top_items_ext))
            recs[u_idx, :num_write] = top_items_ext[:num_write]

        return recs


# =========================
# Main: train on full data, write submission
# =========================

def main() -> None:

    train_path = "../train-1.txt"
    out_path = "ease_submission.txt"

    # Hyperparameters
    regularization = 500.0
    user_row_norm = True
    idf_weight = True

    logger.info(f"Loading training data from: {train_path}")
    train_df = load_train_txt_as_dataframe(train_path, user_col="user", item_col="item")

    logger.info(
        f"Fitting TorchEASE on FULL data (lambda={regularization}, "
        f"user_row_norm={user_row_norm}, idf_weight={idf_weight})"
    )
    te = TorchEASE(
        train_df,
        user_col="user",
        item_col="item",
        score_col=None,
        regularization=regularization,
        user_row_norm=user_row_norm,
        idf_weight=idf_weight,
    )
    te.fit()

    logger.info("Generating top-20 recommendations for all users")
    recs = te.recommend_all_users(k=20)

    logger.info(f"Saving submission file (no user ids) to: {out_path}")
    save_recommendations(recs, out_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()

