"""
- Train on FULL train-1.txt with chosen hyperparams
- Write 20 items per user (no user IDs) to lightgcn_submission.txt
"""

import sys
import logging
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger("LightGCN_submit")
if not logger.handlers:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


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
                    continue
                seen.add(it_int)
                users.append(u)
                items.append(it_int)
    return pd.DataFrame({user_col: users, item_col: items})


class ImplicitCFData:
    def __init__(self, df: pd.DataFrame, user_col: str = "user", item_col: str = "item"):
        self.user_col = user_col
        self.item_col = item_col

        users = df[user_col].values
        items = df[item_col].values

        self.user2id: Dict[int, int] = {}
        self.id2user: List[int] = []
        self.item2id: Dict[int, int] = {}
        self.id2item: List[int] = []

        for u in users:
            if u not in self.user2id:
                self.user2id[u] = len(self.id2user)
                self.id2user.append(u)
        for i in items:
            if i not in self.item2id:
                self.item2id[i] = len(self.id2item)
                self.id2item.append(i)

        self.n_users = len(self.id2user)
        self.n_items = len(self.id2item)
        logger.info(f"LightGCN submit data: {self.n_users} users, {self.n_items} items")

        self.user_pos_items: List[List[int]] = [[] for _ in range(self.n_users)]
        for u_ext, i_ext in zip(users, items):
            u = self.user2id[u_ext]
            i = self.item2id[i_ext]
            self.user_pos_items[u].append(i)
        for u in range(self.n_users):
            self.user_pos_items[u] = list(sorted(set(self.user_pos_items[u])))

        self.A_norm = self._build_normalized_adj()

    def _build_normalized_adj(self) -> torch.sparse.FloatTensor:
        n_nodes = self.n_users + self.n_items
        rows: List[int] = []
        cols: List[int] = []

        for u in range(self.n_users):
            for i in self.user_pos_items[u]:
                ui = self.n_users + i
                rows.append(u)
                cols.append(ui)
                rows.append(ui)
                cols.append(u)

        rows = np.array(rows, dtype=np.int64)
        cols = np.array(cols, dtype=np.int64)
        data = np.ones_like(rows, dtype=np.float32)

        deg = np.bincount(rows, minlength=n_nodes).astype(np.float32)
        norm = 1.0 / np.sqrt(deg[rows] * deg[cols])
        values = data * norm

        indices = torch.from_numpy(np.vstack([rows, cols]))
        values_t = torch.from_numpy(values)
        return torch.sparse_coo_tensor(
            indices, values_t, size=(n_nodes, n_nodes), dtype=torch.float32
        ).coalesce()

    def sample_bpr_batch(
        self,
        batch_size: int,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        users = rng.randint(0, self.n_users, size=batch_size)
        pos_items = np.zeros(batch_size, dtype=np.int64)
        neg_items = np.zeros(batch_size, dtype=np.int64)

        for idx, u in enumerate(users):
            pos_list = self.user_pos_items[u]
            pos_items[idx] = pos_list[rng.randint(len(pos_list))]
            while True:
                j = rng.randint(0, self.n_items)
                if j not in pos_list:
                    neg_items[idx] = j
                    break
        return users, pos_items, neg_items


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, A_norm: torch.sparse.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(A_norm, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1).mean(dim=1)
        user_final, item_final = torch.split(embs, [self.n_users, self.n_items], dim=0)
        return user_final, item_final


def bpr_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    users: torch.Tensor,
    pos_items: torch.Tensor,
    neg_items: torch.Tensor,
    l2_reg: float,
) -> torch.Tensor:
    u_e = user_emb[users]
    pos_e = item_emb[pos_items]
    neg_e = item_emb[neg_items]
    pos_scores = torch.sum(u_e * pos_e, dim=1)
    neg_scores = torch.sum(u_e * neg_e, dim=1)
    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
    reg = (u_e.norm(2).pow(2) +
           pos_e.norm(2).pow(2) +
           neg_e.norm(2).pow(2)) / users.shape[0]
    return loss + l2_reg * reg


def save_recommendations(recs: np.ndarray, path: str) -> None:
    """
    recs: (num_users, k) with EXTERNAL item IDs in each row.
    Writes one line per user: item1 item2 ... item20
    """
    with open(path, "w") as f:
        for row in recs:
            f.write(" ".join(str(int(it)) for it in row) + "\n")


def main():
    train_path = "../train-1.txt"
    out_path = "lightgcn_submission.txt"

    # ========== best config from lightGCN.py ==========
    embed_dim = 64
    n_layers = 2
    lr = 1e-3
    l2_reg = 5e-4
    epochs = 40
    batch_size = 2048
    seed = 42
    # ===========================================================================

    logger.info(f"Loading training data from: {train_path}")
    df = load_train_txt_as_dataframe(train_path, user_col="user", item_col="item")

    data = ImplicitCFData(df, "user", "item")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    A_norm = data.A_norm.to(device)
    model = LightGCN(data.n_users, data.n_items, embed_dim=embed_dim, n_layers=n_layers).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    rng = np.random.RandomState(seed)

    n_interactions = sum(len(v) for v in data.user_pos_items)
    steps_per_epoch = max(1, n_interactions // batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            users, pos_items, neg_items = data.sample_bpr_batch(batch_size, rng)
            users_t = torch.from_numpy(users).long().to(device)
            pos_t = torch.from_numpy(pos_items).long().to(device)
            neg_t = torch.from_numpy(neg_items).long().to(device)

            user_emb, item_emb = model(A_norm)
            loss = bpr_loss(user_emb, item_emb, users_t, pos_t, neg_t, l2_reg)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= steps_per_epoch
        logger.info(f"Epoch {epoch}/{epochs} loss = {epoch_loss:.4f}")

    # Generate recommendations
    logger.info("Generating top-20 recommendations for all users")
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(A_norm)
        scores = torch.matmul(user_emb, item_emb.t())
        for u in range(data.n_users):
            pos_items = data.user_pos_items[u]
            if pos_items:
                scores[u, pos_items] = -1e9
        topk = torch.topk(scores, k=20, dim=1).indices.cpu().numpy()

    id2item = np.array(data.id2item, dtype=np.int64)
    recs = id2item[topk]  # (num_users, 20) external item IDs

    logger.info(f"Saving submission file to: {out_path}")
    save_recommendations(recs, out_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()