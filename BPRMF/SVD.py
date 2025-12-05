"""
svd_rank_estimate.py

Use truncated SVD on the user–item interaction matrix from train-1.txt
to estimate a reasonable range of n_factors for MF / BPRMF.

- Loads: ./train-1.txt
- Builds: sparse (n_users x n_items) matrix with 0/1 entries
- Computes top-K singular values with scipy.sparse.linalg.svds
- Prints k for 80%, 90%, 95%, 99% energy
- Saves:
    ./images/svd_singular_values.png
    ./images/svd_cumulative_energy.png
"""
# Portions of this file (initial skeleton and some functions) were generated with assistance from OpenAI's ChatGPT (GPT-5.1 Thinking) and then modified and debugged by the authors.

import os
import sys
import math
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

logger = logging.getLogger("SVD_rank")
if not logger.handlers:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# ========= Utilities =========

def load_train_txt_as_dataframe(
    path: str,
    user_col: str = "user",
    item_col: str = "item",
) -> pd.DataFrame:
    """
    Each line: user_id item_1 item_2 ... item_n
    -> one row per (user, item) pair, deduped per user.
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
                    continue
                seen.add(it_int)
                users.append(u)
                items.append(it_int)

    df = pd.DataFrame({user_col: users, item_col: items})
    return df


def build_sparse_matrix(
    df: pd.DataFrame,
    user_col: str = "user",
    item_col: str = "item",
) -> Tuple[coo_matrix, np.ndarray, np.ndarray]:
    """
    Build a sparse user–item matrix X (0/1) from df.
    Uses pandas.factorize to map external ids to [0..n_users-1], [0..n_items-1].
    """
    # factorize users and items -> internal indices
    user_codes, user_index = pd.factorize(df[user_col])
    item_codes, item_index = pd.factorize(df[item_col])

    n_users = len(user_index)
    n_items = len(item_index)

    logger.info(f"Users: {n_users}, Items: {n_items}, Interactions: {len(df)}")

    data = np.ones(len(df), dtype=np.float32)
    X = coo_matrix((data, (user_codes, item_codes)),
                   shape=(n_users, n_items),
                   dtype=np.float32)
    return X, user_index.to_numpy(), item_index.to_numpy()


def run_truncated_svd(
    X: coo_matrix,
    k_max: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute top-k_max singular values for sparse matrix X using svds.
    Returns singular values sorted descending.
    """
    n_users, n_items = X.shape
    k_max = int(k_max)
    k_max = max(1, k_max)
    # svds needs k < min(m, n)
    k_max = min(k_max, min(n_users, n_items) - 1)
    if k_max <= 0:
        raise ValueError("k_max too small for SVD (matrix too small).")

    logger.info(f"Running truncated SVD with k={k_max} on shape {X.shape}")

    # svds returns singular values ascending by default
    u, s, vt = svds(X, k=k_max, which="LM", return_singular_vectors=True,
                    v0=None, tol=0.0, maxiter=None)
    s_sorted = np.sort(s)[::-1]  # descending
    return s_sorted


def summarize_energy(svals: np.ndarray) -> List[Tuple[float, int]]:
    """
    Given singular values, compute k needed to reach several energy thresholds.
    Returns list of (threshold, k).
    """
    energy = svals ** 2
    total = energy.sum()
    cum = np.cumsum(energy)
    frac = cum / total

    thresholds = [0.8, 0.9, 0.95, 0.99]
    result: List[Tuple[float, int]] = []
    for t in thresholds:
        idx = int(np.searchsorted(frac, t))
        # idx is 0-based; k = idx+1 components
        k = min(idx + 1, len(svals))
        result.append((t, k))
    return result


def plot_spectrum(
    svals: np.ndarray,
    out_dir: str = "./images",
) -> None:
    """
    Save spectral plots:
      - singular values (log scale on y)
      - cumulative energy
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) singular values
    plt.figure()
    x = np.arange(1, len(svals) + 1)
    plt.plot(x, svals, marker="o")
    plt.yscale("log")
    plt.xlabel("Component index (k)")
    plt.ylabel("Singular value (log scale)")
    plt.title("Top singular values of user–item matrix")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path1 = os.path.join(out_dir, "svd_singular_values.png")
    plt.savefig(path1)
    plt.close()
    logger.info(f"Saved singular values plot to: {path1}")

    # 2) cumulative energy
    energy = svals ** 2
    cum = np.cumsum(energy)
    frac = cum / energy.sum()

    plt.figure()
    plt.plot(x, frac, marker="o")
    plt.xlabel("Number of components (k)")
    plt.ylabel("Cumulative energy (fraction)")
    plt.title("Cumulative explained energy vs number of components")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path2 = os.path.join(out_dir, "svd_cumulative_energy.png")
    plt.savefig(path2)
    plt.close()
    logger.info(f"Saved cumulative energy plot to: {path2}")


# ========= Main =========

def main() -> None:
    # ---- CONFIG (no external config file) ----
    train_path = "../train-1.txt"
    # Max components to keep from SVD.
    K_MAX = 256
    seed = 42
    # -----------------------------------------

    logger.info(f"Loading data from {train_path}")
    df = load_train_txt_as_dataframe(train_path, user_col="user", item_col="item")

    X, users, items = build_sparse_matrix(df, user_col="user", item_col="item")

    # For very large matrices, reduce K_MAX:
    n_users, n_items = X.shape
    eff_k = min(K_MAX, min(n_users, n_items) - 1)
    if eff_k < K_MAX:
        logger.info(f"Adjusted K_MAX from {K_MAX} to {eff_k} based on matrix shape.")
    K = eff_k

    svals = run_truncated_svd(X, k_max=K, seed=seed)

    logger.info("Top 10 singular values (descending):")
    for i, sv in enumerate(svals[:10], start=1):
        logger.info(f"  s[{i}] = {sv:.6f}")

    energy_summaries = summarize_energy(svals)
    logger.info("Estimated k to reach different energy thresholds:")
    for t, k in energy_summaries:
        logger.info(f"  {int(t*100)}% energy -> k ≈ {k}")

    # Make plots for report
    plot_spectrum(svals, out_dir="../images")

    # Print a simple suggestion for BPRMF search range
    # e.g., around the 90% threshold
    t90, k90 = energy_summaries[1]  # second element is 90%
    k_low = max(8, k90 // 2)
    k_high = min(len(svals), int(k90 * 1.5))
    logger.info(
        f"For BPRMF, you can try n_factors in a range like "
        f"[{k_low}, {k90}, {k_high}] (centered at 90% energy)."
    )


if __name__ == "__main__":
    main()