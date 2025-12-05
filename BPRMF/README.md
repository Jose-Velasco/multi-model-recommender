# BPR-MF Baseline (Bayesian Personalized Ranking – Matrix Factorization)

This folder contains code and plots for the MF-BPR baseline used in our CMPE 256 project.
The implementation is a NumPy-based BPR-MF model with warm train/eval splits and NDCG@20
evaluation. It is not part of the main slide deck (only Appendix A of the report),
because it did not yield competitive NDCG@20 and introduces no new modeling ideas beyond
standard MF-BPR.

## 1. Folder structure

Repository layout (relevant parts):
```
project-root/
├─ train-1.txt # implicit feedback dataset (user item1 item2 ...)
├─ images/ # all plots are saved here
│ ├─ bpr_best_train_val.png
│ ├─ bpr_loss_k_lambda.png
│ ├─ bpr_ndcg_k_lambda.png
│ ├─ svd_cumulative_energy.png
│ ├─ svd_singular_values.png
│ └─ ...
├─ BPRMF/
│ ├─ BPRMF.py # MF-BPR training + grid search + plots
│ ├─ SVD.py # SVD-based latent-dimension analysis
│ └─ README.md # this file
└─ requirements.txt
```
All paths inside the BPRMF scripts assume this layout:
- Dataset is at `../train-1.txt`
- Plots are written to `../images/`

---

## 2. Environment (Python 3.11)

Create and activate a virtual environment with Python 3.11, then install dependencies
from the main requirements.txt:

```bash
# from the project root
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Scripts
### 3.1 BPRMF.py – MF-BPR training + grid search + plots
This is a simple MF-BPR implementation used as a weak baseline and a sanity check
for our preprocessing and evaluation pipeline.

**What it does**

Parse implicit interactions

Expands `../train-1.txt` into `(user, item)` pairs, deduplicated per user.

**Preprocessing**

Item k-core filter: keep items with at least 3 users (`item_min = 3`).

Per-user warm stratified split: ~80% of each user’s items go to train and 20% to
eval (`eval_ratio = 0.2`), ensuring eval items are “warm” (appear for at least
2 users and at least once in training).

Heavy-user cap: limit each user to at most 300 training interactions
(`cap_per_user = 300`) so very active users do not dominate the updates.

Optional IDF-based weighting when sampling positives to slightly down-weight
extremely popular items.

**Model**

Represents each user and item with a length-`k` NumPy embedding vector.
Scores are inner products: `s(u, i) = p_u^T q_i`.

**Training and grid search**

Uses the standard BPR loss with uniform negative sampling:
for each positive `(u, i)` sample one negative `j` such that user `u` has not
interacted with j.

Runs a grid search over:

`n_factors_grid` = `[96, 128, 160, 192, 224]`

Regularization strengths such as `[1e-5, 5e-5, 1e-4, 5e-4, 1e-3]`
and refined grids around `1e-4`.

Logs average training loss per epoch and computes validation NDCG@20 every
5 epochs.

**Outputs**

Best-combination train + validation curves:

`../images/bpr_best_train_val.png`

Training loss vs epoch for a given `(k, λ)`:

`../images/bpr_loss_k_lambda.png`

NDCG@20 vs `λ` for different `n_factors`:

`../images/bpr_ndcg_k_lambda.png`

Console logs summarizing the grid search and best `(k, λ)`.