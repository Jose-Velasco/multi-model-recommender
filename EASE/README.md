# EASE Baseline (Embarrassingly Shallow AutoEncoder)

This folder contains all code and logs for the EASE baseline used in our CMPE 256 project.  
The implementation is based on the original EASE paper (Steck, WWW 2019) and the TorchEASE
open-source implementation (Jay Franck, GitHub), with our own preprocessing, stratified
splits, NDCG@20 evaluation, and λ grid search.

## 1. Folder structure
Repository layout:
```
project-root/
├─ train-1.txt # implicit feedback dataset (user item1 item2 ...)
├─ data_analysis.py # dataset statistics and plots
├─ images/ # all plots are saved here
│ ├─ ease_ndcg_vs_lambda.png
│ ├─ ease_idf_ablation.png
│ ├─ user_interaction_histogram.png
│ ├─ item_popularity_histogram.png
│ └─ ...
├─ EASE/
│ ├─ EASE.py # tuning pipeline (λ grid search + plots)
│ ├─ EASE_submission.py # final model training + leaderboard submission
│ ├─ EASE_vanilla.py # SciPy/NumPy reference (not practical on full data)
│ ├─ hyperparams.txt # logged NDCG@20 for λ and IDF ablation
│ └─ README.md # this file
└─ requirements.txt
```

All paths inside the EASE scripts assume this layout:
- Dataset is at `../train-1.txt`
- Plots are written to `../images/`
- Submission file is written next to `EASE_submission.py` as `ease_submission.txt`

If you change the layout, update the hard-coded paths in the scripts accordingly.

---

## 2. Environment (Python 3.11)

Create and activate a virtual environment with **Python 3.11**, then install dependencies
from the main `requirements.txt`:

```bash
# from the project root
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 3. Scripts
### 3.1 EASE.py – tuning pipeline (λ grid search + plots)

This is the main PyTorch EASE pipeline used for analysis and model selection.

**What it does**

Parse implicit interactions
Expands `train-1.txt` into `(user, item)` pairs, deduplicated per user.

**Preprocessing**

Item k-core filter: keep items with at least 3 users (`item_min = 3`).

Per-user warm stratified split: ~80% of each user’s items go to train and 20% to eval
(`eval_ratio = 0.2`), ensuring eval items are “warm” (appear for at least 2 users and
at least once in training).

Optional heavy-user cap (0 for EASE).

**Model**

Builds a sparse PyTorch COO user–item matrix X, computes the dense Gram matrix
`G = XᵀX + λI` in float32 on CPU, inverts it with `torch.linalg.inv`, and constructs
the item–item weight matrix B with zero diagonal.

**Grid search over λ**

Trains EASE for a list of λ values and evaluates NDCG@20 on the validation split.

**Outputs**

Writes (λ, NDCG@20) for each run to hyperparams.txt.
Saves a plot of λ vs NDCG@20 to `../images/ease_ndcg_vs_lambda.png`.
Saves an IDF ablation bar chart to `../images/ease_idf_ablation.png`
(NDCG@20 with idf_weight=False vs True).

```bash
cd EASE
python EASE.py
```
This script is for analysis only: it does not create a leaderboard submission file.

### 3.2 EASE_submission.py – final model + submission file

This script trains a single EASE model on the full dataset with the chosen
hyperparameters and produces the submission file expected by the leaderboard
(one line per user, 20 item IDs, no user IDs).

**What it does**

Loads `../train-1.txt`.

Applies the same k-core and preprocessing configuration used for the best model
(row-normalization, IDF weighting, final λ value).

Trains EASE once on all (train + eval) interactions.

Generates top-20 recommendations per user, masking out already seen items.

Writes ease_submission.txt in this folder with:
`item1 item2 ... item20`
for each user in the same order as `train-1.txt`.

**How to run**
```bash
cd EASE
python EASE_submission.py
```

### 3.3 EASE_vanilla.py – SciPy/NumPy reference (NOT recommended to run)

This file contains a “vanilla” EASE implementation using SciPy CSR matrices and
`numpy.linalg.inv`. On this dataset (≈ 38K items), the dense Gram matrix is
38,048 × 38,048, which:

Requires around 5 GB of RAM even in float32, and much more in float64.

Is often ill-conditioned, leading to numerical warnings (invalid value encountered in divide) and nearly identical recommendation lists.

We keep this script for reference and comparison only; in practice, it is not
used in our experiments, and the PyTorch version (`EASE.p`y / `EASE_submission.py`)
should be preferred.

### 3.4 hyperparams.txt – λ and IDF results

This text file logs the hyperparameter search results from `EASE.py`, including:
Each tested `λ` and its validation NDCG@20.
The IDF ablation results (`idf_weight = False vs True`).
These numbers are used to create:
`../images/ease_ndcg_vs_lambda.png`
`../images/ease_idf_ablation.`png`

You can regenerate the file and plots by re-running:
```bash
cd EASE
python EASE.py
```

---

## 4. Data analysis script (in project root)

The dataset-level analysis is handled by data_analysis.py in the project root (not
inside this folder).

**What it does**

Loads `train-1.txt`.

Computes user and item interaction statistics (min/median/max, quantiles, etc.).

Produces histograms and zoomed-in plots (e.g., user interaction histogram, item
popularity histogram, item frequency quantiles).

Saves all figures into `images/`, e.g.:

`user_interaction_histogram.png`
`item_popularity_histogram.png`
`item_freq_quantiles.png`

**How to run**
```bash
cd path/to/project-root
python data_analysis.py
```



