from collections import Counter, defaultdict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "train-1.txt"
img_dir = "images"
os.makedirs(img_dir, exist_ok=True)

user_items = {}
item_cnt = Counter()

with open(path, "r") as f:
    for line in f:
        toks = line.strip().split()
        if not toks:
            continue
        u, items = toks[0], list(dict.fromkeys(toks[1:]))  # dedupe per user, keep order
        if not items:
            continue
        user_items[u] = items
        item_cnt.update(items)

# Print top-10 most frequent items
print("Top 5 most popular items:")
for item, cnt in item_cnt.most_common(5):
    print(f"Item {item}: {cnt} interactions")
# Item 791: 1258 interactions
# Item 794: 1147 interactions
# Item 3253: 1061 interactions
# Item 2910: 1037 interactions
# Item 2022: 1014 interactions

# Build histogram of item frequencies
freqs = list(item_cnt.values())

plt.figure(figsize=(8, 5))
plt.hist(freqs, bins=50, log=True, color='steelblue', edgecolor='black')
plt.xlabel("Number of interactions per item")
plt.ylabel("Number of items (log scale)")
plt.title("Item popularity histogram")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "item_popularity_histogram.png"))
plt.show()

# 1) Basic stats
num_users = len(user_items)
all_items = set(i for it in user_items.values() for i in it)
num_items = len(all_items)
num_interactions = sum(len(v) for v in user_items.values())
density = num_interactions / (num_users * num_items)
print(f"Number of users: {num_users}")  # 31668
print(f"Number of items: {num_items}")  # 38048
print(f"Number of interactions: {num_interactions}")  # 1237259
print(f"Matrix density: {density:.6f}")  # 0.00103

# 2) User interaction histogram
hist = Counter(len(v) for v in user_items.values())
# Convert histogram to sorted lists
x = sorted(hist.keys())   # number of interactions per user
y = [hist[k] for k in x]  # number of users with that many interactions
print("Lowest interaction levels (x[:3]):")
for xi in x[:3]:
    print(f"{xi} interactions → {hist[xi]} users")

print("\nHighest interaction levels (x[-5:]):")
for xi in x[-5:]:
    print(f"{xi} interactions → {hist[xi]} users")

plt.figure(figsize=(8, 5))
plt.bar(x, y, color='steelblue', edgecolor='steelblue')
plt.xlabel("Number of interactions per user")
plt.ylabel("Number of users")
plt.title("User interaction histogram")
plt.yscale("log")  # for long-tail distributions
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "user_interaction_histogram.png"))
plt.show()


# 3) User interaction quantiles via boxplots

user_interactions = np.array([len(v) for v in user_items.values()], dtype=np.int64)

# Heavy user stats (e.g., >300 interactions)
threshold = 300
heavy_mask = user_interactions > threshold
frac_users_heavy = heavy_mask.mean()
frac_interactions_heavy = user_interactions[heavy_mask].sum() / user_interactions.sum()
print(f"\nFraction of users > {threshold}: {frac_users_heavy:.6f}")
print(f"Fraction of interactions from users > {threshold}: {frac_interactions_heavy:.6f}")

# --- Full boxplot (no annotations) ---
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.boxplot(
    user_interactions,
    vert=False,
    showfliers=True,
    whis=[5, 95],
)
ax.set_xlabel("Number of interactions per user")
ax.set_yticks([])
ax.set_title("User interaction distribution (boxplot)")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "user_interaction_boxplot_full.png"))
plt.show()

# --- Zoomed, annotated boxplot (≤ 300 interactions) ---

# Key quantiles
q5, q25, q50, q75, q95 = np.percentile(user_interactions, [5, 25, 50, 75, 95])

fig, ax = plt.subplots(figsize=(8, 2.8))
ax.boxplot(
    user_interactions,
    vert=False,
    showfliers=False,   # hide outliers so the box is clear
    whis=[5, 95],
)
ax.set_xlim(0, 200)
ax.set_xlabel("Number of interactions per user (capped at 300)")
ax.set_yticks([])
ax.set_title("User interaction distribution (zoomed, ≤300 interactions)")

# Annotate main quantiles on the zoomed box
for val, label in [
    (q25, "Q1 (25%)"),
    (q50, "Median (50%)"),
    (q75, "Q3 (75%)"),
]:
    if val <= 200:  # only annotate if visible in zoom
        ax.axvline(val, color="grey", linestyle="--", alpha=0.6)
        ax.text(
            val,
            1.02,  # slightly above the box
            f"{label}\n{val:.0f}",
            rotation=90,
            va="bottom",
            ha="center",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
        )

# Optional: annotate whiskers (5% and 95%) if inside zoom
# if q5 <= 300:
#     ax.text(q5, 0.2, f"5%: {q5:.0f}", ha="center", va="bottom", fontsize=7)
# if q95 <= 300:
#     ax.text(q95, 0.2, f"95%: {q95:.0f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, "user_interaction_boxplot_zoom300_annotated.png"))
plt.show()

#count 31668
# mean 39
# std 45
# min 16
# 25% 19
# 50% 25
# 75% 41
# max 1848
# Fraction of users > 300: 0.004420866489832007

# 4) Item frequency quantiles and boxplots (analogous to user interactions)

item_freqs = np.array(freqs, dtype=np.int64)

# Basic stats for items
print("\nItem frequency stats:")
print(f"count  {item_freqs.size}")
print(f"mean   {item_freqs.mean():.2f}")
print(f"std    {item_freqs.std():.2f}")
print(f"min    {item_freqs.min()}")
print(f"25%    {np.percentile(item_freqs, 25):.0f}")
print(f"50%    {np.percentile(item_freqs, 50):.0f}")
print(f"75%    {np.percentile(item_freqs, 75):.0f}")
print(f"max    {item_freqs.max()}")

# Heavy item stats (e.g., > 300 interactions)
item_threshold = 300
heavy_item_mask = item_freqs > item_threshold
frac_items_heavy = heavy_item_mask.mean()
frac_interactions_in_heavy_items = item_freqs[heavy_item_mask].sum() / item_freqs.sum()
print(f"\nFraction of items > {item_threshold} interactions: {frac_items_heavy:.6f}")
print(
    "Fraction of all interactions on items > "
    f"{item_threshold}: {frac_interactions_in_heavy_items:.6f}"
)

# --- Full boxplot for item frequencies ---
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.boxplot(
    item_freqs,
    vert=False,
    showfliers=True,
    whis=[5, 95],
)
ax.set_xlabel("Number of interactions per item")
ax.set_yticks([])
ax.set_title("Item frequency distribution (boxplot)")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "item_frequency_boxplot_full.png"))
plt.show()

# --- Zoomed, annotated boxplot for items (≤ 300 interactions) ---

# Key quantiles
iq5, iq25, iq50, iq75, iq95 = np.percentile(item_freqs, [5, 25, 50, 75, 95])

fig, ax = plt.subplots(figsize=(8, 2.8))
ax.boxplot(
    item_freqs,
    vert=False,
    showfliers=False,   # hide outliers so the box is clear
    whis=[5, 95],
)
ax.set_xlim(0, 200)
ax.set_xlabel("Number of interactions per item (capped at 300)")
ax.set_yticks([])
ax.set_title("Item frequency distribution (zoomed, ≤300 interactions)")

# Annotate main quantiles on the zoomed box
for val, label in [
    (iq25, "Q1 (25%)"),
    (iq50, "Median (50%)"),
    (iq75, "Q3 (75%)"),
]:
    if val <= 200:  # only annotate if visible in zoom
        ax.axvline(val, color="grey", linestyle="--", alpha=0.6)
        ax.text(
            val,
            1.02,  # slightly above the box
            f"{label}\n{val:.0f}",
            rotation=90,
            va="bottom",
            ha="center",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
        )

# # Optional: annotate whiskers (5% and 95%) if inside zoom
# if iq5 <= 300:
#     ax.text(iq5, 0.2, f"5%: {iq5:.0f}", ha="center", va="bottom", fontsize=7)
# if iq95 <= 300:
#     ax.text(iq95, 0.2, f"95%: {iq95:.0f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, "item_frequency_boxplot_zoom300_annotated.png"))
plt.show()

# #Item frequency stats:
# count  38048
# mean   32.52
# std    49.27
# min    1
# 25%    11
# 50%    17
# 75%    34
# max    1258
