# File format: each line: user_id item1 item2 item3 ...
from collections import Counter, defaultdict
import random
import matplotlib.pyplot as plt

path = "train-1.txt"

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

plt.figure(figsize=(8,5))
plt.hist(freqs, bins=50, log=True, color='steelblue', edgecolor='black')
plt.xlabel("Number of interactions per item")
plt.ylabel("Number of items (log scale)")
plt.title("Item popularity histogram")
plt.tight_layout()
plt.savefig("images/item_popularity_histogram.png")
plt.show()


# 1) Basic stats
num_users = len(user_items)
all_items = set(i for it in user_items.values() for i in it)
num_items = len(all_items)
num_interactions = sum(len(v) for v in user_items.values())
density = num_interactions / (num_users * num_items)
print(f"Number of users: {num_users}") #31668
print(f"Number of items: {num_items}") # 38048
print(f"Number of interactions: {num_interactions}") # 1237259
print(f"Matrix density: {density:.6f}") # 0.00103

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
# Plot histogram
plt.figure(figsize=(8,5))
plt.bar(x, y, color='steelblue', edgecolor='steelblue')
plt.xlabel("Number of interactions per user")
plt.ylabel("Number of users")
plt.title("User interaction histogram")
plt.yscale("log")  # for long-tail distributions
plt.tight_layout()
plt.savefig("images/user_interaction_histogram.png")
plt.show()
