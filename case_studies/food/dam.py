import numpy as np
from sklearn.preprocessing import QuantileTransformer
import MLExtreme as mlx
import pandas as pd

# ----------------------
# Step 1: Load data
# ----------------------
csv_filename = "data/nhanes_dr1tot.csv"

data = pd.read_csv(csv_filename)
data = data.dropna()

X = np.array(data)
d = X.shape[1]
n = X.shape[0]

print(X)
print(n, d)

# ----------------------
# Étape 2 : Transformation en rangs
# ----------------------
X_ranked = mlx.rank_transform(X)
norm_Xt = np.linalg.norm(X_ranked, axis=1, ord=2)
# ----------------------
# Étape 3 : Threshold Exceedance
# ----------------------
r=1
k = int(0.25 * np.power(np.log(4 * d * n**2), 1/(2*r+1))
        * np.power(n, 2*r/(2*r+1)))  # 0.075
quantile_threshold = 1-k/n
quantile_threshold = np.quantile(norm_Xt, quantile_threshold)
extreme_mask = norm_Xt > quantile_threshold
X_extreme = X_ranked[extreme_mask]

# ----------------------
# Step 4: Apply DAMEX
# ----------------------
damex = mlx.damex(epsilon=0.4)
damex.fit(X_extreme)

# ----------------------
# Step 5: Keep only maximal extremal directions
# ----------------------

# Convert DAMEX subfaces to sets
subfaces_sets = [set(cluster) for cluster in damex.subfaces]

def keep_maximal_sets(sets):
    """
    Keep only maximal sets under inclusion.
    If s ⊂ t, keep t only.
    """
    maximal_sets = []
    for s in sets:
        if not any(s < other for other in sets):  # strict inclusion
            maximal_sets.append(s)
    return maximal_sets

# Keep only largest extremal directions
maximal_subfaces = keep_maximal_sets(subfaces_sets)

# Remove duplicates if any
maximal_subfaces = list({frozenset(s) for s in maximal_subfaces})
maximal_subfaces = [set(s) for s in maximal_subfaces]

# ----------------------
# Step 6: Display results
# ----------------------
print("Maximal extremal directions (non-redundant):\n")

for i, cluster in enumerate(maximal_subfaces):
    support_names = data.columns[list(cluster)]
    print(f"Cluster {i+1}:")
    print(support_names)
    print()
