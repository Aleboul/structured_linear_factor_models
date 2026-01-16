from itertools import product
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import est_impure
import networkx as nx
from tqdm import tqdm
from sklearn.metrics import r2_score
from matplotlib.colors import ListedColormap

# Define custom color palette for visualizations
arcane_colors = ["#C71585", "#9D7FBA", "#324AB2"]
arcane_cmap = ListedColormap(arcane_colors, name='arcane')


def custom_matrix_product(matrix_a, matrix_b):
    """Compute matrix product using sum of element-wise minima."""
    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError("Matrix dimension mismatch: columns of first matrix must equal rows of second matrix.")

    # Reshape matrices for broadcasting and compute element-wise minimum
    a_reshaped = matrix_a[:, :, np.newaxis]
    b_reshaped = matrix_b[np.newaxis, :, :]
    minima = np.minimum(a_reshaped, b_reshaped)
    return np.sum(minima, axis=1)


def ext_cor_mat_optimized(X, n, k):
    """Compute extremal correlation matrix using rank thresholding."""
    datarank = np.array(pd.DataFrame(X).rank(axis=0, method='first'))
    indicator = datarank > n - k  # Identify top k ranks for each variable
    d = indicator.shape[1]
    test_sum = np.zeros((d, d), dtype=np.float64)

    # Accumulate outer products across all observations
    for row in indicator:
        test_sum += np.outer(row, row)

    return test_sum / k


def compute_crit(A_hat, Theta):
    """Calculate R² between upper triangular parts of estimated and empirical chi matrices."""
    Chi_A = custom_matrix_product(A_hat, A_hat.T)
    iu = np.triu_indices_from(Theta, k=1)
    return r2_score(Theta[iu], Chi_A[iu])


def calibrate(Theta, kappa_1_list, kappa_2_list):
    """Perform grid search to find optimal kappa parameters using graph cliques."""
    value_crit = -10e6
    tuned_kappa_1 = tuned_kappa_2 = None
    tuned_pure = []

    for kappa_1 in tqdm(kappa_1_list, desc="Calibrating kappa_1"):
        # Construct adjacency graph based on threshold
        adjacency_matrix = (Theta < kappa_1).astype(float)
        G = nx.from_numpy_array(adjacency_matrix)
        cliques = list(nx.find_cliques(G))
        clique_max = max(cliques, key=len)

        for kappa_2 in tqdm(kappa_2_list, 
                          desc=f"Testing kappa_2 with kappa_1={kappa_1:.4f}", 
                          leave=False):
            # Test only maximal cliques
            for clique in cliques:
                if len(clique) == len(clique_max):
                    A_hat, pure = est_impure.est_A(Theta, kappa_1, kappa_2, clique)
                    crit = compute_crit(A_hat, Theta)

                    if crit > value_crit:
                        value_crit = crit
                        tuned_kappa_1, tuned_kappa_2, tuned_pure = kappa_1, kappa_2, pure
                        tqdm.write(f"New optimum: R²={crit:.4f}, "
                                  f"kappa_1={kappa_1:.4f}, kappa_2={kappa_2:.4f}")

    return tuned_kappa_1, tuned_kappa_2, tuned_pure


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
data = pd.read_csv("data/nhanes_dr1tot.csv").dropna()
X = np.array(data)
n, d = X.shape
r = 1  # Regularization parameter for threshold calculation

# Calculate threshold parameter k based on theoretical considerations
k = int(0.25 * np.power(np.log(4 * d * n**2), 1/(2*r+1)) 
        * np.power(n, 2*r/(2*r+1)))

print(f"Dataset dimensions: {n} observations, {d} variables")
print(f"Threshold parameter k: {k}")

# Compute extremal correlation matrix
Chi = ext_cor_mat_optimized(X, n, k)

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
plt.matshow(Chi, cmap='coolwarm', vmin=0, vmax=1, fignum=1)
plt.colorbar()
plt.title("Extremal Correlation Matrix (Chi)")
plt.show()

# =============================================================================
# Parameter Grid Construction
# =============================================================================
num_kappa = 100
constants_kappa = np.linspace(0.25, 1.5, num_kappa)
scaling_factor = np.power(np.log(4 * d * n**2) / n, r / (2*r + 1))
kappa_1_list = [c * scaling_factor for c in constants_kappa]
kappa_2_list = [c * scaling_factor for c in constants_kappa]

# =============================================================================
# Model Calibration
# =============================================================================
tuned_kappa_1, tuned_kappa_2, tuned_pure = calibrate(Chi, kappa_1_list, kappa_2_list)

print(f"\nOptimal kappa_1: {tuned_kappa_1}")
print(f"Optimal kappa_2: {tuned_kappa_2}")
print(f"Identified pure variables: {tuned_pure}")

# Estimate factor loading matrix using optimal parameters
A_hat, _ = est_impure.est_A(Chi, tuned_kappa_1, tuned_kappa_2, clique_max=tuned_pure)

# Ensure results directory exists
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# =============================================================================
# Model Evaluation
# =============================================================================
Chi_fitted = custom_matrix_product(A_hat, A_hat.T)
rows, cols = np.triu_indices(Chi_fitted.shape[0], k=1)
x_values, y_values = Chi_fitted[rows, cols], Chi[rows, cols]
r2 = r2_score(y_values, x_values)

print(f"\nModel performance:")
print(f"  R² between fitted and empirical correlations: {r2:.4f}")
print(f"  Pure variable names: {list(data.columns[tuned_pure])}")

# Create scatter plot comparing fitted vs observed correlations
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(x_values, y_values, alpha=0.6, edgecolor='k', color='#324AB2')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, color='#C71585')
ax.text(0.05, 0.95, f'$R^2$ = {r2:.3f}', transform=ax.transAxes,
        fontsize=18, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
ax.set_xlabel(r'$\tilde{\mathcal{X}}^{\kappa^{*}, \bar{\kappa}^{*}}$', fontsize=25)
ax.set_ylabel(r'$\hat{\mathcal{X}}$', fontsize=25)
ax.tick_params(labelsize=18)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'Chi_fitted_vs_Chi_scatter.pdf'))
plt.close()

# =============================================================================
# Factor Analysis Results
# =============================================================================
K_hat = A_hat.shape[1]
cluster_data = []

# Extract factor assignments and weights for each variable
for factor_idx in range(K_hat):
    support = np.where(A_hat[:, factor_idx] > 0)[0]
    weights = A_hat[support, factor_idx]

    for idx, weight in zip(support, weights):
        cluster_data.append({
            'Index': idx,
            'Feature': data.columns[idx],
            'Factor': factor_idx + 1,
            'Weight': weight,
            'Is Pure': 'Yes' if weight == 1 else 'No'
        })

# Save factor assignments to CSV
cluster_df = pd.DataFrame(cluster_data)
cluster_df = cluster_df[['Index', 'Feature', 'Factor', 'Weight', 'Is Pure']]
cluster_df.to_csv('data/cluster_df.csv', index=False)

# Create heatmap visualization of factor weights
pivot_df = cluster_df.pivot(index='Feature', columns='Factor', values='Weight').fillna(0)
sorted_features = pivot_df.max(axis=1).sort_values(ascending=False).index
pivot_df = pivot_df.loc[sorted_features]

plt.figure(figsize=(7, 6))
ax = sns.heatmap(pivot_df, cmap='YlGnBu', linewidths=0.5, 
                 cbar_kws={'label': 'Factor Weight'})
ax.set_xlabel('Factor', fontsize=20)
ax.set_ylabel('Variables', fontsize=20)
ax.tick_params(labelsize=18)

# Properly align y-axis tick labels
ax.set_yticks(np.arange(len(pivot_df)) + 0.5)
ax.set_yticklabels(pivot_df.index)

# Format colorbar
cbar = ax.collections[0].colorbar
cbar.set_label('Factor Weight', fontsize=20)
cbar.ax.tick_params(labelsize=18)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'Feature_Assignments_Heatmap.pdf'))
plt.close()