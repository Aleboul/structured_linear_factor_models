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

def extract_pairs(matrix, pure_indices, impure_indices):
    """
    Extract pure-pure, pure-impure, and impure-impure pairs from a matrix (upper diagonal only).
    
    Parameters:
    -----------
    matrix : ndarray, shape (d, d)
        Square matrix
    pure_indices : ndarray
        Indices of pure variables
    impure_indices : ndarray
        Indices of impure variables
        
    Returns:
    --------
    dict: Dictionary with pairs for each type
    """
    d = matrix.shape[0]
    
    # Create boolean masks for fast access
    pure_mask = np.zeros(d, dtype=bool)
    pure_mask[pure_indices] = True
    impure_mask = np.zeros(d, dtype=bool)
    impure_mask[impure_indices] = True
    
    # Get all upper diagonal indices
    rows, cols = np.triu_indices(d, k=1)
    
    # Determine pair types
    row_is_pure = pure_mask[rows]
    col_is_pure = pure_mask[cols]
    row_is_impure = impure_mask[rows]
    col_is_impure = impure_mask[cols]
    
    # Create masks for each pair type
    pure_pure_mask = row_is_pure & col_is_pure
    pure_impure_mask = (row_is_pure & col_is_impure) | (row_is_impure & col_is_pure)
    impure_impure_mask = row_is_impure & col_is_impure
    
    # Extract values
    all_values = matrix[rows, cols]
    
    return {
        'pure_pure': all_values[pure_pure_mask],
        'pure_impure': all_values[pure_impure_mask],
        'impure_impure': all_values[impure_impure_mask],
        'all': all_values
    }



# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
csv_filename = "data/hourly_ff_schleswig_holstein_matrix.csv"

# First row = header, first column = index
df = pd.read_csv(
    csv_filename,
    header=0,
    index_col=0
)

# Ensure index is datetime
df.index = pd.to_datetime(df.index)

# Keep only winter months (Dec, Jan, Feb)
#df = df[df.index.month.isin([10, 11, 12, 1, 2, 3])]
df = df[df.index.month.isin([12, 1, 2])]

# Drop rows that contain at least one NA
df = df.dropna()
X = np.array(df)
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
# Model Evaluation with Pair Type Classification
# =============================================================================
Chi_fitted = custom_matrix_product(A_hat, A_hat.T)
clique_max_array = np.array(tuned_pure) if not isinstance(tuned_pure, np.ndarray) else tuned_pure
full_range = np.arange(d)
impure = np.setdiff1d(full_range, clique_max_array)

# Extract pairs for fitted and empirical matrices
x_pairs = extract_pairs(Chi_fitted, clique_max_array, impure)
y_pairs = extract_pairs(Chi, clique_max_array, impure)

# Access values for each pair type
x_values_pure_pure = x_pairs['pure_pure']
x_values_pure_impure = x_pairs['pure_impure']
x_values_impure_impure = x_pairs['impure_impure']
x_values_all = x_pairs['all']

y_values_pure_pure = y_pairs['pure_pure']
y_values_pure_impure = y_pairs['pure_impure']
y_values_impure_impure = y_pairs['impure_impure']
y_values_all = y_pairs['all']

# Calculate global R²
r2_total = r2_score(y_values_all, x_values_all)

print(f"\nModel performance by pair type:")
print(f"  Global R²: {r2_total:.4f}")
print(f"  Total pairs: {len(x_values_all)}")
print(f"  Pure-pure pairs: {len(x_values_pure_pure)}")
print(f"  Pure-impure pairs: {len(x_values_pure_impure)}")
print(f"  Impure-impure pairs: {len(x_values_impure_impure)}")

# =============================================================================
# Colored Scatter Plot by Pair Type
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 7))

# Define colors for each pair type
colors = {
    'pure_pure': '#2E8B57',      # Green
    'pure_impure': '#FF8C00',    # Orange
    'impure_impure': '#1E90FF',  # Blue
}

# Plot each pair type with corresponding color
if len(x_values_pure_pure) > 0:
    ax.scatter(x_values_pure_pure, y_values_pure_pure, 
               alpha=0.7, edgecolor='k', s=40,
               color=colors['pure_pure'], 
               label='Pure-Pure')

if len(x_values_pure_impure) > 0:
    ax.scatter(x_values_pure_impure, y_values_pure_impure, 
               alpha=0.7, edgecolor='k', s=40,
               color=colors['pure_impure'], 
               label='Pure-Impure')

if len(x_values_impure_impure) > 0:
    ax.scatter(x_values_impure_impure, y_values_impure_impure, 
               alpha=0.7, edgecolor='k', s=40,
               color=colors['impure_impure'], 
               label='Impure-Impure')

# Add y = x reference line
min_val = min(np.min(x_values_all), np.min(y_values_all))
max_val = max(np.max(x_values_all), np.max(y_values_all))
ax.plot([min_val, max_val], [min_val, max_val], 
        'r--', linewidth=2, color='#C71585')

# Add R² text annotation
ax.text(
    0.05, 0.95, f'$R^2$ = {r2_total:.3f}',
    transform=ax.transAxes,
    fontsize=18,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
)

# Configure axes
ax.set_xlabel(r'$\tilde{\mathcal{X}}^{\kappa^{*}, \bar{\kappa}^{*}}$', fontsize=25)
ax.set_ylabel(r'$\hat{\mathcal{X}}$', fontsize=25)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 1.0))

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'Chi_fitted_vs_Chi_colored.pdf'), dpi=300, bbox_inches='tight')
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
            'Feature': df.columns[idx],
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

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'Feature_Assignments_Heatmap.pdf'))
plt.close()