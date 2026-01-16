from itertools import product
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import est_impure
import est_pure
import networkx as nx
import concurrent.futures
from tqdm import tqdm
from sklearn.metrics import r2_score
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rc
from matplotlib.colors import ListedColormap
arcane_colors = [
    "#C71585",  # intense magenta
    "#9D7FBA",  # dusty violet
    "#324AB2"
]

# Create colormap
arcane_cmap = ListedColormap(arcane_colors, name='arcane')


def custom_matrix_product(matrix_a, matrix_b):
    """
    Calculate the product between two matrices using the sum of minima component-wise.

    Parameters:
    - matrix_a: First matrix (shape m×n)
    - matrix_b: Second matrix (shape n×p)

    Returns:
    - Resulting matrix (shape m×p)
    """
    # Check if the number of columns in the first matrix matches the number of rows in the second matrix
    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError(
            "Shape mismatch: matrix_a columns must equal matrix_b rows.")

    # Reshape matrix_a to (m, n, 1) and matrix_b to (1, n, p)
    a_reshaped = matrix_a[:, :, np.newaxis]  # Shape: (2, 3, 1)
    b_reshaped = matrix_b[np.newaxis, :, :]  # Shape: (1, 3, 2)

    # Compute minima and sum along axis=1 (the shared 'n' dimension)
    minima = np.minimum(a_reshaped, b_reshaped)  # Shape: (2, 3, 2)
    result = np.sum(minima, axis=1)  # Shape: (2, 2)

    return result


def ext_cor_mat_optimized(X, n, k):
    data = pd.DataFrame(X)
    datarank = np.array(data.rank(axis=0, method='first'))
    indicator = datarank > n - k  # shape (n, d)

    d = indicator.shape[1]
    test_sum = np.zeros((d, d), dtype=np.float64)

    for row in indicator:
        # outer product for one observation
        test_sum += np.outer(row, row)

    test_sum /= k
    return test_sum

def compute_crit(A_hat, Theta):
    """
    Compute the criterion as the R^2 between the upper triangular
    parts of Chi_A and Theta.

    Parameters:
    - A_hat (np.array): Estimated matrix A
    - Theta (np.array): Extremal correlation (chi) matrix
    - pure: unused (kept for interface compatibility)

    Returns:
    - crit (float): R^2 value
    """
    # Compute Chi_A = A_hat A_hat^T
    Chi_A = custom_matrix_product(A_hat, A_hat.T)
    # Extract upper triangular entries (excluding diagonal)
    iu = np.triu_indices_from(Theta, k=1)
    x = Chi_A[iu]
    y = Theta[iu]
    crit = r2_score(y, x)
    return crit

def calibrate(Theta, kappa_1_list, kappa_2_list):
    """
    Calibrate the model by tuning kappa_1 and kappa_2

    Returns:
        tuned_kappa_1 (float)
        tuned_kappa_2 (float)
        tuned_pure (list)
    """
    value_crit = -10e6
    tuned_kappa_1 = None
    tuned_kappa_2 = None
    tuned_pure = []

    for kappa_1 in tqdm(kappa_1_list, desc="kappa_1 loop"):
        # Build adjacency matrix with kappa_1
        adjacency_matrix = ((Theta < kappa_1) * 1.0)
        G = nx.from_numpy_array(np.array(adjacency_matrix))
        cliques = list(nx.find_cliques(G))
        clique_max = max(cliques, key=len)

        for kappa_2 in tqdm(kappa_2_list, desc=f"kappa_2 loop (k1={kappa_1:.4f})", leave=False):

            for clique in cliques:
                if len(clique) == len(clique_max):

                    A_hat, pure = est_impure.est_A(
                        Theta, kappa_1, kappa_2, clique
                    )

                    crit = compute_crit(A_hat, Theta)

                    if crit > value_crit:
                        value_crit = crit
                        tuned_kappa_1 = kappa_1
                        tuned_kappa_2 = kappa_2
                        tuned_pure = pure

                        tqdm.write(
                            f"New optimal: crit={crit:.4f}, "
                            f"kappa_1={kappa_1:.4f}, "
                            f"kappa_2={kappa_2:.4f}"
                        )

    return tuned_kappa_1, tuned_kappa_2, tuned_pure

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
# 2. Keep only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# 3. Transform to standard Pareto margins
def to_standard_pareto(series):
    ranks = series.rank(method="average")
    n = len(series)
    u = ranks / (n + 1)          # empirical CDF
    return 1 / (1 - u)

df_pareto = numeric_df.apply(to_standard_pareto)

def ext_cor_mat_optimized(X, n, k):
    data = pd.DataFrame(X)
    datarank = np.array(data.rank(axis=0, method='first'))
    indicator = datarank > n - k  # shape (n, d)

    d = indicator.shape[1]
    test_sum = np.zeros((d, d), dtype=np.float64)

    for row in indicator:
        # outer product for one observation
        test_sum += np.outer(row, row)

    test_sum /= k
    return test_sum

n, d = df.shape
print(n, d)
r = 1
k = int(0.25 * np.power(np.log(4 * d * n**2), 1/(2*r+1))
        * np.power(n, 2*r/(2*r+1)))
print(k)

Chi = ext_cor_mat_optimized(df_pareto, df_pareto.shape[0], k)

def plot_matrix(M, labels=None, title="Matrix plot"):
    plt.figure()
    plt.imshow(M)
    plt.colorbar()
    
    if labels is not None:
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)

    plt.title(title)
    plt.tight_layout()
    plt.show()


plot_matrix(Chi, labels=df_pareto.columns,
            title="Tail dependence coefficient (χ)")

# Number of values
num_kappa = 100

# Constants (same idea as before, but length = 20)
constants_kappa = np.linspace(0.25, 1.5, num_kappa)

# Define kappa lists
kappa_1_list = [
    c * np.power(np.log(4 * d * n**2) / n, r / (2*r + 1))
    for c in constants_kappa
]

kappa_2_list = [
    c * np.power(np.log(4 * d * n**2) / n, r / (2*r + 1))
    for c in constants_kappa
]

print(kappa_1_list)

# Calibrate the model and obtain the tuned delta and pure indices
tuned_kappa_1, tuned_kappa_2, tuned_pure = calibrate(
    Chi, kappa_1_list, kappa_2_list
)

# Output the tuned delta and pure indices
print("Tuned kappa_1:", tuned_kappa_1)
print("Tuned kappa_2:", tuned_kappa_2)
print("Tuned Pure Indices:", tuned_pure)

A_hat, clique_max = est_impure.est_A(Chi, tuned_kappa_1, tuned_kappa_2, clique_max=tuned_pure)
# Find binary rows (rows that contain exactly one 1 and the rest 0s)
binary_rows = []
for i in range(A_hat.shape[0]):
    if np.sum(A_hat[i] == 1) == 1 and np.all(np.logical_or(A_hat[i] == 0, A_hat[i] == 1)):
        binary_rows.append(i)

# Create results directory if it doesn't exist
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# =============================================================================
# 1. CONFIGURATION ET PRÉPARATION
# =============================================================================
# Calcul de Chi_fitted
Chi_fitted = custom_matrix_product(A_hat, A_hat.T)

# Assurer que clique_max est un tableau numpy
clique_max_array = np.array(clique_max) if not isinstance(clique_max, np.ndarray) else clique_max

# Calcul des indices impurs (complément de clique_max par rapport à [0, d-1])
full_range = np.arange(d)
impure = np.setdiff1d(full_range, clique_max_array)

# =============================================================================
# 2. FONCTION OPTIMISÉE POUR EXTRACTION DES PAIRES
# =============================================================================
def extract_pairs(matrix, pure_indices, impure_indices):
    """
    Extrait les paires pures-pures, pures-impures et impures-impures 
    d'une matrice (upper diagonal uniquement).
    
    Parameters:
    -----------
    matrix : ndarray, shape (d, d)
        Matrice carrée
    pure_indices : ndarray
        Indices des variables pures
    impure_indices : ndarray
        Indices des variables impures
        
    Returns:
    --------
    dict: Dictionnaire avec les paires pour chaque type
    """
    # Vérifications
    d = matrix.shape[0]
    n_pure = len(pure_indices)
    n_impure = len(impure_indices)
    
    # Créer des masques booléens pour un accès rapide
    pure_mask = np.zeros(d, dtype=bool)
    pure_mask[pure_indices] = True
    impure_mask = np.zeros(d, dtype=bool)
    impure_mask[impure_indices] = True
    
    # Obtenir tous les indices de l'upper diagonal
    rows, cols = np.triu_indices(d, k=1)
    n_pairs = len(rows)
    
    # Déterminer le type de chaque paire
    row_is_pure = pure_mask[rows]
    col_is_pure = pure_mask[cols]
    row_is_impure = impure_mask[rows]
    col_is_impure = impure_mask[cols]
    
    # Masques pour chaque type de paire
    pure_pure_mask = row_is_pure & col_is_pure
    pure_impure_mask = (row_is_pure & col_is_impure) | (row_is_impure & col_is_pure)
    impure_impure_mask = row_is_impure & col_is_impure
    
    # Extraire les valeurs
    all_values = matrix[rows, cols]
    
    return {
        'pure_pure': all_values[pure_pure_mask],
        'pure_impure': all_values[pure_impure_mask],
        'impure_impure': all_values[impure_impure_mask],
        'all': all_values,
        'masks': {
            'pure_pure': pure_pure_mask,
            'pure_impure': pure_impure_mask,
            'impure_impure': impure_impure_mask
        }
    }

# =============================================================================
# 3. EXTRACTION OPTIMISÉE POUR CHI_FITTED (x) ET CHI (y)
# =============================================================================
# Extraire les paires pour Chi_fitted (x)
x_pairs = extract_pairs(Chi_fitted, clique_max_array, impure)

# Extraire les paires pour Chi (y) - même procédé
y_pairs = extract_pairs(Chi, clique_max_array, impure)

# =============================================================================
# 4. ACCÈS AUX VALEURS
# =============================================================================
# Valeurs x (Chi_fitted)
x_values_pure_pure = x_pairs['pure_pure']
x_values_pure_impure = x_pairs['pure_impure']
x_values_impure_impure = x_pairs['impure_impure']
x_values_all = x_pairs['all']

# Valeurs y (Chi) - correspondant exactement aux mêmes paires
y_values_pure_pure = y_pairs['pure_pure']
y_values_pure_impure = y_pairs['pure_impure']
y_values_impure_impure = y_pairs['impure_impure']
y_values_all = y_pairs['all']

# =============================================================================
# 1. CALCUL DU R² GLOBAL
# =============================================================================
# Calculer R² global
r2_total = r2_score(y_values_all, x_values_all)

print(f"R² global: {r2_total:.4f}")
print(f"Nombre de paires total: {len(x_values_all)}")

# =============================================================================
# 2. SCATTER PLOT AVEC COULEURS DIFFÉRENTES
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 7))

# Définir les couleurs pour chaque type
colors = {
    'pure_pure': '#2E8B57',      # Vert
    'pure_impure': '#FF8C00',    # Orange
    'impure_impure': '#1E90FF',  # Bleu
}

# Tracer chaque type avec sa couleur
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

# Ligne y = x
min_val = min(np.min(x_values_all), np.min(y_values_all))
max_val = max(np.max(x_values_all), np.max(y_values_all))
ax.plot([min_val, max_val], [min_val, max_val], 
        'r--', linewidth=2, color='#C71585')

# =============================================================================
# 3. AJOUT DU R² ET CONFIGURATION
# =============================================================================
# Texte R²
ax.text(
    0.05, 0.95, f'$R^2$ = {r2_total:.3f}',
    transform=ax.transAxes,
    fontsize=18,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
)

# Labels des axes
ax.set_xlabel(r'$\tilde{\mathcal{X}}^{\kappa^{*}, \bar{\kappa}^{*}}$', fontsize=25)
ax.set_ylabel(r'$\hat{\mathcal{X}}$', fontsize=25)

# Taille des ticks
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

# Grille
ax.grid(True, alpha=0.3)

# Légende
ax.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 1.0))
plt.tight_layout()

# =============================================================================
# 4. SAUVEGARDE
# =============================================================================
output_path = os.path.join(results_dir, 'Chi_fitted_vs_Chi_colored.pdf')
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Get upper triangular indices (excluding diagonal)
d = Chi_fitted.shape[0]
rows, cols = np.triu_indices(d, k=1)  # k=1 excludes the diagonal

# Flatten only the upper diagonal values
x_values = Chi_fitted[rows, cols]
y_values = Chi[rows, cols]

# Calculate R²
r2 = r2_score(y_values, x_values)
print(f"R² score for Chi_fitted vs Chi (upper diagonal only): {r2:.4f}")

# Create scatter plot
fig1, ax1 = plt.subplots(figsize=(7, 6))
ax1.scatter(x_values, y_values, alpha=0.6, edgecolor='k', color='#324AB2')
ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, color='#C71585')

# Add R² text
ax1.text(
    0.05, 0.95, f'$R^2$ = {r2:.3f}',
    transform=ax1.transAxes,
    fontsize=14,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
)

# Axis labels
ax1.set_xlabel(r'$\tilde{\mathcal{X}}^{\kappa^{*}, \bar{\kappa}^{*}}$', fontsize=20)
ax1.set_ylabel(r'$\hat{\mathcal{X}}$', fontsize=20)

# Increase tick label sizes
ax1.tick_params(axis='x', labelsize=18)
ax1.tick_params(axis='y', labelsize=18)

ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'Chi_fitted_vs_Chi_scatter.pdf'))
plt.close()

K_hat = A_hat.shape[1]

# Convert to set for faster lookup (recommended)
tuned_pure_set = set(tuned_pure)

# Create the K_hat table
cluster_data = []
for a in range(K_hat):
    support = np.where(A_hat[:, a] > 0)[0]
    weights = A_hat[support, a]

    for idx, weight in zip(support, weights):
        cluster_data.append({
            'Index': idx,
            'Factor': a + 1,
            'Weight': weight,
            'Is Pure': 'Yes' if weight == 1 else 'No',
            'Feature': df.columns[idx]
        })


# Convert to DataFrame
cluster_df = pd.DataFrame(cluster_data)
cluster_df.to_csv('data/cluster_df.csv')

# Pivot data for heatmap
pivot_df = cluster_df.pivot(
    index='Feature',
    columns='Factor',
    values='Weight'
).fillna(0)

# Sort features by strongest factor assignment
sorted_features = pivot_df.max(axis=1).sort_values(ascending=False).index
pivot_df = pivot_df.loc[sorted_features]

# Create figure for heatmap only
plt.figure(figsize=(7, 6))

ax = sns.heatmap(
    pivot_df,
    cmap='YlGnBu',
    linewidths=0.5,
    cbar_kws={'label': 'Factor Weight'}
)

# Formatting
ax.set_xlabel('Factor', fontsize=20)
ax.set_ylabel('Variables', fontsize=20)

# Increase tick label sizes
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

# Ensure proper y-tick alignment
ax.set_yticks(np.arange(len(pivot_df)) + 0.5)
ax.set_yticklabels(pivot_df.index)

plt.tight_layout()

# Save only the heatmap
plt.savefig(os.path.join(results_dir, 'Feature_Assignments_Heatmap.pdf'))
plt.close()