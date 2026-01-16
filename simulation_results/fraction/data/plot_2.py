from scipy.optimize import linear_sum_assignment
import itertools
import numpy as np
from scipy.stats import pareto
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_helpers import *
plt.style.use('qb-light.mplstyle')


def find_best_column_permutation(A, A_hat):
    """
    Find the permutation matrix P that minimizes ‖A - A_hat @ P‖_F
    """
    # Compute cost matrix: cost[i, j] = ||A[:, i] - A_hat[:, j]||^2
    n = A.shape[1]
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = A[:, i] - A_hat[:, j]
            cost[i, j] = np.sum(diff ** 2)

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost)

    # Construct permutation matrix P
    P = np.zeros((n, n))
    P[col_ind, row_ind] = 1  # Note: A_hat @ P aligns columns with A

    return P


# Parameters for the simulation
d_values = [100, 1000]  # Dimensionality of the data
n_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000,
            8000, 9000, 10000]  # Different values for n
# Different values for k as a fraction of n
k_values = [0.01, 0.05, "adaptive"]
K_values = [5, 20]  # Different values for K
niter = 100  # Number of iterations
alpha = 1.0  # Parameter for margin transformation
eta = 0.2  # Minimum value for each dimension of the samples in the simplex
s = 4  # Maximum number of non-zero elements in each row of A
kappa = 0.1  # Tuning parameter for estimation
operators = ["max", "sum"]  # Operators for data generation
noise_values = ["false", "true"]  # Whether to add noise

# Iterate over each combination of parameters
for operator in operators:
    for noise in noise_values:
        # Create all combinations of parameters
        all_params = list(itertools.product(n_values, k_values, K_values))
        mat_stock_norm_infty_2 = np.zeros(
            (len(d_values), len(K_values), len(k_values), len(n_values)))
        mat_stock_recovery_rate_K = np.zeros(
            (len(d_values), len(K_values), len(k_values), len(n_values)))
        mat_stock_recovery_pure = np.zeros(
            (len(d_values), len(K_values), len(k_values), len(n_values)))
        mat_stock_recovery_rate_s = np.zeros(
            (len(d_values), len(K_values), len(k_values), len(n_values)))
        mat_stock_recovery_TFPP = np.zeros(
            (len(d_values), len(K_values), len(k_values), len(n_values)))
        mat_stock_recovery_TFNP = np.zeros(
            (len(d_values), len(K_values), len(k_values), len(n_values)))

        for d_idx, d in enumerate(d_values):
            for n, k_fraction, K in all_params:
                # Calculate k based on the fraction of n
                k = int(k_fraction * n) if k_fraction != "adaptive" else "adaptive"
                recovery_rate_K = []
                norm_infty_2 = []
                recovery_rate_pure = []
                recovery_rate_s = []
                TFPP = []
                TFNP = []
                for i in range(niter):
                    params = {
                        'n': n, 'k': k, 'd': d, 'K': K, 's': s, 'eta': eta,
                        'alpha': alpha, 'kappa': kappa, 'operator': operator,
                        'noise': noise, 'niter': i
                    }
                    if k_fraction == "adaptive":
                        filename_A_bar = f"results_adaptive/operator{params['operator']}/noise{params['noise']}/results_n{params['n']}/d{params['d']}/K{params['K']}/s{params['s']}_eta{params['eta']}_alpha{params['alpha']}/A_bar_niter{params['niter']}.csv"
                        filename_A_hat = f"results_adaptive/operator{params['operator']}/noise{params['noise']}/results_n{params['n']}/d{params['d']}/K{params['K']}/s{params['s']}_eta{params['eta']}_alpha{params['alpha']}/A_hat_niter{params['niter']}.csv"
                    else:
                        filename_A_bar = f"results/operator{params['operator']}/noise{params['noise']}/results_n{params['n']}/d{params['d']}/K{params['K']}/k{params['k']}_s{params['s']}_eta{params['eta']}_alpha{params['alpha']}_kappa{params['kappa']}/A_bar_niter{params['niter']}.csv"
                        filename_A_hat = f"results/operator{params['operator']}/noise{params['noise']}/results_n{params['n']}/d{params['d']}/K{params['K']}/k{params['k']}_s{params['s']}_eta{params['eta']}_alpha{params['alpha']}_kappa{params['kappa']}/A_hat_niter{params['niter']}.csv"

                    A_bar = np.array(pd.read_csv(filename_A_bar))
                    A_hat = np.array(pd.read_csv(filename_A_hat))

                    condition = A_hat > 0
                    indices = np.where(condition, A_hat, -1)
                    row_indices, col_indices = np.where(condition)
                    unique_rows, counts = np.unique(
                        row_indices, return_counts=True)
                    s_hat = np.max(counts)

                    K_hat = A_hat.shape[1]
                    pure = np.where(A_bar == 1)[0]
                    est_pure = np.where(A_hat == 1)[0]

                    # Computation of exact recovery rate of number of clusters
                    FP = []
                    FN = []
                    hat_positive = []
                    hat_null = []
                    positive = []
                    null = []
                    if K == K_hat:
                        # Find best permutation matrix

                        P = find_best_column_permutation(A_hat, A_bar)
                        A_bar = A_bar @ P

                        recovery_rate_K.append(1)
                        norm_infty_2.append(
                            np.max(np.linalg.norm(A_hat - A_bar, axis=1)))
                        for a in range(K_hat):
                            hat_positive_group = np.where(A_hat[:, a] > 0.0)[0]
                            hat_null_group = np.where(A_hat[:, a] == 0.0)[0]
                            positive_group = np.where(A_bar[:, a] > 0.0)[0]

                            null_group = np.where(A_bar[:, a] == 0.0)[0]

                            hat_positive.append(len(hat_positive_group))
                            hat_null.append(len(hat_null_group))
                            positive.append(len(positive_group))
                            null.append(len(null_group))

                            FP.append(
                                len(np.intersect1d(null_group, hat_positive_group)))
                            FN.append(
                                len(np.intersect1d(positive_group, hat_null_group)))

                        TFPP.append(np.sum(FP) / np.sum(null))
                        TFNP.append(np.sum(FN) / np.sum(positive))
                    else:
                        recovery_rate_K.append(0)
                        norm_infty_2.append(np.nan)
                        TFPP.append(np.nan)
                        TFNP.append(np.nan)
                    if s_hat == s:
                        recovery_rate_s.append(1)
                    else:
                        recovery_rate_s.append(0)
                    if np.array_equal(pure, est_pure):
                        recovery_rate_pure.append(1)
                    else:
                        recovery_rate_pure.append(0)

                # Find the indices for k and K
                k_index = k_values.index(k_fraction)
                K_index = K_values.index(K)
                n_index = n_values.index(n)

                # Store norm_infty_2 values in mat_stock
                mat_stock_norm_infty_2[d_idx, K_index, k_index,
                                       n_index] = np.nanmean(norm_infty_2)
                mat_stock_recovery_rate_K[d_idx, K_index, k_index,
                                          n_index] = np.mean(recovery_rate_K)
                mat_stock_recovery_rate_s[d_idx, K_index, k_index,
                                          n_index] = np.mean(recovery_rate_s)
                mat_stock_recovery_pure[d_idx, K_index, k_index,
                                        n_index] = np.mean(recovery_rate_pure)
                mat_stock_recovery_TFPP[d_idx, K_index,
                                        k_index, n_index] = np.nanmean(TFPP)
                mat_stock_recovery_TFNP[d_idx, K_index,
                                        k_index, n_index] = np.nanmean(TFNP)

        # Create a single figure with subplots arranged in six rows and two columns
        fig, axs = plt.subplots(6, 2, figsize=(15, 20))

        # Row titles
        row_titles = ['Factor dimension',
                      'Sparsity index',
                      'Pure Variables',
                      'TFNP',
                      'TFPP',
                      'Matrix Estimation Error']

        # Column titles
        column_titles = ['$K=5$', '$K=20$']

        # Plot the data for each d value and k value
        for d_idx, d in enumerate(d_values):
            linestyle = '--' if d == 100 else '-'
            color = '#324AB2' if d == 100 else '#C71585'
            for k_idx, k_fraction in enumerate(k_values):
                marker = 'o' if k_fraction == 0.01 else '^' if k_fraction == 0.05 else 'D'
                for K_idx, K in enumerate(K_values):
                    # Create appropriate label with LaTeX notation
                    if k_fraction == "adaptive":
                        label = f'd={d}, adaptive'
                    else:
                        label = fr'd={d}, $k={k_fraction} \cdot n$'  # Using LaTeX notation
                    
                    # Plot for each metric
                    axs[0, K_idx].plot(n_values, mat_stock_recovery_rate_K[d_idx, K_idx, k_idx, :],
                                       marker=marker, linestyle=linestyle, markerfacecolor='white',
                                       color=color, label=label)
                    axs[1, K_idx].plot(n_values, mat_stock_recovery_rate_s[d_idx, K_idx, k_idx, :],
                                       marker=marker, linestyle=linestyle, markerfacecolor='white',
                                       color=color, label=label)
                    axs[2, K_idx].plot(n_values, mat_stock_recovery_pure[d_idx, K_idx, k_idx, :],
                                       marker=marker, linestyle=linestyle, markerfacecolor='white',
                                       color=color, label=label)
                    axs[3, K_idx].plot(n_values, mat_stock_recovery_TFNP[d_idx, K_idx, k_idx, :],
                                       marker=marker, linestyle=linestyle, markerfacecolor='white',
                                       color=color, label=label)
                    axs[4, K_idx].plot(n_values, mat_stock_recovery_TFPP[d_idx, K_idx, k_idx, :],
                                       marker=marker, linestyle=linestyle, markerfacecolor='white',
                                       color=color, label=label)
                    axs[5, K_idx].plot(n_values, mat_stock_norm_infty_2[d_idx, K_idx, k_idx, :],
                                       marker=marker, linestyle=linestyle, markerfacecolor='white',
                                       color=color, label=label)

        # Calculate global min/max for each relevant row
        # For TFNP (row 3)
        tfnp_min = np.nanmin(mat_stock_recovery_TFNP)
        tfnp_max = np.nanmax(mat_stock_recovery_TFNP)

        # For TFPP (row 4)
        tfpp_min = np.nanmin(mat_stock_recovery_TFPP)
        tfpp_max = np.nanmax(mat_stock_recovery_TFPP)

        # For Matrix Estimation Error (row 5)
        error_min = np.nanmin(mat_stock_norm_infty_2)
        error_max = np.nanmax(mat_stock_norm_infty_2)

        # Set common y-axis limits for each row with padding
        padding_factor = 0.05  # 5% padding
        row_ylimits = [
            # Exact Recovery Rate Factors
            (0.0 - padding_factor, 1.0 + padding_factor),
            # Exact Recovery Rate Sparsity
            (0.0 - padding_factor, 1.0 + padding_factor),
            # Exact Recovery Rate Pure Variables
            (0.0 - padding_factor, 1.0 + padding_factor),
            (tfnp_min - padding_factor*(tfnp_max-tfnp_min),
             tfnp_max + padding_factor*(tfnp_max-tfnp_min)),
            (tfpp_min - padding_factor*(tfpp_max-tfpp_min),
             tfpp_max + padding_factor*(tfpp_max-tfpp_min)),
            (error_min - padding_factor*(error_max-error_min),
             error_max + padding_factor*(error_max-error_min))
        ]

        for row_idx, (ax_row, ylim) in enumerate(zip(axs, row_ylimits)):
            for ax in ax_row:
                ax.set_ylim(ylim)
                ax.tick_params(axis='both', labelsize=14)  # <--- Add this line

        # Add horizontal grid lines for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        # For the first three rows (rates), set fixed ticks
        if row_idx < 3:
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        # Add row titles
        for ax_row, row_title in zip(axs, row_titles):
            ax_row[0].set_ylabel(row_title, fontsize=16,
                                 labelpad=40, color='#8A8A8C')

        # Add column titles
        for ax, col_title in zip(axs[0], column_titles):
            ax.set_title(col_title, fontsize=16, pad=20)

        # Add common axis labels
        for ax in axs[-1, :]:
            ax.set_xlabel('n')

        # Add legend
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                   ncol=2, fontsize=16, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Ajuste pour la légende
        plt.savefig(
            f"results_operator{params['operator']}_noise{params['noise']}.pdf", bbox_inches='tight')
