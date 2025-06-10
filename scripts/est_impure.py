"""
    The provided code consists of a set of functions designed to estimate the matrix A in a Linear Factor Model based on given data.

    Once the pure indices and maximum clique are determined, the est_AI function sets the corresponding values in the matrix A to 1. 
    On the other hand, the est_AJ function handles the impure indices by adjusting their values based on a specified criteria involving
    the extremal correlation matrix Chi and a tuning parameter delta. Finally, the est_A function orchestrates the entire process,
    first estimating the pure indices and maximum clique, then utilizing these results to populate the matrix A with appropriate values,
    thus providing a comprehensive representation of the underlying relationships within the data.
"""

import numpy as np
import est_pure


def projection_simplex_sort(v, z=1):
    """
    Project a vector v onto the probability simplex.

    Args:
        v (np.array[float]): Input vector to be projected.
        z (float): Radius of the simplex, default is 1.

    Returns:
        np.array[float]: The projected vector.
    """
    n_features = v.shape[0]  # Get the number of features in the input vector

    # Sort the input vector v in descending order
    u = np.sort(v)[::-1]

    # Compute the cumulative sum of the sorted vector minus z
    cssv = np.cumsum(u) - z

    # Create an array of indices (1, 2, ..., n_features)
    ind = np.arange(n_features) + 1

    # Determine the condition where u - cssv / ind is positive
    cond = u - cssv / ind > 0

    # Find the largest index rho that satisfies the condition
    rho = ind[cond][-1]

    # Calculate the threshold theta
    theta = cssv[cond][-1] / float(rho)

    # Compute the projection by subtracting theta from v and applying a maximum with 0
    w = np.maximum(v - theta, 0)

    return w


def est_AI(A, pure):
    """
    Estimate the matrix of pure indices.

    Args:
        A (np.array[float, float]): A d x K matrix to be updated.
        pure (list[]): List of pure indices, where each element is an array of indices for a cluster.

    Returns:
        np.array[float, float]: The updated matrix A with pure indices set to 1.
    """
    # Iterate over each cluster and its corresponding indices in the pure list
    for clst, index in enumerate(pure):
        # Iterate over each index in the current cluster
        for i in index:
            # Set the value at position (i, clst) in the matrix A to 1
            A[i, clst] = 1

    # Return the updated matrix A
    return A


def est_AJ(A, Chi, delta, pure):
    """
    Estimate the submatrix associated to impure indices.

    Args:
        A (np.array[float, float]): A d x K matrix to be updated.
        Chi (np.array([float, float])): Extremal correlation matrix.
        delta (float): Tuning parameter.
        pure (list[]): List of pure indices, where each element is an array of indices for a cluster.

    Returns:
        np.array[float, float]: The updated matrix A with impure indices set based on the given criteria.
    """
    K_hat = len(pure)  # Number of clusters in the pure indices
    d = Chi.shape[0]   # Dimensionality of the extremal correlation matrix
    # Find indices of impure variables by subtracting pure indices from all indices
    impure = np.setdiff1d(np.array(range(d)), np.hstack(pure))

    # Iterate over each impure feature
    for j in impure:
        # Calculate the mean of correlations between the impure feature and each cluster
        chi_bar = np.array([np.mean(Chi[j, pure[i]]) for i in range(K_hat)])

        # Apply a hard threshold to the mean correlations
        hard_threshold = chi_bar * (chi_bar > delta)

        # Find indices where the hard threshold is greater than 0
        index = np.where(hard_threshold > 0)[0]

        # If there are indices where the hard threshold is greater than 0
        if len(index) > 0:
            # Initialize beta_hat with zeros
            beta_hat = np.zeros(K_hat)
            # Apply projection onto the simplex to ensure sum of beta_hat is 1
            beta_hat[index] = projection_simplex_sort(hard_threshold[index])
        else:
            # If no index satisfies the condition, set beta_hat to zeros
            beta_hat = np.zeros(K_hat)

        # Update the row corresponding to the impure feature in matrix A with beta_hat
        A[j, :] = beta_hat

    # Return the updated matrix A
    return A


def est_A(Chi, delta):
    """
    Estimate the matrix A in a Linear Factor Model.

    Args:
        Chi (np.array([float, float])): Extremal correlation matrix.
        delta (float): Tuning parameter.

    Returns:
        np.array[float, float]: The estimated matrix A.
    """
    d = Chi.shape[0]  # Dimensionality of the extremal correlation matrix
    # Estimate the set of pure indices and the maximum clique using est_pure_set
    pure, clique_max = est_pure.est_pure_set(Chi, delta)
    K_hat = len(pure)  # Number of clusters in the pure indices

    # Initialize matrix A_hat with zeros
    A_hat = np.zeros((d, K_hat))

    # Set values to pure indices using est_AI function
    A_hat = est_AI(A_hat, pure)

    # Set values to impure indices using est_AJ function
    A_hat = est_AJ(A_hat, Chi, delta, pure)

    return A_hat  # Return the estimated matrix A_hat
