"""
    This file contains all the helping functions to estimate the set of pure variables. The whole methodology is described by Theorem 2 in the main paper.

    The `est_pure_set' function identifies the pure indices, representing the strongest correlations between variables, using Mixed Integer
    Linear Programming. These indices are crucial for understanding the underlying structure of the data. The `est_clique' function complements
    this by finding the maximum clique, a subset of variables with mutual correlations surpassing a given threshold.

"""

from pickletools import optimize
import numpy as np
import networkx as nx
from scipy.optimize import LinearConstraint
from scipy.optimize import milp


def est_clique(Chi, kappa):
    """
    Estimate the maximum clique (using mixed integer linear programming) in a undirected graph constructed with a extremal correlation matrix Chi and a threshold parameter.

    Args:
        Chi (np.array[float, float]): Extremal correlation matrix
        kappa (float): Tuning parameter

    Returns:
        np.array[int]: Indices of the maximum clique
    """
    # Create a complementary matrix where elements greater than or equal to kappa are 1, otherwise 0
    complementary_matrix = (Chi >= kappa) * 1.0
    d = complementary_matrix.shape[0]  # Dimension of the matrix

    vect = []  # List to store constraint vectors

    # Build the constrain matrix, see the binary problem in the main paper
    for j in range(d):
        # Find indices in the j-th row of the complementary matrix where the value is greater than min(j, 0)
        index = np.where(complementary_matrix[j, :] > min(j, 0))[0]
        # Further filter indices to only include those greater than j
        index = index[np.where(index > j)]

        for i in index:
            # Create a zero vector of length d
            input = np.zeros(d)
            # Set the j-th and i-th positions to 1
            input[j] = 1
            input[i] = 1
            # Add the vector to the list
            vect.append(input)

    # Convert the list of vectors to a numpy array
    vect = np.array(vect)
    b_u = np.ones(vect.shape[0])  # Upper bounds for constraints (all ones)
    b_l = np.zeros(vect.shape[0])  # Lower bounds for constraints (all zeros)

    # Create linear constraints for the MILP problem
    constraints = LinearConstraint(vect, b_l, b_u)
    # Objective function coefficients
    c = -np.ones(complementary_matrix.shape[0])

    # Specify that all variables in the MILP problem should be integers
    integrality = np.ones_like(c)

    # Solve the MILP problem
    res = milp(c=c, constraints=constraints, integrality=integrality)

    # Extract indices where the solution vector x is greater than 0.5 (indicating inclusion in the clique)
    clique = np.where(res.x > 0.5)[0]

    return clique


def est_pure_set(Chi, kappa):
    """
    Estimate pure indices set using Mixed Integer Linear Programming

    Args:
        Chi (np.array[float, float]): Extremal correlation matrix
        kappa (float): Tuning parameter of the algorithm

    Returns:
        Pure indices    
    """
    # Compute the maximum clique using a helper function est_clique
    clique_max = est_clique(Chi, kappa)

    # Initialize the list to store pure indices
    pure = []
    # Create an array S containing all indices not in the maximum clique
    S = np.array(np.setdiff1d(range(Chi.shape[0]), clique_max))

    # Loop over each index in the maximum clique
    for i in clique_max:
        # Find indices where the condition 1 - Chi[i, :] < kappa holds
        index = np.where(1 - Chi[i, :] < kappa)[0]
        # Update index to include the intersection with S and add the current index i
        index = np.union1d(np.intersect1d(S, index), i)
        # Add the current set of indices to the pure list
        pure.append(index)
        # Update S by removing the indices in the current set
        S = np.setdiff1d(S, index)
    # Return the list of pure indices and the maximum clique
    return pure, clique_max
