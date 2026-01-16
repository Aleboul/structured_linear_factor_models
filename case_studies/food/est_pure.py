"""
    This file contains all the helping functions to estimate the set of pure variables. The whole methodology is described by Theorem 2 in the main paper.

    The `est_pure_set' function identifies the pure indices, representing the strongest correlations between variables, using Mixed Integer
    Linear Programming. These indices are crucial for understanding the underlying structure of the data. The `est_clique' function complements
    this by finding the maximum clique, a subset of variables with mutual correlations surpassing a given threshold.

"""

from pickletools import optimize
import numpy as np
import networkx as nx


def est_clique(Chi, delta):
    """Estimate the maximum clique in the adjacency matrix using extremal correlation


        Args:
            Chi (np.array[float, float]):
                Extremal correlation matrix
            delta (float) : 
                Tuning parameter of the algorithm

        Returns:
            List of clique
    """
    adjacency_matrix = ((Chi < delta) * 1.0)  # compute adjacency matrix
    # transform the adjacency matrix
    G = nx.from_numpy_array(np.array(adjacency_matrix))
    # into a python graph.
    clique = max(nx.find_cliques(G), key=len)
    clique = list(np.sort(clique))
    
    return clique