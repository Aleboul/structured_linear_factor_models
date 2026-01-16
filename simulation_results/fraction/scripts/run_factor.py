# run_factor.py
import os
import numpy as np
import pandas as pd
import est_impure
from clayton.rng.evd import Logistic
from concurrent.futures import ProcessPoolExecutor
import sys


def sample_factors(K, n_samples, alpha):
    """
    Sample uniform margins and transform them into Pareto margins.

    Parameters:
    - K (int): Dimension of the samples.
    - n_samples (int): Total number of samples to generate.
    - alpha (float): Parameter for the margin transformation.

    Returns:
    - np.ndarray: Samples transformed into Pareto margins.
    """
    # Initialize the Logistic object to generate Gaussian samples
    clayton = Logistic(dim=K, n_samples=n_samples, theta=1.0)

    # Sample uniform margins from the Gaussian samples
    sample_unimargin = clayton.sample_unimargin()

    # Transform the uniform margins into Pareto margins
    sample_Fact = np.power(1 / (1 - sample_unimargin), 1/alpha)

    return sample_Fact


def sample_noise(d, n_samples, alpha):
    """
    Sample uniform margins and transform them into Pareto margins for noise.

    Parameters:
    - d (int): Dimension of the samples.
    - n_samples (int): Total number of samples to generate.
    - alpha (float): Parameter for the margin transformation.

    Returns:
    - np.ndarray: Samples transformed into Pareto margins.
    """
    # Initialize the Logistic object to generate Gaussian samples
    clayton = Logistic(dim=d, n_samples=n_samples, theta=1.0)

    # Sample uniform margins from the Gaussian samples
    sample_unimargin = clayton.sample_unimargin()

    # Transform the uniform margins into Pareto margins
    sample_Noise = np.power(1 / (1 - sample_unimargin), 1/(2*alpha))

    return sample_Noise


def sample_uniform_simplex(eta, d, num_samples=1):
    """
    Uniformly sample points in a simplex with constraints.

    Parameters:
    - eta (float): Minimum value for each dimension of the samples.
    - d (int): Dimension of the space.
    - num_samples (int): Number of desired samples. Default is 1.

    Returns:
    - np.ndarray: Uniform samples that satisfy the constraints.
    """
    samples = []
    while len(samples) < num_samples:
        # Uniformly sample in the simplex
        u = np.sort(np.random.uniform(0, 1, d - 1))  # Draw d-1 uniform numbers
        u = np.concatenate(([0], u, [1]))  # Add 0 and 1 to the ends
        y = np.diff(u)  # Compute differences to obtain vector y

        # Check constraints: all components must be within [eta, 1-eta]
        if np.all(y >= eta) and np.all(y <= 1 - eta):
            samples.append(y)
    
    return np.array(samples)


def create_matrix_A(d, K, s, eta, alpha):
    """
    Create a matrix A with specific properties and normalize its rows.

    Parameters:
    - d (int): Number of rows in the final matrix.
    - K (int): Number of columns in the matrix A and size of the identity matrix.
    - s (int): Maximum number of non-zero elements in each row of A.
    - eta (float): Minimum value for each dimension of the samples in the simplex.
    - alpha (float): Exponent for normalization.

    Returns:
    - np.ndarray: The original matrix A and the normalized matrix A_bar.
    """
    A = np.zeros((d - K, K))
    for j in range(d - K):
        if j == 0:
            # Randomly choose the support for the first row
            support = np.random.choice(K, size=s, replace=False)
            A[j, support] = sample_uniform_simplex(eta, s, 1)
        else:
            # Randomly choose the number of non-zero elements in each row
            sparsity = np.random.randint(1, s + 1)
            # Randomly choose the positions of non-zero elements
            support = np.random.choice(K, size=sparsity, replace=False)
            # Set the non-zero elements using a random vector in the simplex
            if sparsity > 1:
                A[j, support] = sample_uniform_simplex(eta, sparsity, 1)
            if sparsity == 1:
                A[j, support] = 1

    # Concatenate the identity matrix with A
    A = np.concatenate((np.eye(K), A))

    # Normalize the rows of A
    norms = np.sum(np.power(A, alpha), axis=1)
    A_bar = np.power(A, alpha) / norms[:, None]

    return A, A_bar


def ext_cor_mat(X, n, k):
    """
    Estimate the extremal correlation matrix.

    Parameters:
    - X (np.ndarray): Input data matrix.
    - n (int): Total number of observations.
    - k (int): Threshold parameter for the rank condition.

    Returns:
    - np.ndarray: Result of the estimation.
    """
    # Convert the input data to a pandas DataFrame
    data = pd.DataFrame(X)

    # Compute the rank of each element along the specified axis
    datarank = np.array(data.rank(axis=0))

    # Create an indicator matrix where elements exceed the threshold n-k
    indicator = datarank > n - k

    # Compute the test sum using the indicator matrix
    test_sum = (indicator[:, :, None] * indicator[:, None, :]).sum(0) / k

    return test_sum


def save_results(A_bar, A_hat, params):
    """
    Save the matrices A_bar and A_hat as CSV files in a structured directory.

    Parameters:
    - A_bar (np.ndarray): Normalized matrix.
    - A_hat (np.ndarray): Estimated matrix.
    - params (dict): Dictionary containing parameters for naming the files.
    """
    # Ensure the results directory exists
    os.makedirs(f"../data/results/operator{params['operator']}/noise{params['noise']}/results_n{params['n']}/d{params['d']}/K{params['K']}/k{params['k']}_s{params['s']}_eta{params['eta']}_alpha{params['alpha']}_kappa{params['kappa']}", exist_ok=True)

    # Construct filenames based on parameters
    filename_A_bar = f"../data/results/operator{params['operator']}/noise{params['noise']}/results_n{params['n']}/d{params['d']}/K{params['K']}/k{params['k']}_s{params['s']}_eta{params['eta']}_alpha{params['alpha']}_kappa{params['kappa']}/A_bar_niter{params['niter']}.csv"
    filename_A_hat = f"../data/results/operator{params['operator']}/noise{params['noise']}/results_n{params['n']}/d{params['d']}/K{params['K']}/k{params['k']}_s{params['s']}_eta{params['eta']}_alpha{params['alpha']}_kappa{params['kappa']}/A_hat_niter{params['niter']}.csv"

    # Save matrices as CSV files
    pd.DataFrame(A_bar).to_csv(filename_A_bar, index=False)
    pd.DataFrame(A_hat).to_csv(filename_A_hat, index=False)


def run_iteration(i, params):
    """
    Run a single iteration of the simulation process.

    Parameters:
    - i (int): Iteration number.
    - params (tuple): Tuple containing parameters for the simulation.
    """
    d, K, s, eta, alpha, n, k, kappa, operator, noise = params
    print(f"Running iteration {i}")

    # Sample factors and create matrix A
    sample_Fact = sample_factors(K, n_samples=n, alpha=alpha)
    A, A_bar = create_matrix_A(d, K, s, eta, alpha)

    # Generate data X based on the operator
    if operator == "sum":
        X = np.array([np.matmul(A, sample_Fact[i, :]) for i in range(n)])
        if noise == "true":
            E = sample_noise(d, n, alpha)
            X = X + E
    if operator == "max":
        X = np.max(A[..., None] * sample_Fact.T, axis=1).T
        if noise == "true":
            E = sample_noise(d, n, alpha)
            X = np.maximum(X, E)

    # Compute the extremal correlation matrix
    chi = ext_cor_mat(X, n, k)

    # Estimate the matrix A_hat
    A_hat = est_impure.est_A(chi, kappa)

    del(chi)

    # Save results for this iteration
    iteration_params = {
        'n': n, 'k': k, 'd': d, 'K': K, 's': s, 'eta': eta,
        'alpha': alpha, 'kappa': kappa, 'operator': operator,
        'noise': noise, 'niter': i
    }
    save_results(A_bar, A_hat, iteration_params)


def main(d, K, s, eta, alpha, n, k, kappa, operator, noise):
    np.random.seed(29041996)  # Seed fixe
    
    params = (d, K, s, eta, alpha, n, k, kappa, operator, noise)
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(run_iteration, range(100), [params] * 100)

if __name__ == "__main__":
    # Récupère les arguments passés au script
    d = int(sys.argv[1])
    K = int(sys.argv[2])
    s = int(sys.argv[3])
    eta = float(sys.argv[4])
    alpha = float(sys.argv[5])
    n = int(sys.argv[6])
    k = int(sys.argv[7])
    kappa = float(sys.argv[8])
    operator = sys.argv[9]
    noise = sys.argv[10]
    
    main(d, K, s, eta, alpha, n, k, kappa, operator, noise)
