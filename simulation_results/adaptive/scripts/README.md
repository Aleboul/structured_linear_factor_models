# Simulation for adaptive estimator

## Overview
Monte Carlo simulation study to evaluate the finite-sample performance of the proposed algorithm where k = \{0.01n, 0.05n\} and \kappa = 0.1.

# Project Structure

### Core Python Files

#### `run_factor_model.sh`

A SLURM batch script that launches an array job on an HPC cluster to systematically evaluate 80 different parameter combinations for factor model simulations, managing resource allocation, task distribution, and output logging.

#### `local_test.sh`
A local testing script that runs run_factor.py with 80 different combinations of sample sizes, factor counts, operator types, and noise settings for debugging and validation.

#### `run_factor.py`
The main script that orchestrates the entire analysis pipeline. This file:
- Loads and preprocesses data
- Computes extremal correlation matrices
- Saves results

#### `est_impure.py`
Contains functions for estimating the factor loading matrix A.

#### `est_pure.py`
Dedicated functions for estimating pure variables.

## Required Packages

numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
networkx>=2.5.0
tqdm>=4.60.0

## Usage

### Basic Execution

#### `local_test.sh`

On a computer:

chmod +x local_test.sh
./local_test.sh

#### `run_factor_model.sh`

On a HPC cluster:

chmod +x run_factor_model.sh
sbatch run_factor_model.sh