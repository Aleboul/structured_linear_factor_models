#!/bin/bash
# SLURM job submission script for factor model simulation

# =============================================
# SLURM Directives - Resource Allocation
# =============================================

# Node allocation
#SBATCH -N 1                 # Request 1 Node

# Partition specification
#SBATCH --partition=cpu
# Array job configuration
#SBATCH --array=0-159%1        # 159 tasks, max 1 concurrent

# Resource requirements
#SBATCH --mem=256GB               # 50GB memory per node
#SBATCH --time=24:00:00           # Time limit (24 hour)

# Output files
#SBATCH --output=logs/factor_model_%A_%a.out
#SBATCH --error=logs/factor_model_%A_%a.err

# Project accounting
#SBATCH --account=buechack_0002

# Load required Python environment
module load python/3.8

# =============================================
# Simulation Parameter Definitions
# =============================================

# Possible values for each parameter:
n_values=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)       # Sample sizes
k_fractions=(0.01 0.05)     # Threshold fractions
K_values=(5 20)             # Factor dimensions
operators=("sum" "max")     # Operators to test
noise_values=("true" "false") # Noise conditions

# =============================================
# Parameter Calculation for Current Task
# =============================================

# Current array index (0-31)
index=$SLURM_ARRAY_TASK_ID

# Calculate parameter indices:
# (Distributes 32 possible combinations)

# 1. Select n (2 choices):
n_index=$((index / 16))      # 0 or 1 → 5000 or 10000
remainder=$((index % 16))    # Remainder for other params

# 2. Select k_fraction (2 choices):
k_index=$((remainder / 8))   # 0 or 1 → 0.01 or 0.05
remainder=$((remainder % 8)) # New remainder

# 3. Select K (2 choices):
K_index=$((remainder / 4))   # 0 or 1 → 5 or 20
remainder=$((remainder % 4)) # New remainder

# 4. Select operator and noise (2×2 choices):
op_index=$((remainder / 2))  # 0 or 1 → sum or max
noise_index=$((remainder % 2)) # 0 or 1 → true or false

# Get current parameter values
n=${n_values[$n_index]}
k_fraction=${k_fractions[$k_index]}
K=${K_values[$K_index]}
operator=${operators[$op_index]}
noise=${noise_values[$noise_index]}

# Calculate k value (rounded to nearest integer)
k=$(python3 -c "print(int($n * $k_fraction + 0.5))")

# =============================================
# Fixed Simulation Parameters
# =============================================
d=1000       # Total data dimension
s=4         # Max non-zeros per row
eta=0.2     # Minimum threshold
alpha=1     # Shape parameter
kappa=0.1   # Regularization parameter

# =============================================
# Execute Python Script with Parameters
# =============================================
python3 run_factor.py $d $K $s $eta $alpha $n $k $kappa $operator $noise