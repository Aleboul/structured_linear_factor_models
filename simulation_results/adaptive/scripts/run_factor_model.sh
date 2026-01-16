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
#SBATCH --array=0-79%1        # 79 tasks, max 1 concurrent

# Resource requirements
#SBATCH --mem=256GB               # 50GB memory per node
#SBATCH --time=3:00:00           # Time limit (24 hour)

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

# Parameters
n_values=(10000 9000 8000 7000 6000 5000 4000 3000 2000 1000)
K_values=(5 20)
operators=("sum" "max")
noise_values=("true" "false")

# =============================================
# Parameter Calculation for Current Task
# =============================================

# Current array index (0-31)
index=$SLURM_ARRAY_TASK_ID

# Calculate indices (updated for 10 n_values)
index=$SLURM_ARRAY_TASK_ID
n_index=$((index / 8))           # 8 =  2 (K_values) * 2 (operators) * 2 (noise_values)
remainder=$((index % 8))
K_index=$((remainder / 4))        # 4 = 2 (operators) * 2 (noise_values)
remainder=$((remainder % 4))
op_index=$((remainder / 2))       # 2 = 2 (noise_values)
noise_index=$((remainder % 2))
noise_index=$((remainder % 2)) # 0 or 1 → true or false

# Get current parameter values
n=${n_values[$n_index]}
K=${K_values[$K_index]}
operator=${operators[$op_index]}
noise=${noise_values[$noise_index]}

# =============================================
# Fixed Simulation Parameters
# =============================================
# Fixed parameters (reduced for local testing)
d=1000
s=4
eta=0.2
alpha=1
r=1
c_kappa=0.75 # Constant for kappa calculation

# Calculate k using the formula
k=$(python3 -c "import math; d=$d; n=$n; r=$r; print(int(0.25*(math.log(4*d*n**2))**(1/(2*r+1)) * n**(2*r/(2*r+1))))")

# Calculate kappa using the formula
kappa=$(python3 -c "import math; c_kappa=$c_kappa; d=$d; n=$n; r=$r; print(c_kappa * (math.log(4*d*n**2) / n)**(r/(2*r+1)))")

# =============================================
# Execute Python Script with Parameters
# =============================================
python3 run_factor.py $d $K $s $eta $alpha $n $k $kappa $operator $noise