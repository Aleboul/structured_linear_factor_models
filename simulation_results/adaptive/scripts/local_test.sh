#!/bin/bash

# Simulate SLURM_ARRAY_TASK_ID to test all combinations
for SLURM_ARRAY_TASK_ID in {0..79}; do  # Test just 4 combinations to start
    echo "=== Simulation of job $SLURM_ARRAY_TASK_ID ==="

    # Parameters
    n_values=(10000 9000 8000 7000 6000 5000 4000 3000 2000 1000)
    K_values=(5 20)
    operators=("sum" "max")
    noise_values=("true" "false")

    # Calculate indices (updated for 10 n_values)
    index=$SLURM_ARRAY_TASK_ID
    n_index=$((index / 8))           # 8 =  2 (K_values) * 2 (operators) * 2 (noise_values)
    remainder=$((index % 8))
    K_index=$((remainder / 4))        # 4 = 2 (operators) * 2 (noise_values)
    remainder=$((remainder % 4))
    op_index=$((remainder / 2))       # 2 = 2 (noise_values)
    noise_index=$((remainder % 2))

    # Parameter values
    n=${n_values[$n_index]}
    k_fraction=${k_fractions[$k_index]}
    K=${K_values[$K_index]}
    operator=${operators[$op_index]}
    noise=${noise_values[$noise_index]}

    # Fixed parameters (reduced for local testing)
    d=100
    s=4
    eta=0.2
    alpha=1
    r=1
    c_kappa=0.75 # Constant for kappa calculation

    # Calculate k using the formula
    k=$(python3 -c "import math; d=$d; n=$n; r=$r; print(int(0.25*(math.log(4*d*n**2))**(1/(2*r+1)) * n**(2*r/(2*r+1))))")

    # Calculate kappa using the formula
    kappa=$(python3 -c "import math; c_kappa=$c_kappa; d=$d; n=$n; r=$r; print(c_kappa * (math.log(4*d*n**2) / n)**(r/(2*r+1)))")

    echo "Tested combination:"
    echo "n=$n, k=$k, K=$K, op=$operator, noise=$noise"
    echo "d=$d, s=$s, kappa=$kappa"

    # Execution with error checking
    python3 run_factor.py $d $K $s $eta $alpha $n $k $kappa $operator $noise || {
        echo "Failure for simulated job $SLURM_ARRAY_TASK_ID"
        continue
    }
done
