#!/bin/bash

# Simule SLURM_ARRAY_TASK_ID pour tester toutes les combinaisons
for SLURM_ARRAY_TASK_ID in {0..159}; do  # Testez juste 4 combinaisons pour commencer
    echo "=== Simulation du job $SLURM_ARRAY_TASK_ID ==="
    
    # Paramètres
    n_values=(10000 9000 8000 7000 6000 5000 4000 3000 2000 1000)
    k_fractions=(0.01 0.05)
    K_values=(10 20)
    operators=("sum" "max")
    noise_values=("true" "false")

    # Calcul des indices (updated for 10 n_values)
    index=$SLURM_ARRAY_TASK_ID
    n_index=$((index / 16))           # 16 = 2 (k_fractions) * 2 (K_values) * 2 (operators) * 2 (noise_values)
    remainder=$((index % 16))
    k_index=$((remainder / 8))        # 8 = 2 (K_values) * 2 (operators) * 2 (noise_values)
    remainder=$((remainder % 8))
    K_index=$((remainder / 4))        # 4 = 2 (operators) * 2 (noise_values)
    remainder=$((remainder % 4))
    op_index=$((remainder / 2))       # 2 = 2 (noise_values)
    noise_index=$((remainder % 2))

    # Valeurs des paramètres
    n=${n_values[$n_index]}
    k_fraction=${k_fractions[$k_index]}
    K=${K_values[$K_index]}
    operator=${operators[$op_index]}
    noise=${noise_values[$noise_index]}

    # Calcul de k
    k=$(python3 -c "print(int($n * $k_fraction + 0.5))")

    # Paramètres fixes (réduits pour le test local)
    d=100  
    s=4
    eta=0.2
    alpha=1
    kappa=0.1

    echo "Combinaison testée:"
    echo "n=$n, k=$k, K=$K, op=$operator, noise=$noise"
    echo "d=$d, s=$s"

    # Exécution avec vérification d'erreur
    python3 run_factor.py $d $K $s $eta $alpha $n $k $kappa $operator $noise || {
        echo "Échec pour le job simulé $SLURM_ARRAY_TASK_ID"
        continue
    }
done
