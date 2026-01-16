import numpy as np
from sklearn.preprocessing import QuantileTransformer
import MLExtreme as mlx
import pandas as pd
import pprint as pp

# ----------------------
# Étape 1 : Charger tes données (remplace ceci par ton propre chargement)
# ----------------------
# Exemple : simulation de données aléatoires
# Remplace cette ligne par : X = ton_tableau_numpy

csv_filename = "data/nhanes_dr1tot.csv"

data = pd.read_csv(csv_filename)
# 1. Drop rows with NA values
data = data.dropna()
X = np.array(data)  # Trans, q = 1-0.005, lasso 0.035
# Banks avec q = 1-0.01 (ou 0.05) lasso = 0.035
d = X.shape[1]
n = X.shape[0]

# X = X[:, contained_indices]
print(X)
print(n, d)

# ----------------------
# Étape 2 : Transformation en rangs
# ----------------------
X_ranked = mlx.rank_transform(X)
norm_Xt = np.linalg.norm(X_ranked, axis=1, ord=2)
# ----------------------
# Étape 3 : Seuil pour détecter les extrêmes
# ----------------------
# Ici on garde les 10% les plus extrêmes
r=1
k = int(0.25 * np.power(np.log(4 * d * n**2), 1/(2*r+1))
        * np.power(n, 2*r/(2*r+1)))  # 0.075
quantile_threshold = 1-k/n
#thresholds = np.quantile(X_ranked, quantile_threshold, axis=0)
#extreme_mask = np.any(X_ranked > thresholds, axis=1)
quantile_threshold = np.quantile(norm_Xt, quantile_threshold)
extreme_mask = norm_Xt > quantile_threshold
X_extreme = X_ranked[extreme_mask]
print(np.sum(extreme_mask))

# ----------------------
# Étape 4 : Appliquer DAMEX
# ----------------------
clef = mlx.clef(kappa_min=0.1) 
clef.fit(X_extreme)

# ----------------------
# Étape 5 : Afficher les clusters détectés
# ----------------------
clusters = clef.subfaces
for i, cluster in enumerate(clusters):
    support_names = data.columns[cluster]
    print(support_names)

#subfaces_select, masses_select = damex.fit(X_extreme)
#weights_select = masses_select / np.sum(masses_select)
#
#dict_faces_select, dict_weights_select = mlx.list_to_dict(subfaces_select,
#                                                          weights_select)
#
#print("Damex with selected epsilon: subfaces")
#pp.pprint(dict_faces_select)
#
#
#print("final output: weights = normalized masses")
#pp.pprint(dict_weights_select)