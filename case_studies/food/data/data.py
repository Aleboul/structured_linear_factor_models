import pandas as pd
import requests
import io

# 1. Charger les données
url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DR1TOT_I.xpt"
response = requests.get(url)
df = pd.read_sas(io.BytesIO(response.content), format='xport')

# Afficher quelques colonnes
print(df['DR1TPROT'])
print(df.iloc[:,33])
print(df.iloc[:,73])

# Extraire les colonnes 32 à 73
df = df.iloc[:,32:73]

# Supprimer les colonnes spécifiques si elles existent
columns_to_keep = ['DR1TRET', 'DR1TACAR', 'DR1TLZ', 'DR1TVARA', 'DR1TBCAR', 'DR1TVK']
df = df[columns_to_keep]

rename_dict = {
    'DR1TRET': 'RET',
    'DR1TACAR': 'AC',
    'DR1TLZ': 'LZ',
    'DR1TVARA': 'VA',
    'DR1TBCAR': 'BC',
    'DR1TVK': 'VK'
}

df = df.rename(columns=rename_dict)


print(f"\nDimensions finales: {df.shape}")
print(f"Colonnes finales: {list(df.columns)}")
print(df.shape)

# 2. Sauvegarder en CSV
csv_filename = "data/nhanes_dr1tot.csv"
df.to_csv(csv_filename, index=False)
print(f"Données sauvegardées dans : {csv_filename}")
print(f"Taille du fichier : {df.shape}")

# 3. Recharger pour vérification
df_loaded = pd.read_csv(csv_filename)

# 4. Vérifications rapides
print("\n=== VÉRIFICATION RAPIDE ===")
print(f"Dimensions originales : {df.shape}")
print(f"Dimensions après rechargement : {df_loaded.shape}")

# Vérifier l'égalité
if df.shape == df_loaded.shape:
    print("✓ Dimensions identiques")
else:
    print("✗ Problème de dimensions")

# Vérifier les sommes pour quelques colonnes numériques
print("\nVérification des sommes :")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:3]
for col in numeric_cols:
    sum_orig = df[col].sum()
    sum_loaded = df_loaded[col].sum()
    if abs(sum_orig - sum_loaded) < 0.0001:
        print(f"  {col}: ✓ Sommes identiques ({sum_orig:.2f})")
    else:
        print(f"  {col}: ✗ Différence ({sum_orig:.2f} vs {sum_loaded:.2f})")

print("\n✓ Vérification terminée !")