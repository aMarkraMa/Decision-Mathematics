import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

exc = pd.ExcelFile("AGRIBALYSE3.1_partie agriculture_conv_vf.xlsx")
df = pd.read_excel(exc, 'AGB_agri_conv', usecols = [0]+[i for i in range(3,20)])
data = df[2:].values

data_without_description = [ligne[1:] for ligne in data]

data_without_description = data_without_description[:-3]
for row in data_without_description:
    print(row)

print(len(data_without_description[0]))

facteurs_normalisation = [
    7.55e3, 5.23e-2, 4.22e3, 4.09e1, 5.95e-4, 1.29e-4,
    1.73e-5, 5.56e1, 1.61, 1.95e1, 1.77e2, 5.67e4,
    8.19e5, 1.15e4, 6.5e4, 6.36e-2
]
donnees_normalisees = [
    [valeur_brute / facteur for valeur_brute, facteur in zip(ligne, facteurs_normalisation)]
    for ligne in data_without_description
]

print("Données normalisées:")

print(len(donnees_normalisees))

# for row in donnees_normalisees:
#     print(row)

c1 = [row[0] for row in donnees_normalisees]  # Critère 1
c14 = [row[13] for row in donnees_normalisees]  # Critère 14
c16 = [row[15] for row in donnees_normalisees]  # Critère 16

# Graphique: Critère 1 (x) vs Critère 14 (y)
plt.figure()
plt.scatter(c1, c14)
plt.xlabel("Impact sur le dérèglement climatique (kgCO2eq/kg)")
plt.ylabel("Impact sur l'épuisement des ressources en eau (m3/kg)")
plt.title("Impact climatique vs Épuisement des ressources en eau")
plt.show()

# Graphique: Critère 1 (x) vs Critère 16 (y)
plt.figure()
plt.scatter(c1, c16)
plt.xlabel("Impact sur le dérèglement climatique (kgCO2eq/kg)")
plt.ylabel("Impact de l'empreinte matière (kgSbeq/kg)")
plt.title("Impact climatique vs Empreinte matière")
plt.show()

# Graphique: Critère 14 (x) vs Critère 16 (y)
plt.figure()
plt.scatter(c14, c16)
plt.xlabel("Impact sur l'épuisement des ressources en eau (m3/kg)")
plt.ylabel("Impact de l'empreinte matière (kgSbeq/kg)")
plt.title("Épuisement des ressources en eau vs Empreinte matière")
plt.show()