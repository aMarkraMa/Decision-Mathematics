import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mip import *
import sys
#exo1
exc = pd.ExcelFile("AGRIBALYSE3.1_partie agriculture_conv_vf.xlsx")
# usecols choisir colone 0 et colone 3 à colone 19 de file xlsx
df = pd.read_excel(exc, 'AGB_agri_conv', usecols = [0]+[i for i in range(3,20)])
# 0 partir de ligne 3
data = df[2:].values

print("----------------------------------------start----------------------------------------")

#exo2
#les facteurs coorespondant
#facteurs de critère 1 = facteurs[0]
facteurs = [7.55e3, 5.23e-2, 4.22e3, 4.09e1, 5.95e-4, 1.29e-4,
            1.73e-5, 5.56e1, 1.61, 1.95e1, 1.77e2, 5.67e4, 
            8.19e5, 1.15e4, 6.5e4, 6.35e-2]
#Itérer sur data et normaliser toutes les données demandées
#critères 1 = data[i][2]
for i in range(len(data)):
    for j in range(2, len(data[0])):
        data[i][j] = data[i][j] / facteurs[j - 2]

print(data[0])
#exo3

c1 = [alter[2] for alter in data] #critère 1
c14 = [alter[15] for alter in data] #critère 14
c16 = [alter[17] for alter in data] #critère 16

plt.scatter(c1, c14, s = 10)
plt.xlabel("Critère1")
plt.ylabel("Critère14")
plt.title("Critère1 vs Critère14")
plt.show()

plt.scatter(c1, c16, s = 10)
plt.xlabel("Critère1")
plt.ylabel("Critère16")
plt.title("Critère1 vs Critère16")
plt.show()

plt.scatter(c14, c16, s = 10)
plt.xlabel("Critère14")
plt.ylabel("Critère16")
plt.title("Critère14 vs Critère16")
plt.show()

#Approche incrémental
#w1 = np.arange(0, 1, 0.1)
#sp = [(c1[0] * i + c14[0] * (1 - i)) for i in w1]
#plt.plot(w1, sp)
#plt.show()


#exo5

'''
description:
Détermine si l'alternative X Pareto-domine l'alternative Y.

arguments:
x : Liste des valeurs pour chaque critère de l'alternative X.
y : Liste des valeurs pour chaque critère de l'alternative Y.
    
retourne:
bool: True si X Pareto-domine Y, sinon False.
'''
def pareto_domine(x, y):

    au_moins_1_mieux = False
    for xi, yi in zip(x, y):
        if xi < yi:
            au_moins_1_mieux = True
        elif xi > yi:
            return False
    return au_moins_1_mieux

print(pareto_domine(data[0][2:], data[1][2:]))

#exo6
#retourne True si le propriété est vérifiée, False sinon.
def transitivite(x, y, z):
    if pareto_domine(x, y) and pareto_domine(y, z):
        return pareto_domine(x, z)
    return True
def antisymetrique(x, y):
    if pareto_domine(x, y):
        return not pareto_domine(y, x)
    if pareto_domine(y, x):
        return not pareto_domine(x, y)
    return True
def irreflexivite(x):
    return not pareto_domine(x, x)

#exo7

n = len(data)
nb_paire_total = n * (n - 1) / 2
nb_paire_pareto = 0

for i in range(len(data) - 1):
    for j in range(i+1, len(data)):
        if(pareto_domine(data[i][2:], data[j][2:])):
            nb_paire_pareto = nb_paire_pareto + 1

print('Pourcentage avec 16 critères: {:.3%}'.format(nb_paire_pareto / nb_paire_total))

nb_paire_pareto = 0

for i in range(len(data) - 1):
    for j in range(i+1, len(data)):
        if(pareto_domine([data[i][2], data[i][15], data[i][17]], [data[j][2], data[j][15], data[j][17]])):
            nb_paire_pareto = nb_paire_pareto + 1

print('Pourcentage avec 1, 14, 16 critère: {:.3%}'.format(nb_paire_pareto / nb_paire_total))

#exo10
print("------------------------------------exo10------------------------------------")
w_hat = [21.06e-2, 6.31e-2, 5.01e-2, 4.78e-2, 8.96e-2, 1.84e-2, 2.13e-2, 6.2e-2, 2.8e-2, 2.96e-2, 3.71e-2, 1.92e-2, 7.94e-2, 8.51e-2, 8.32e-2, 7.55e-2] 

def L1_inv(X, Y, w_hat):

    if(SPw_hat(X) >=  SPw_hat(Y)):
        print("SPw_hat(X) >= SPw_hat(Y)")
        return
    n = len(X)  # nombre des critères
    m = Model("L1_inv")  # crée module de programmation linéaire

    # ajouter des variable
    # lb : lower bound
    # CONTINUOUS: variable continu
    # difference: variable auxiliaire
    w = [m.add_var(var_type=CONTINUOUS, lb=0) for i in range(n)]
    difference = [m.add_var(var_type=CONTINUOUS, lb=0) for i in range(n)]

    # ajouter objet
    m.objective = minimize(xsum(difference[i] for i in range(n)))

    # ajouter des contraintes
    m += xsum(w[i] * (X[i] - Y[i]) for i in range(n)) >= 0  # SPw(X) >= SPw(Y)
    m += xsum(w[i] for i in range(n)) == 1  # Σ wi = 1 (i∈[1,16])

    # contrainte de valeur absolue
    for i in range(n):
        m += w[i] - w_hat[i] <= difference[i]
        m += w_hat[i] - w[i] <= difference[i]

    # trouver le résolution optimal
    m.optimize()

    # retourner l'optimal
    return m.objective_value

def SPw_hat(X):
    return xsum(w_hat[i] * X[i] for i in range(len(w_hat)))

print("SPw_hat 0",SPw_hat(data[3][2:]))
print("SPw_hat 1",SPw_hat(data[4][2:]))
print("w_hat:", w_hat)
optimal_value = L1_inv(data[3][2:], data[4][2:], w_hat)
print("valeur optimal：", optimal_value)
