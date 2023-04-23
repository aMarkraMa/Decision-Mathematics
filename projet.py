import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from mip import *
from functools import reduce
from operator import mul


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
print("------------------------------------exo7------------------------------------")
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
        return float('inf')
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

    #Vérifier l'état après la résolution du problème d'optimisation. 
    # Si l'état est OPTIMAL ou FEASIBLE, m.objective_value est renvoyé. Sinon, l'infini est renvoyé.
    if m.status == OptimizationStatus.OPTIMAL or m.status == OptimizationStatus.FEASIBLE:
        return m.objective_value
    else:
        return float('inf')

def SPw_hat(X):
    return xsum(w_hat[i] * X[i] for i in range(len(w_hat)))

print("SPw_hat 0: ",SPw_hat(data[3][2:]))
print("SPw_hat 1: ",SPw_hat(data[4][2:]))
print("w_hat:", w_hat)
optimal_value = L1_inv(data[3][2:], data[4][2:], w_hat)
print("valeur optimal:", optimal_value)

#exo11
print("------------------------------------exo11------------------------------------")

#  le nombre de paires telles qu’on peut inverser la préférence en changeant 
# les poids d’une distance L1 inférieure ou égale à x, pour x une valeur donnée.

def count_pairs_l1_inv_le_x(x_values, sample_size=10):
    count_pairs = []
    data_list = data.tolist()  # Conversion des tableaux NumPy en listes
    if sample_size > len(data_list):
        sample_size = len(data_list)
    data_sample = random.sample(data_list, sample_size)  # Échantillons sélectionnés de manière aléatoire Taille de l'échantillon Échantillons
    for x in x_values:
        count = 0
        for i in range(len(data_sample)):
            for j in range(i + 1, len(data_sample)):
                if L1_inv(data_sample[i][2:], data_sample[j][2:], w_hat) <= x:
                    count += 1
        count_pairs.append(count)
    return count_pairs


# donner des valeurs de x
x_values = np.arange(0, 2, 0.2).tolist()
count_pairs = count_pairs_l1_inv_le_x(x_values)


plt.plot(x_values, count_pairs)
plt.xlabel('x')
plt.ylabel('Number of pairs (X, Y) with L1_inv(X, Y) ≤ x')
plt.title('Number of pairs (X, Y) vs. x')
plt.show()

#exo12
print("------------------------------------exo12------------------------------------")
def p_ord(X, w):
    X_sorted = sorted(X)  
    MPO = sum(w[i] * X_sorted[i] for i in range(len(X)))
    return MPO

def p_geo(X, w):
    X_weighted = [max(x_i, 0) ** w_i for x_i, w_i in zip(X, w)] 
    MG = reduce(mul, X_weighted) ** (1 / sum(w))
    return MG


w = [1/16] * 16  

for X in data:
    print(f"MPO pour {X[0]}: {p_ord(X[2:], w)}")
    print(f"MG pour {X[0]}: {p_geo(X[2:], w)}")
    
# exo 15
print("------------------------------------exo15------------------------------------")

#Classement des alternatives 
def rank_alternatives(values):
    return sorted(range(len(values)), key=lambda x: values[x])

#Calcul de la distance de Kendall-Tau
def distance(order1, order2):
    n = len(order1)
    assert n == len(order2), "ils doivent avoir la même longueur"
    
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            rel_order1 = order1[i] - order1[j]
            rel_order2 = order2[i] - order2[j]

            count += 0.5 * (np.sign(rel_order1) != np.sign(rel_order2))
            count += 0.5 * (rel_order1 * rel_order2 < 0)

    return count

values_MPO = [p_ord(X[2:], w) for X in data]
values_MG = [p_geo(X[2:], w) for X in data]

order_MPO = rank_alternatives(values_MPO)
order_MG = rank_alternatives(values_MG)

dkt_MPO_MG = distance(order_MPO, order_MG)

print("dKT(MPO, MG):", dkt_MPO_MG)

#exo 16
def score(x):
    return 100 - (np.log(10 * x + 1) / np.log(2 + 1 / 100 ** 4)) * 20

x_values = np.linspace(0, 10, 1000)  
y_values = score(x_values)  

plt.plot(x_values, y_values)
plt.xlabel('x')
plt.ylabel('score(x)')
plt.title('Graphique de la fonction score(x)')
plt.grid(True)
plt.show()

#exo 17
def categorie(score):
    if 100 >= score >= 80:
        return "A"
    elif 80 > score >= 60:
        return "B"
    elif 60 > score >= 40:
        return "C"
    elif 40 > score >= 20:
        return "D"
    else:
        return "E"
 
 