import pandas as pd
exc = pd.ExcelFile("AGRIBALYSE3.1_partie agriculture_conv_vf.xlsx")
# usecols choisir colone 0 et colone 3 à colone 19 de file xlsx
df = pd.read_excel(exc, 'AGB_agri_conv', usecols = [0]+[i for i in range(3,20)])
# 0 partir de ligne 3
data = df[2:].values

print("---------------------------start---------------------------")
print(data[1:])

facteurs = [7.55e3, 5.23e-2, 4.22e3, 4.09e1, 5.95e-4, 1.29e-4,
            1.73e-5, 5.56e1, 1.61, 1.95e1, 1.77e2, 5.67e4, 
            8.19e5, 1.15e4, 6.5e4, 6.35e-2]

for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j] = data[i][j] * facteurs[j]
