import numpy as np
from scipy.optimize import linear_sum_assignment
from algoHungaro import hungarian_algorithm

# -----------------------------
# 1) Dados
# -----------------------------
n = 10
# Nº de alunos na 2ª aula (N_i)
N = np.array([54, 51, 35, 40, 42, 31, 20, 51, 50, 24])
# Capacidades das salas (C_j)
C = np.array([44, 44, 42, 44, 30, 55, 52, 50, 51, 50])
# Sala de 1ª aula de cada disciplina
salas_primeira_aula = ['S9','S2','S10','S8','S5','S7','S4','S3','S1','S6']
sala_idx = {'S1':0,'S2':1,'S3':2,'S4':3,'S5':4,'S6':5,'S7':6,'S8':7,'S9':8,'S10':9}
origem = np.array([sala_idx[s] for s in salas_primeira_aula])
# Matriz de distâncias D[a][b]
D = np.array([
    [1,2,3,3,3,3,3,3,3,3],
    [2,1,3,3,3,3,3,3,3,3],
    [3,3,1,2,2,2,2,2,2,2],
    [3,3,2,1,2,2,2,2,2,2],
    [3,3,2,2,1,2,2,2,2,2],
    [3,3,2,2,2,1,2,2,2,2],
    [3,3,2,2,2,2,1,2,2,2],
    [3,3,2,2,2,2,2,1,2,2],
    [3,3,2,2,2,2,2,2,1,2],
    [3,3,2,2,2,2,2,2,2,1],
])

# -----------------------------
# 2) Matriz de custo já ponderada W[i,j]
# -----------------------------
W = np.zeros((n,n), dtype=int)
for i in range(n):
    for j in range(n):
        if N[i] <= C[j]:
            W[i,j] = D[origem[i], j] * N[i]
        else:
            W[i,j] = 1000 

print(W)
# -----------------------------
# 3) Aplicar o método Húngaro em W
# -----------------------------
row_ind, col_ind = hungarian_algorithm(W)

# -----------------------------
# 4) Somar o custo direto
# -----------------------------
total = W[row_ind, col_ind].sum()

# -----------------------------
# 5) Mostrar resultado
# -----------------------------
print(f"Deslocamento total (artigo): {total}\n")
print("Alocação segundo o artigo:")
for i, j in zip(row_ind, col_ind):
    print(f"  Disciplina D{i+1} → Sala S{j+1}")
