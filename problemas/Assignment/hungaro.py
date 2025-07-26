import numpy as np
from scipy.optimize import linear_sum_assignment

# -----------------------------
# 1) Dados
# -----------------------------
# Número de disciplinas / salas
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
# 2) Matriz de custo penalizada Cp[i,j]
# -----------------------------
Cp = np.zeros((n,n), dtype=int)
for i in range(n):
    for j in range(n):
        if N[i] <= C[j]:
            Cp[i,j] = D[origem[i], j] * N[i]
        else:
            Cp[i,j] = 1000

print(Cp)
# -----------------------------
# 3) Aplicar o método Húngaro
# -----------------------------
row_ind, col_ind = linear_sum_assignment(Cp)

# -----------------------------
# 4) Calcular o custo “real”
# -----------------------------
real_cost = sum(D[origem[i], col_ind[idx]] * N[i] for idx, i in enumerate(row_ind))

# -----------------------------
# 5) Exibir resultados
# -----------------------------
print(f"Deslocamento total (recontado): {real_cost}\n")
print("Alocação obtida:")
for i, j in zip(row_ind, col_ind):
    print(f"  Disciplina D{i+1} → Sala S{j+1}")
