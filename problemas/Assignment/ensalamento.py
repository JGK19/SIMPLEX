import pulp
import numpy as np

# -----------------------------
# Dados do problema (extraídos do artigo)
# -----------------------------

# Número de disciplinas e salas
n_d = 10
n_s = 10

# Número de alunos por disciplina (segunda aula)
N = [54, 51, 35, 40, 42, 31, 20, 51, 50, 24]

# Capacidade das salas
C = [44, 44, 42, 44, 30, 55, 52, 50, 51, 50]

# Sala da 1ª aula de cada disciplina
salas_primeira_aula = ['S9', 'S2', 'S10', 'S8', 'S5', 'S7', 'S4', 'S3', 'S1', 'S6']
sala_idx = {'S1':0, 'S2':1, 'S3':2, 'S4':3, 'S5':4, 'S6':5, 'S7':6, 'S8':7, 'S9':8, 'S10':9}
origem = [sala_idx[s] for s in salas_primeira_aula]

# Matriz de distância entre salas (Figura 1)
D = [
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
]

# -----------------------------
# Modelo em PuLP
# -----------------------------

# Criar o modelo
model = pulp.LpProblem("Problema_de_Alocacao_de_Salas", pulp.LpMinimize)

# Variáveis binárias x[i][j] = 1 se disciplina i vai para sala j
x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(n_s)] for i in range(n_d)]

# Função objetivo: minimizar distância total ponderada pelo número de alunos
model += pulp.lpSum(
    x[i][j] * D[origem[i]][j] * N[i] for i in range(n_d) for j in range(n_s)
)

# Restrição 1: cada disciplina deve ser alocada em exatamente uma sala
for i in range(n_d):
    model += pulp.lpSum(x[i][j] for j in range(n_s)) == 1

# Restrição 2: cada sala recebe no máximo uma disciplina
for j in range(n_s):
    model += pulp.lpSum(x[i][j] for i in range(n_d)) <= 1

# Restrição 3: respeitar a capacidade da sala
for i in range(n_d):
    for j in range(n_s):
        if N[i] > C[j]:
            model += x[i][j] == 0  # Não pode alocar se a sala não comporta os alunos

# -----------------------------
# Resolver o modelo
# -----------------------------
solver = pulp.PULP_CBC_CMD(msg=True)
model.solve(solver)

# -----------------------------
# Resultados
# -----------------------------

print(f"\nStatus da solução: {pulp.LpStatus[model.status]}")
print(f"Deslocamento total ótimo: {pulp.value(model.objective)}")

print("\nAlocação ótima (Disciplina -> Sala):")
for i in range(n_d):
    for j in range(n_s):
        if pulp.value(x[i][j]) == 1:
            print(f"  Disciplina D{i+1} → Sala S{j+1}")
