import os
import pandas as pd
import matplotlib.pyplot as plt

tamanhos = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001, 2001]
solvers = ["highs", "cbc", "gurobi", "xpress", "cplex", "mosek", "copt"]
K = 10

min_alunos = 10
max_alunos = 90
min_cap = 30
max_cap = 90
min_d = 1
max_d = 10
seed = 42

resultados = {solver: [] for solver in solvers}

for n in tamanhos:
    nome_instancia = f"D{n}_S{n}_[{min_alunos},{max_alunos}]_[{min_cap},{max_cap}]_[{min_d},{max_d}]_seed{seed}"
    caminho_csv = f"solutions/resultado_{nome_instancia}.csv"

    if not os.path.exists(caminho_csv):
        print(f"[AVISO] Arquivo não encontrado: {caminho_csv}")
        for solver in solvers:
            resultados[solver].append(None)
        continue

    df = pd.read_csv(caminho_csv)

    for solver in solvers:
        nome = f"ensalamento_ampl_{solver}"
        tempos = df[df['codigo'] == nome]['tempo_execucao'].tolist()

        if len(tempos) < K:
            print(f"[AVISO] Apenas {len(tempos)} execuções encontradas para {solver} na instância {nome_instancia}. Esperado: {K}")
            resultados[solver].append(None)
        else:
            media_tempo = sum(tempos[:K]) / K
            resultados[solver].append(media_tempo)

# Plotar gráfico
plt.figure(figsize=(12, 6))
for solver in solvers:
    tempos = resultados[solver]
    plt.plot(tamanhos, tempos, marker='o', label=solver)

plt.xlabel("Tamanho da Instância (n x n)")
plt.ylabel("Tempo Médio de Execução (s)")
plt.title(f"Tempo médio de execução por solver (média de {K} execuções)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figuras/tempo_execucao_solvers.png")
plt.show()
