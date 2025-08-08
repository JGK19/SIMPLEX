import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configurações
tamanhos = [101,201,301,401,501,601,701,801,901,1001,2001]
K = 10

min_alunos = 10
max_alunos = 90
min_cap = 30
max_cap = 90
min_d = 1
max_d = 10
seed = 42

solvers = ["hungaro", "gulosa"]
solver_otimo = "ensalamento_ampl_gurobi"

resultados_tempo = {solver: [] for solver in solvers + [solver_otimo]}
resultados_gap = {solver: [] for solver in solvers}

for n in tamanhos:
    nome_instancia = f"D{n}_S{n}_[{min_alunos},{max_alunos}]_[{min_cap},{max_cap}]_[{min_d},{max_d}]_seed{seed}"
    caminho_csv = f"solutions/resultado_{nome_instancia}.csv"

    if not os.path.exists(caminho_csv):
        print(f"[AVISO] Arquivo não encontrado: {caminho_csv}")
        for solver in solvers + [solver_otimo]:
            resultados_tempo[solver].append(None)
        for solver in solvers:
            resultados_gap[solver].append(None)
        continue

    df = pd.read_csv(caminho_csv)

    # Obter média dos tempos para cada solver (até K execuções)
    medias_tempo = {}
    custos = {}
    for solver in solvers + [solver_otimo]:
        linhas_solver = df[df["codigo"] == solver]
        tempos = linhas_solver["tempo_execucao"].tolist()
        custos_solver = linhas_solver["custo"].tolist()

        if len(tempos) < K or len(custos_solver) < K:
            print(f"[AVISO] Menos de {K} execuções para solver {solver} na instância {nome_instancia}")
            medias_tempo[solver] = None
            custos[solver] = None
        else:
            medias_tempo[solver] = np.mean(tempos[:K])
            custos[solver] = np.mean(custos_solver[:K])

    # Guardar tempos
    for solver in solvers + [solver_otimo]:
        resultados_tempo[solver].append(medias_tempo[solver])

    # Calcular gap percentual para heurísticas em relação à solução ótima
    custo_otimo = custos[solver_otimo]
    for solver in solvers:
        custo_heur = custos[solver]
        if custo_otimo is None or custo_heur is None:
            gap = None
        else:
            gap = 100.0 * (custo_heur - custo_otimo) / custo_otimo
        resultados_gap[solver].append(gap)

# --- Plotar gráfico tempo de execução ---
plt.figure(figsize=(12,6))
for solver in solvers + [solver_otimo]:
    tempos = resultados_tempo[solver]
    plt.plot(tamanhos, tempos, marker='o', label=solver)
plt.xlabel("Tamanho da Instância (n x n)")
plt.ylabel("Tempo Médio de Execução (s)")
plt.title(f"Tempo médio de execução (média de {K} execuções)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figuras/tempo_execucao_heuristicas.png")
plt.show()

# --- Plotar gráfico gap percentual ---
plt.figure(figsize=(12,6))
for solver in solvers:
    gaps = resultados_gap[solver]
    plt.plot(tamanhos, gaps, marker='o', label=solver)
plt.xlabel("Tamanho da Instância (n x n)")
plt.ylabel("Gap Médio Percentual (%)")
plt.title(f"Gap percentual médio em relação a {solver_otimo} (média de {K} execuções)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figuras/gap_percentual_heuristicas.png")
plt.show()
