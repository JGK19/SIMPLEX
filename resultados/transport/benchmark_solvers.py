import os
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tClassico_ampl import ler_instancia_csv, resolver_ampl


def ler_tempos_existentes(filepath_csv, codigo, max_repeticoes):
    if not os.path.exists(filepath_csv):
        return []

    try:
        df = pd.read_csv(filepath_csv)
        df_filtrado = df[df["codigo"] == codigo]
        df_amostra = df_filtrado.sample(n=min(max_repeticoes, len(df_filtrado)), random_state=None)
        tempos = df_amostra["tempo_execucao"].astype(float).tolist()
        return tempos
    except Exception as e:
        print(f"Erro ao ler {filepath_csv}: {e}")
        return []


def benchmark_ampl_solvers():
    tamanhos = [11, 101, 201, 301, 401, 501, 601, 701, 801, 901, 1001, 2001]
    min_val = 1
    max_val = 100
    seed = 42
    rep = 10
    solvers = ["cbc", "gurobi", "xpress", "cplex", "mosek"]

    tempos_por_solver = {solver: [] for solver in solvers}

    for n in tamanhos:
        print(f"\nTamanho: {n}x{n}")
        nome_arquivo = f"instancias/problema_{n}x{n}_[{min_val},{max_val}]_seed{seed}"
        if not os.path.exists(nome_arquivo):
            print(f"Arquivo não encontrado: {nome_arquivo}")
            continue

        caminho_csv = nome_arquivo.replace("instancias/", "solutions/") + "_resultado.csv"
        i, j, Oi, Dj, Cost = ler_instancia_csv(nome_arquivo)

        # Define ordem intercalada
        execucoes = solvers * rep
        random.shuffle(execucoes)

        tempos_solver_local = {solver: ler_tempos_existentes(caminho_csv, f"tClassico_ampl_{solver}", rep) for solver in solvers}

        for solver in execucoes:
            if len(tempos_solver_local[solver]) >= rep:
                continue
            print(f"  {solver} - Resolvendo ({len(tempos_solver_local[solver])+1}/{rep})")
            resolver_ampl(i, j, Oi, Dj, Cost, solver, nome_arquivo)

            # Releitura após adicionar resultado
            tempos_solver_local[solver] = ler_tempos_existentes(caminho_csv, f"tClassico_ampl_{solver}", rep)

        for solver in solvers:
            media_tempo = np.mean(tempos_solver_local[solver])
            tempos_por_solver[solver].append(media_tempo)
            print(f"Tempo médio {solver}: {media_tempo:.4f}s")

    # Gráfico
    os.makedirs("figuras", exist_ok=True)
    plt.figure(figsize=(10, 6))
    for solver in solvers:
        plt.plot(tamanhos, tempos_por_solver[solver], marker='o', label=solver)

    plt.xlabel("Tamanho do problema (n x n)")
    plt.ylabel("Tempo médio de execução (s)")
    plt.title("Comparação de Solvers AMPL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figuras/tempo_solvers_ampl.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    benchmark_ampl_solvers()
