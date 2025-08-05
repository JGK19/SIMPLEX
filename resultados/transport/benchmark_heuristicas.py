import os
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tCantoNoroeste import solve_transport_nw, salvar_resultado as salvar_nw
from tVogel import solve_transport_vogel, salvar_resultado as salvar_vogel
from tGuloso import solve_transport_greedy, salvar_resultado as salvar_guloso

def ler_tempos_custos(filepath_csv, codigo, max_repeticoes):
    """
    Lê do CSV os tempos e custos para o código da heurística, até max_repeticoes.
    Retorna listas: tempos, custos
    """
    if not os.path.exists(filepath_csv):
        return [], []

    try:
        df = pd.read_csv(filepath_csv)
        df_filtrado = df[df["codigo"] == codigo]
        df_amostra = df_filtrado.sample(n=min(max_repeticoes, len(df_filtrado)), random_state=None)
        tempos = df_amostra["tempo_execucao"].astype(float).tolist()
        custos = df_amostra["custo"].astype(float).tolist()
        return tempos, custos
    except Exception as e:
        print(f"Erro ao ler {filepath_csv}: {e}")
        return [], []

def calcular_gap_percentual(custo_heuristica, custo_otimo):
    if custo_otimo == 0:
        return 0.0
    return 100.0 * (custo_heuristica - custo_otimo) / custo_otimo

def benchmark_heuristicas():
    tamanhos = [11, 101, 201, 301, 401, 501, 601, 701, 801, 901, 1001, 2001]
    min_val = 1
    max_val = 100
    seed = 42
    rep = 10

    heuristicas = {
        #"tCantoNoroeste": (solve_transport_nw, salvar_nw),
        "tVogel": (solve_transport_vogel, salvar_vogel),
        "tGuloso": (solve_transport_greedy, salvar_guloso),
    }

    tempos_medios = {h: [] for h in heuristicas}
    gaps_medios = {h: [] for h in heuristicas}

    for n in tamanhos:
        print(f"\nTamanho: {n}x{n}")
        nome_arquivo = f"instancias/problema_{n}x{n}_[{min_val},{max_val}]_seed{seed}"
        if not os.path.exists(nome_arquivo):
            print(f"Arquivo não encontrado: {nome_arquivo}")
            continue

        caminho_csv = nome_arquivo.replace("instancias/", "solutions/") + "_resultado.csv"

        # Ler resultados ótimos (considera tClassico, exato)
        # Aqui lemos do arquivo solutions e pegamos o menor custo dentre os códigos que contenham "tClassico"
        custo_otimo = None
        if os.path.exists(caminho_csv):
            try:
                df_resultados = pd.read_csv(caminho_csv)
                custos_otimos = df_resultados[df_resultados["codigo"].str.contains("tClassico")]["custo"]
                if not custos_otimos.empty:
                    custo_otimo = custos_otimos.min()
            except Exception as e:
                print(f"Erro ao ler solução ótima: {e}")

        if custo_otimo is None:
            print(f"Não encontrado custo ótimo para a instância {nome_arquivo}. Ignorando cálculo de gap.")
            custo_otimo = 0

        # Executar as heurísticas em ordem aleatória para completar repetições
        execucoes = list(heuristicas.keys()) * rep
        random.shuffle(execucoes)

        tempos_local = {h: [] for h in heuristicas}
        custos_local = {h: [] for h in heuristicas}

        # Carregar resultados já existentes
        for h in heuristicas:
            t, c = ler_tempos_custos(caminho_csv, h, rep)
            tempos_local[h].extend(t)
            custos_local[h].extend(c)

        for h in execucoes:
            if len(tempos_local[h]) >= rep:
                continue

            print(f"  Heurística {h} - Resolvendo repetição {len(tempos_local[h])+1}/{rep}")
            solve_func, salvar_func = heuristicas[h]
            status, custo, tempo = solve_func(nome_arquivo)
            salvar_func(nome_arquivo, status, custo, tempo)

            tempos_local[h].append(tempo)
            custos_local[h].append(custo)

        # Calcular médias de tempo e gap percentual
        for h in heuristicas:
            media_tempo = np.mean(tempos_local[h])
            media_custo = np.mean(custos_local[h])
            gap = calcular_gap_percentual(media_custo, custo_otimo)
            tempos_medios[h].append(media_tempo)
            gaps_medios[h].append(gap)
            print(f"  Heurística {h} - Tempo médio: {media_tempo:.4f}s, Gap médio: {gap:.4f}%")

    # Criar diretório figuras
    os.makedirs("figuras", exist_ok=True)

    # Plot tempo médio
    plt.figure(figsize=(10, 6))
    for h in heuristicas:
        plt.plot(tamanhos, tempos_medios[h], marker='o', label=h)
    plt.xlabel("Tamanho do problema (n x n)")
    plt.ylabel("Tempo médio de execução (s)")
    plt.title("Comparação de Tempo Médio das Heurísticas")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figuras/tempo_medio_heuristicas.png", dpi=300)
    plt.show()

    # Plot gap percentual médio
    plt.figure(figsize=(10, 6))
    for h in heuristicas:
        plt.plot(tamanhos, gaps_medios[h], marker='o', label=h)
    plt.xlabel("Tamanho do problema (n x n)")
    plt.ylabel("Gap percentual médio (%)")
    plt.title("Comparação do Gap Percentual Médio das Heurísticas")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figuras/gap_percentual_heuristicas.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    benchmark_heuristicas()
