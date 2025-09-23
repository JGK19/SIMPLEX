import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
from tClassico import solve_transport_problem, salvar_resultado as salvar_classico
from tRestrito import solve_flow_problem, salvar_resultado as salvar_fluxo

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

def benchmark_transport():
    tamanhos = [11, 101, 201, 301, 401, 501, 601, 701, 801, 901, 1001, 2001]
    min_val = 1
    max_val = 100
    seed = 42
    rep = 10  # número de repetições

    tempos_classico = []
    tempos_fluxo = []

    for n in tamanhos:
        print(f"\nTamanho: {n}x{n}")

        nome_arquivo = f"instancias/problema_{n}x{n}_[{min_val},{max_val}]_seed{seed}"
        if not os.path.exists(nome_arquivo):
            print(f"Arquivo não encontrado: {nome_arquivo}")
            continue

        caminho_csv = nome_arquivo.replace("instancias/", "solutions/") + "_resultado.csv"

        # ----- Clássico -----
        t_classico = ler_tempos_existentes(caminho_csv, "tClassico", rep)
        print(len(t_classico))
        if len(t_classico) >= rep:
            print("  Usando soluções já existentes para tClassico")
        else:
            for r in range(len(t_classico), rep):
                print(f"  Clássico - Resolvendo repetição {r+1}/{rep}")
                status_c, custo_c, tempo_c = solve_transport_problem(nome_arquivo)
                salvar_classico(nome_arquivo, status_c, custo_c, tempo_c)
                t_classico.append(tempo_c)

        # ----- Fluxo -----
        t_fluxo = ler_tempos_existentes(caminho_csv, "tRestrito", rep)
        if len(t_fluxo) >= rep:
            print("  Usando soluções já existentes para tRestrito")
        else:
            for r in range(len(t_fluxo), rep):
                print(f"  Fluxo - Resolvendo repetição {r+1}/{rep}")
                status_f, custo_f, tempo_f = solve_flow_problem(nome_arquivo)
                salvar_fluxo(nome_arquivo, status_f, custo_f, tempo_f)
                t_fluxo.append(tempo_f)

        # Média dos tempos
        media_c = np.mean(t_classico)
        media_f = np.mean(t_fluxo)

        tempos_classico.append(media_c)
        tempos_fluxo.append(media_f)

        print(f"Tempo médio Clássico: {media_c:.4f}s")
        print(f"Tempo médio Fluxo:    {media_f:.4f}s")

    # Criar diretório de figuras
    os.makedirs("figuras", exist_ok=True)

    # Gráfico
    plt.figure(figsize=(10,6))
    plt.plot(tamanhos, tempos_classico, label="Modelo Clássico", marker='o')
    plt.plot(tamanhos, tempos_fluxo, label="Modelo Restrito", marker='s')
    plt.xlabel("Tamanho do problema (n x n)")
    plt.ylabel("Tempo médio de execução (s)")
    plt.title("Comparação de tempo médio de execução")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figuras/tempo_execucao_comparacao.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    benchmark_transport()
