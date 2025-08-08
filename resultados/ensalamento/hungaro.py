import os
import argparse
import time
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from helpf import ler_instancia_csv, salvar_resultado

def resolver_com_hungaro(caminho_csv, nome_instancia):
    # Lê os dados da instância
    disciplinas, salas, N, C, D = ler_instancia_csv(caminho_csv)

    n = len(disciplinas)

    # Converte dados para np.array
    N = np.array(N)
    C = np.array(C)
    D = np.array(D)

    origem = [i for i in range(0, n)]

    inicio = time.time()
    # Calcula matriz de custo penalizada
    Cp = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if N[i] <= C[j]:
                Cp[i, j] = D[origem[i], j] * N[i]
            else:
                Cp[i, j] = 10**6  # penalização alta

    # Resolve com método húngaro
    inicio = time.time()
    row_ind, col_ind = linear_sum_assignment(Cp)
    fim = time.time()

    tempo_execucao = fim - inicio
    real_cost = sum(D[origem[i], col_ind[idx]] * N[i] for idx, i in enumerate(row_ind))

    status = "solved"
    codigo = f"hungaro"

    salvar_resultado(status, real_cost, tempo_execucao, codigo, nome_instancia)

    print("\nStatus:", status)
    print("Deslocamento total:", real_cost)
    print("Tempo de execução:", tempo_execucao)

def main():
    parser = argparse.ArgumentParser(description="Resolve problema de ensalamento com algoritmo de Kuhn-Munkres (Hungarian method).")
    parser.add_argument("-i", type=int, help="Número de disciplinas")
    parser.add_argument("-j", type=int, help="Número de salas")
    parser.add_argument("--min_alunos", type=int, default=10)
    parser.add_argument("--max_alunos", type=int, default=90)
    parser.add_argument("--min_cap", type=int, default=30)
    parser.add_argument("--max_cap", type=int, default=90)
    parser.add_argument("--min_d", type=int, default=1)
    parser.add_argument("--max_d", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folder", type=str, default="instancias")

    args = parser.parse_args()

    nome_arquivo = f"ensalamento_D{args.i}_S{args.j}_[{args.min_alunos},{args.max_alunos}]_"
    nome_arquivo += f"[{args.min_cap},{args.max_cap}]_[{args.min_d},{args.max_d}]_seed{args.seed}.csv"

    caminho_csv = os.path.join(args.folder, nome_arquivo)

    if not os.path.exists(caminho_csv):
        print(f"Arquivo {caminho_csv} não encontrado.")
        return

    nome_instancia = f"D{args.i}_S{args.j}_[{args.min_alunos},{args.max_alunos}]_"
    nome_instancia += f"[{args.min_cap},{args.max_cap}]_[{args.min_d},{args.max_d}]_seed{args.seed}"

    resolver_com_hungaro(caminho_csv, nome_instancia)

if __name__ == "__main__":
    main()
