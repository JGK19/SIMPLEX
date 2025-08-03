import numpy as np
import pandas as pd
import pulp
import argparse
import time
import os
import csv

def solve_flow_problem(filepath):
    df = pd.read_csv(filepath, header=None)

    num_ofertas = int(df.iloc[0, 0])
    num_demandas = int(df.iloc[0, 1])

    Oi = df.iloc[1, :num_ofertas].to_numpy()
    Dj = df.iloc[2, :num_demandas].to_numpy()
    Cost = df.iloc[3:3+num_ofertas, :num_demandas].to_numpy()

    O = [f"O{i}" for i in range(num_ofertas)]
    D = [f"D{j}" for j in range(num_demandas)]
    S = "S"
    T = "T"
    NODES = [S] + O + D + [T]

    costs = {(O[i], D[j]): Cost[i][j] for i in range(num_ofertas) for j in range(num_demandas)}
    ofertas = {O[i]: Oi[i] for i in range(num_ofertas)}
    demandas = {D[j]: Dj[j] for j in range(num_demandas)}

    edges = [(S, o) for o in O] + [(o, d) for o in O for d in D] + [(d, T) for d in D]

    x = pulp.LpVariable.dicts("x", edges, lowBound=0, cat=pulp.LpInteger)

    prob = pulp.LpProblem("Fluxo_em_Rede", pulp.LpMinimize)

    prob += pulp.lpSum(x[(o, d)] * costs[(o, d)] for (o, d) in costs), "Custo_Total"

    for n in O + D:
        inflow = pulp.lpSum(x[(i, n)] for (i, j) in edges if j == n)
        outflow = pulp.lpSum(x[(n, j)] for (i, j) in edges if i == n)
        prob += (inflow - outflow == 0), f"Fluxo_Conservado_{n}"

    for o in O:
        prob += x[(S, o)] <= ofertas[o], f"Capacidade_Oferta_{o}"

    for d in D:
        prob += x[(d, T)] >= demandas[d], f"Atende_Demanda_{d}"

    start = time.time()
    prob.solve()
    end = time.time()

    status = pulp.LpStatus[prob.status]
    custo_total = pulp.value(prob.objective)
    tempo_exec = end - start

    return status, custo_total, tempo_exec

def salvar_resultado(filepath, status, custo, tempo):

    script_path = __file__
    script_name = os.path.basename(script_path)
    codigo = os.path.splitext(script_name)[0]

    os.makedirs("solutions", exist_ok=True)
    csv_path = filepath.replace("instancias/", "solutions/")
    csv_path = csv_path + f"_resultado.csv"

    escrever_cabecalho = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if escrever_cabecalho:
            writer.writerow(["status", "custo", "tempo_execucao", "codigo"])
        writer.writerow([status, custo, f"{tempo:.6f}", codigo])

def main():
    parser = argparse.ArgumentParser(description="Resolve problema de fluxo em rede equivalente ao problema de transporte.")
    parser.add_argument("i", type=int, help="Número de ofertas")
    parser.add_argument("j", type=int, help="Número de demandas")
    parser.add_argument("--min_val", type=int, default=1, help="Valor mínimo (padrão: 1)")
    parser.add_argument("--max_val", type=int, default=100, help="Valor máximo (padrão: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória (padrão: 42)")
    args = parser.parse_args()

    nome_arquivo = f"instancias/problema_{args.i}x{args.j}_[{args.min_val},{args.max_val}]_seed{args.seed}"
    if not os.path.exists(nome_arquivo):
        print(f"Arquivo {nome_arquivo} não encontrado.")
        return

    status, custo, tempo = solve_flow_problem(nome_arquivo)
    salvar_resultado(nome_arquivo, status, custo, tempo)

    print("Problema resolvido.")
    print(f"Status: {status}")
    print(f"Custo total: {custo}")
    print(f"Tempo de execução: {tempo:.6f} segundos")

if __name__ == "__main__":
    main()
