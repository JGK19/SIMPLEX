import numpy as np
import pandas as pd
import argparse
import time
import os
import csv


def vogel_method(supply, demand, costs):
    supply = np.array(supply, dtype=float)
    demand = np.array(demand, dtype=float)
    costs = np.array(costs, dtype=float)
    m, n = costs.shape

    allocation = np.zeros((m, n), dtype=float)
    row_done = np.zeros(m, dtype=bool)
    col_done = np.zeros(n, dtype=bool)

    while True:
        active_rows = np.where(~row_done)[0]
        active_cols = np.where(~col_done)[0]
        if active_rows.size == 0 or active_cols.size == 0:
            break

        row_penalties = np.full(m, -np.inf)
        for i in active_rows:
            valid = [j for j in active_cols if demand[j] > 0]
            if not valid:
                continue
            costs_row = costs[i, valid]
            if costs_row.size >= 2:
                two_smallest = np.partition(costs_row, 1)[:2]
                row_penalties[i] = two_smallest[1] - two_smallest[0]
            else:
                row_penalties[i] = costs_row[0]

        col_penalties = np.full(n, -np.inf)
        for j in active_cols:
            valid = [i for i in active_rows if supply[i] > 0]
            if not valid:
                continue
            costs_col = costs[valid, j]
            if costs_col.size >= 2:
                two_smallest = np.partition(costs_col, 1)[:2]
                col_penalties[j] = two_smallest[1] - two_smallest[0]
            else:
                col_penalties[j] = costs_col[0]

        if row_penalties.max() >= col_penalties.max():
            i = np.argmax(row_penalties)
            valid = [j for j in active_cols if demand[j] > 0]
            j = min(valid, key=lambda j_: costs[i, j_])
        else:
            j = np.argmax(col_penalties)
            valid = [i for i in active_rows if supply[i] > 0]
            i = min(valid, key=lambda i_: costs[i_, j])

        qty = min(supply[i], demand[j])
        allocation[i, j] = qty
        supply[i] -= qty
        demand[j] -= qty

        if supply[i] <= 0:
            row_done[i] = True
        if demand[j] <= 0:
            col_done[j] = True

    return allocation

def calcular_custo_total(allocation, costs):
    return int(np.sum(allocation * costs))

def solve_transport_vogel(filepath):
    df = pd.read_csv(filepath, header=None)

    num_ofertas = int(df.iloc[0, 0])
    num_demandas = int(df.iloc[0, 1])

    supply = df.iloc[1, :num_ofertas].to_numpy(dtype=int)
    demand = df.iloc[2, :num_demandas].to_numpy(dtype=int)
    costs = df.iloc[3:3+num_ofertas, :num_demandas].to_numpy(dtype=int)

    assert costs.shape == (num_ofertas, num_demandas), "Erro na dimensão da matriz de custos"

    start = time.time()
    allocation = vogel_method(supply, demand, costs)
    total_cost = calcular_custo_total(allocation, costs)
    end = time.time()

    status = "aproximada"
    tempo_exec = end - start

    return status, total_cost, tempo_exec

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
    parser = argparse.ArgumentParser(description="Resolve problema de transporte pelo método de Vogel.")
    parser.add_argument("i", type=int, help="Número de ofertas")
    parser.add_argument("j", type=int, help="Número de demandas")
    parser.add_argument("--min_val", type=int, default=1, help="Valor mínimo dos custos/ofertas (padrão: 1)")
    parser.add_argument("--max_val", type=int, default=100, help="Valor máximo dos custos/ofertas (padrão: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória (padrão: 42)")
    args = parser.parse_args()

    nome_arquivo = f"instancias/problema_{args.i}x{args.j}_[{args.min_val},{args.max_val}]_seed{args.seed}"
    if not os.path.exists(nome_arquivo):
        print(f"Arquivo {nome_arquivo} não encontrado.")
        return

    status, custo, tempo = solve_transport_vogel(nome_arquivo)
    salvar_resultado(nome_arquivo, status, custo, tempo)

    print("Problema resolvido.")
    print(f"Status: {status}")
    print(f"Custo total: {custo}")
    print(f"Tempo de execução: {tempo:.6f} segundos")

if __name__ == "__main__":
    main()
