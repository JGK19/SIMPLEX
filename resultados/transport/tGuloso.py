import numpy as np
import pandas as pd
import argparse
import time
import os
import csv

def verifica(supply, demand, allocation):
    """
    Verifica se uma solução de transporte desbalanceada é válida.
    Considera que oferta total <= demanda total.
    
    Parâmetros:
    - supply: lista ou array 1D de ofertas (comprimento m)
    - demand: lista ou array 1D de demandas (comprimento n)
    - allocation: matriz (m x n) com alocações feitas

    Retorna:
    - True se a solução é válida, False caso contrário.
    """
    supply = np.array(supply)
    demand = np.array(demand)
    allocation = np.array(allocation)

    # Verifica dimensões
    if allocation.shape != (len(supply), len(demand)):
        print("Dimensões inválidas na matriz de alocação.")
        return False

    # Verifica não-negatividade
    if np.any(allocation < 0):
        print("A matriz de alocação contém valores negativos.")
        return False

    # Verifica se a oferta não foi ultrapassada
    oferta_usada = allocation.sum(axis=1)
    if not np.all(oferta_usada <= supply):
        print("Alguma oferta foi ultrapassada.")
        return False

    # Verifica se a demanda não foi ultrapassada
    demanda_atendida = allocation.sum(axis=0)
    if not np.all(demanda_atendida <= demand):
        print("Alguma demanda foi ultrapassada.")
        return False

    return True

def metodo_guloso(supply, demand, costs):
    n_rows = len(supply)
    n_cols = len(demand)

    allocation = np.zeros((n_rows, n_cols), dtype=int)
    supply = supply.copy()
    demand = demand.copy()

    positions = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    positions.sort(key=lambda pos: costs[pos[0]][pos[1]])

    for i, j in positions:
        if supply[i] == 0 or demand[j] == 0:
            continue
        qty = min(supply[i], demand[j])
        allocation[i, j] = qty
        supply[i] -= qty
        demand[j] -= qty

    return allocation

def calcular_custo_total(allocation, costs):
    return int(np.sum(allocation * costs))

def solve_transport_greedy(filepath):
    df = pd.read_csv(filepath, header=None)

    num_ofertas = int(df.iloc[0, 0])
    num_demandas = int(df.iloc[0, 1])

    supply = df.iloc[1, :num_ofertas].to_numpy(dtype=int)
    demand = df.iloc[2, :num_demandas].to_numpy(dtype=int)
    costs = df.iloc[3:3+num_ofertas, :num_demandas].to_numpy(dtype=int)

    assert costs.shape == (num_ofertas, num_demandas), "Erro na dimensão da matriz de custos"

    start = time.time()
    allocation = metodo_guloso(supply, demand, costs)
    total_cost = calcular_custo_total(allocation, costs)
    end = time.time()

    status = "aproximada"
    tempo_exec = end - start
    

    print(f"SOLUÇÂO VALIDA {verifica(supply, demand, allocation)}")
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
    parser = argparse.ArgumentParser(description="Resolve problema de transporte por um método guloso.")
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

    status, custo, tempo = solve_transport_greedy(nome_arquivo)
    salvar_resultado(nome_arquivo, status, custo, tempo)

    print("Problema resolvido.")
    print(f"Status: {status}")
    print(f"Custo total: {custo}")
    print(f"Tempo de execução: {tempo:.6f} segundos")

if __name__ == "__main__":
    main()
