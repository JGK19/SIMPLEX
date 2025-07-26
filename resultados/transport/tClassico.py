import numpy as np
import pandas as pd
import pulp
import argparse
import time
import os

def solve_transport_problem(filepath):
    df = pd.read_csv(filepath, header=None)

    num_ofertas = int(df.iloc[0, 0])
    num_demandas = int(df.iloc[0, 1])

    Oi = df.iloc[1, :num_ofertas].to_numpy()
    Dj = df.iloc[2, :num_demandas].to_numpy()
    Cost = df.iloc[3:3+num_ofertas, :num_demandas].to_numpy()

    assert Cost.shape == (num_ofertas, num_demandas), "Erro na dimensão da matriz de custos"

    prob = pulp.LpProblem("Problema_Transporte", pulp.LpMinimize)

    x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat=pulp.LpInteger)
          for j in range(num_demandas)] for i in range(num_ofertas)]

    prob += pulp.lpSum(x[i][j] * Cost[i][j]
                       for i in range(num_ofertas)
                       for j in range(num_demandas)), "Custo_Total"

    for i in range(num_ofertas):
        prob += pulp.lpSum(x[i][j] for j in range(num_demandas)) <= Oi[i], f"Oferta_{i}"

    for j in range(num_demandas):
        prob += pulp.lpSum(x[i][j] for i in range(num_ofertas)) >= Dj[j], f"Demanda_{j}"

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

    result_path = filepath.replace("instancias/", "solutions/")
    result_path = result_path + f"_resultado_{script_name}.txt"
    with open(result_path, "w") as f:
        f.write(f"Status: {status}\n")
        f.write(f"Custo total: {custo}\n")
        f.write(f"Tempo de execução: {tempo:.6f} segundos\n")

def main():
    parser = argparse.ArgumentParser(description="Resolve problema de transporte dado por parâmetros.")
    parser.add_argument("i", type=int, help="Número de ofertas")
    parser.add_argument("j", type=int, help="Número de demandas")
    parser.add_argument("--min_val", type=int, default=1, help="Valor mínimo dos custos/ofertas (padrão: 0)")
    parser.add_argument("--max_val", type=int, default=100, help="Valor máximo dos custos/ofertas (padrão: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória (padrão: 42)")
    args = parser.parse_args()

    nome_arquivo = f"instancias/problema_{args.i}x{args.j}_[{args.min_val},{args.max_val}]_seed{args.seed}"
    if not os.path.exists(nome_arquivo):
        print(f"Arquivo {nome_arquivo} não encontrado.")
        return

    status, custo, tempo = solve_transport_problem(nome_arquivo)
    salvar_resultado(nome_arquivo, status, custo, tempo)

    print("Problema resolvido.")
    print(f"Status: {status}")
    print(f"Custo total: {custo}")
    print(f"Tempo de execução: {tempo:.6f} segundos")

if __name__ == "__main__":
    main()
