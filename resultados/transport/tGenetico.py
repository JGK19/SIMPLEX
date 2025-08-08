import random
import numpy as np
import pandas as pd
import argparse
import time
import os
import csv
from deap import base, creator, tools, algorithms

# -----------------------------
# LEITURA DO PROBLEMA A PARTIR DE ARQUIVO
# -----------------------------
def ler_instancia(filepath):
    df = pd.read_csv(filepath, header=None)

    num_ofertas = int(df.iloc[0, 0])
    num_demandas = int(df.iloc[0, 1])

    supply = df.iloc[1, :num_ofertas].to_numpy(dtype=int)
    demand = df.iloc[2, :num_demandas].to_numpy(dtype=int)
    costs = df.iloc[3:3+num_ofertas, :num_demandas].to_numpy(dtype=int)

    assert costs.shape == (num_ofertas, num_demandas), "Erro na dimensão da matriz de custos"

    return supply, demand, costs

# -----------------------------
# VERIFICAÇÃO DE VIABILIDADE
# -----------------------------
def verifica(supply, demand, allocation):
    supply = np.array(supply)
    demand = np.array(demand)
    allocation = np.array(allocation)

    if allocation.shape != (len(supply), len(demand)):
        return False
    if np.any(allocation < 0):
        return False

    oferta_usada = allocation.sum(axis=1)
    if not np.all(oferta_usada <= supply):
        return False

    demanda_atendida = allocation.sum(axis=0)
    if not np.all(demanda_atendida >= demand):
        return False

    return True

# -----------------------------
# CORREÇÃO (REPARO) DA SOLUÇÃO
# -----------------------------
def reparar_solucao(solucao_raw, a, b, c):
    m, n = len(a), len(b)
    solucao = np.array(solucao_raw).reshape((m, n)).astype(float)
    solucao = np.maximum(np.floor(solucao), 0).astype(int)

    for i in range(m):
        while solucao[i].sum() > a[i]:
            col_indices = np.where(solucao[i] > 0)[0]
            if len(col_indices) == 0:
                break
            custos_linha = c[i, col_indices]
            idx_max_custo = col_indices[np.argmax(custos_linha)]
            solucao[i, idx_max_custo] = 0

    for j in range(n):
        while solucao[:, j].sum() > b[j]:
            row_indices = np.where(solucao[:, j] > 0)[0]
            if len(row_indices) == 0:
                break
            custos_coluna = c[row_indices, j]
            idx_max_custo = row_indices[np.argmax(custos_coluna)]
            solucao[idx_max_custo, j] = 0

    oferta_rest = a - solucao.sum(axis=1)
    demanda_rest = b - solucao.sum(axis=0)
    oferta_rest = np.maximum(oferta_rest, 0)
    demanda_rest = np.maximum(demanda_rest, 0)

    def metodo_guloso(supply, demand, costs):
        n_rows, n_cols = len(supply), len(demand)
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

    complemento = metodo_guloso(oferta_rest, demanda_rest, c)
    return solucao + complemento

# -----------------------------
# AVALIAÇÃO
# -----------------------------
def avaliar(individuo):
    solucao = reparar_solucao(individuo, a, b, c)
    custo_total = np.sum(solucao * c)
    return (custo_total,)

# -----------------------------
# EXECUÇÃO GA
# -----------------------------
def executar_ga(supply, demand, costs, pop_size=100, ngen=100, cxpb=0.8, mutpb=0.2):
    global a, b, c, m, n, N
    a, b, c = np.array(supply), np.array(demand), np.array(costs)
    m, n = len(a), len(b)
    N = m * n

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, max(max(a), max(b)))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, N)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", avaliar)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    start = time.time()
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                   ngen=ngen, stats=stats, halloffame=hof,
                                   verbose=True)
    end = time.time()

    melhor = reparar_solucao(hof[0], a, b, c)
    custo_total = np.sum(melhor * c)
    status = "aproximada" if verifica(a, b, melhor) else "falha"
    return status, custo_total, end - start

# -----------------------------
# SALVAR RESULTADO
# -----------------------------
def salvar_resultado(filepath, status, custo, tempo):
    script_path = __file__
    script_name = os.path.basename(script_path)
    codigo = os.path.splitext(script_name)[0]
    os.makedirs("solutions", exist_ok=True)
    csv_path = filepath.replace("instancias/", "solutions/") + f"_resultado.csv"
    escrever_cabecalho = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if escrever_cabecalho:
            writer.writerow(["status", "custo", "tempo_execucao", "codigo"])
        writer.writerow([status, custo, f"{tempo:.6f}", codigo])

# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Resolve problema de transporte com algoritmo genético.")
    parser.add_argument("i", type=int, help="Número de ofertas")
    parser.add_argument("j", type=int, help="Número de demandas")
    parser.add_argument("--min_val", type=int, default=1)
    parser.add_argument("--max_val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nome_arquivo = f"instancias/problema_{args.i}x{args.j}_[{args.min_val},{args.max_val}]_seed{args.seed}"
    if not os.path.exists(nome_arquivo):
        print(f"Arquivo {nome_arquivo} não encontrado.")
        return

    supply, demand, costs = ler_instancia(nome_arquivo)
    status, custo, tempo = executar_ga(supply, demand, costs, pop_size=10,ngen=5)
    salvar_resultado(nome_arquivo, status, custo, tempo)

    print("Problema resolvido.")
    print(f"Status: {status}")
    print(f"Custo total: {custo}")
    print(f"Tempo de execução: {tempo:.6f} segundos")

if __name__ == "__main__":
    main()
