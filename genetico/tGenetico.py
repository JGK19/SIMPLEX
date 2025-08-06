import random
import numpy as np
from deap import base, creator, tools, algorithms

# -----------------------------
# DEFINIÇÃO DO PROBLEMA
# -----------------------------
"""
Cost = [[3, 4, 5],[6, 7, 8]]
oferta = [100, 200]
demanda = [100, 100, 100]
"""


def gen_transport_problem(oferta, demanda, seed=None, min_val=1, max_val=100):
    if seed is not None:
        np.random.seed(seed)

    custos = np.random.randint(min_val, max_val, size=(oferta, demanda))

    A = np.random.randint(min_val, max_val, size=oferta)
    sum_A = np.sum(A)

    raw_B = np.random.rand(demanda)
    B = (raw_B / raw_B.sum()) * (sum_A * np.random.uniform(0.5, 1.0))
    #B = (raw_B / raw_B.sum()) * (sum_A)
    B = np.floor(B).astype(int)

    return A, B, custos

oferta, demanda, Cost = gen_transport_problem(100, 100, seed=42)

a, b, c = (np.array(oferta), np.array(demanda), np.array(Cost))

m, n = len(a), len(b)
N = m * n  # tamanho do cromossomo

# -----------------------------
# CORREÇÃO (REPARO) DA SOLUÇÃO
# -----------------------------
def reparar_solucao(solucao_raw, a, b):
    solucao = np.array(solucao_raw).reshape((m, n)).astype(float)
    # Garante não-negatividade e inteiros
    solucao = np.maximum(np.floor(solucao), 0).astype(int)

    for i in range(m):
        while solucao[i].sum() > a[i]:
            # Pega índices das colunas com valores > 0
            col_indices = np.where(solucao[i] > 0)[0]

            if len(col_indices) == 0:
                break  # nada para zerar, sai do loop

            # Critério: escolher a célula de maior custo para zerar
            # (supondo que c seja matriz de custos)
            custos_linha = c[i, col_indices]
            idx_max_custo = col_indices[np.argmax(custos_linha)]

            # Zera essa célula
            solucao[i, idx_max_custo] = 0

    # --- Etapa 1B: respeitar restrições de demanda (colunas) ---
    for j in range(n):
        while solucao[:, j].sum() > b[j]:
            # Pega índices das linhas com valores > 0 nessa coluna
            row_indices = np.where(solucao[:, j] > 0)[0]

            if len(row_indices) == 0:
                break  # nada para zerar

            # Critério: zere a célula de maior custo naquela coluna
            custos_coluna = c[row_indices, j]
            idx_max_custo = row_indices[np.argmax(custos_coluna)]

            # Zera a célula
            solucao[idx_max_custo, j] = 0

    # Atualiza oferta e demanda restantes com base na solução ajustada
    oferta_rest = a - solucao.sum(axis=1)
    demanda_rest = b - solucao.sum(axis=0)

    oferta_rest = np.maximum(oferta_rest, 0)
    demanda_rest = np.maximum(demanda_rest, 0)

    # --- Etapa 2: completar com método guloso a partir da solução parcial ---
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

    # Executa o método guloso para o restante
    complemento = metodo_guloso(oferta_rest, demanda_rest, c)

    # Soma a parte inicial com a parte gulosa
    solucao_final = solucao + complemento

    return solucao_final


# -----------------------------
# AVALIAÇÃO
# -----------------------------
def avaliar(individuo):
    solucao = reparar_solucao(individuo, a, b)
    custo_total = np.sum(solucao * c)
    return (custo_total,)

# -----------------------------
# CONFIGURAÇÃO DO DEAP
# -----------------------------
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, max(max(a), max(b)))
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_int, N)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", avaliar)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# -----------------------------
# EXECUÇÃO DO ALGORITMO
# -----------------------------
def ga_transporte(pop_size=100, ngen=100, cxpb=0.8, mutpb=0.2):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                   ngen=ngen, stats=stats, halloffame=hof,
                                   verbose=True)
    melhor = reparar_solucao(hof[0], a, b)
    print("\nMelhor solução encontrada (matriz de transporte):")
    print(melhor)
    print("Custo total:", hof[0].fitness.values[0])

# Executar
if __name__ == "__main__":
    ga_transporte()
