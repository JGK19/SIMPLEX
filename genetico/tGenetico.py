import random
import numpy as np
from deap import base, creator, tools, algorithms

from repair import reparar_solucao
# -----------------------------
# DEFINIÇÃO DO PROBLEMA
# -----------------------------

Cost = [[3, 4, 5],[6, 7, 8]]
oferta = [100, 200]
demanda = [100, 100, 100]



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

def contar_violacoes(supply, demand, allocation_raw):
    allocation = np.array(allocation_raw).reshape((m, n)).astype(float)
    supply = np.array(supply)
    demand = np.array(demand)
    allocation = np.array(allocation)

    # Verifica violações de oferta: soma por linha > oferta
    oferta_usada = allocation.sum(axis=1)
    violacoes_oferta = np.sum(oferta_usada > supply)

    # Verifica violações de demanda: soma por coluna < demanda
    demanda_atendida = allocation.sum(axis=0)
    violacoes_demanda = np.sum(demanda_atendida < demand)

    total_violacoes = violacoes_oferta + violacoes_demanda
    return total_violacoes, allocation

# -----------------------------
# AVALIAÇÃO
# -----------------------------
def avaliar(individuo):
    numero, solucao = contar_violacoes(a, b, individuo)
    custo_total = np.sum(solucao * c)
    custo_total += max(N, 10**6) * max(numero, 0)
    return (custo_total,)

def mutar_individuo(ind, mu=0, sigma=5, indpb=0.2):
    tools.mutGaussian(ind, mu, sigma, indpb)
    for i in range(len(ind)):
        ind[i] = max(0, int(ind[i]))  # força não-negativo e inteiro
    return (ind,)

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
toolbox.register("mutate", mutar_individuo)
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
    vio, melhor = contar_violacoes(a, b, hof[0])
    melhor2 = reparar_solucao(hof[0], a, b, c)
    print("\nMelhor solução encontrada (matriz de transporte):")
    print(melhor)
    print(melhor2)
    print("Custo total:", hof[0].fitness.values[0])
    print("Número de violações:", vio)
    print("Custo total2:", np.sum(melhor2 * c))

# Executar
if __name__ == "__main__":
    ga_transporte(pop_size=50,ngen=100,mutpb=0.2)
