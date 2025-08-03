import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# 1. Criando o problema de maximização
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize
creator.create("Individual", list, fitness=creator.FitnessMax)

# 2. Função de avaliação
def eval_func(individual):
    x = individual[0]
    return x * np.sin(10 * np.pi * x) + 1.0,  # Retorna uma tupla!

def mutate_and_clip(individual, mu=0, sigma=0.1, indpb=0.2):
    tools.mutGaussian(individual, mu=mu, sigma=sigma, indpb=indpb)
    for i in range(len(individual)):
        individual[i] = max(0.0, min(1.0, individual[i]))  # Limita entre 0 e 1
    return individual,

def mate_and_clip(ind1, ind2, alpha=0.5):
    tools.cxBlend(ind1, ind2, alpha)
    for i in range(len(ind1)):
        ind1[i] = max(0.0, min(1.0, ind1[i]))
        ind2[i] = max(0.0, min(1.0, ind2[i]))
    return ind1, ind2

# 3. Registro dos operadores genéticos
toolbox = base.Toolbox()

# Geração de genes aleatórios entre 0 e 1
toolbox.register("attr_float", random.uniform, 0, 1)

# Um indivíduo é uma lista com 1 gene
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=1)

# População é uma lista de indivíduos
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Avaliação
toolbox.register("evaluate", eval_func)

# Operadores genéticos
toolbox.register("mate", mate_and_clip)  # Crossover
toolbox.register("mutate", mutate_and_clip) # mutação
toolbox.register("select", tools.selTournament, tournsize=3)  # Seleção

# 4. Parâmetros do algoritmo
POP_SIZE = 50
N_GEN = 100
CXPB = 0.7
MUTPB = 0.2

# 5. Inicializa população
pop = toolbox.population(n=POP_SIZE)

# 6. Estatísticas
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

# 7. Executa o algoritmo
pop, logbook = algorithms.eaSimple(pop, toolbox,
                                   cxpb=CXPB,
                                   mutpb=MUTPB,
                                   ngen=N_GEN,
                                   stats=stats,
                                   verbose=True)

# 8. Resultados
best = tools.selBest(pop, k=1)[0]
print(f"Melhor indivíduo: {best}")
print(f"Fitness: {best.fitness.values[0]}")

# 9. Plotando a convergência
gen = logbook.select("gen")
maxs = logbook.select("max")
avgs = logbook.select("avg")

plt.plot(gen, maxs, label='Máximo')
plt.plot(gen, avgs, label='Média')
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.legend()
plt.grid()
plt.title("Convergência do Algoritmo Genético")
plt.show()
