import numpy as np
import pandas as pd
import pulp

df = pd.read_csv("problema_201x901_[1,100]_seed42", header=None)

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

edges = []

edges += [(S, o) for o in O]
edges += [(o, d) for o in O for d in D]
edges += [(d, T) for d in D]

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
    prob += x[(d, T)] == demandas[d], f"Atende_Demanda_{d}"

prob.solve()

print("Status:", pulp.LpStatus[prob.status])
print("Custo total:", pulp.value(prob.objective))
