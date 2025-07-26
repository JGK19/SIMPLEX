import numpy as np
import pandas as pd
import pulp

df = pd.read_csv("problema_201x901_[1,100]_seed42", header=None)

num_ofertas = int(df.iloc[0, 0])
num_demandas = int(df.iloc[0, 1])

Oi = df.iloc[1, :num_ofertas].to_numpy()
Dj = df.iloc[2, :num_demandas].to_numpy()

Cost = df.iloc[3:3+num_ofertas, :num_demandas].to_numpy()

assert Cost.shape == (num_ofertas, num_demandas)

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

prob.solve()

print("Status:", pulp.LpStatus[prob.status])
print("Custo total:", pulp.value(prob.objective))
