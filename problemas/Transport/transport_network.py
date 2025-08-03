import numpy as np
import pulp
from generator import gen_transport_problem
import pandas as pd

"""Cost = [[3, 4, 5],[6, 7, 8]]
Oi = [100, 200]
Dj = [100, 100, 100]"""

Oi, Dj, Cost = gen_transport_problem(100, 100, seed=42)

"""df = pd.read_csv("problema_201x901_[1,100]_seed42", header=None)

num_ofertas = int(df.iloc[0, 0])
num_demandas = int(df.iloc[0, 1])

Oi = df.iloc[1, :num_ofertas].to_numpy()
Dj = df.iloc[2, :num_demandas].to_numpy()

Cost = df.iloc[3:3+num_ofertas, :num_demandas].to_numpy()"""


O = [_ for _ in range(1, len(Oi)+1)]
D = [_ for _ in range(len(Oi)+1, len(Oi)+len(Dj)+1)]
V = O + D + [0] + [len(O) + len(D) + 1]

indexed_cost = {(o, d): Cost[i][j] for i, o in enumerate(O) for j, d in enumerate(D)}


#m = len(Cost)
#n = len(Cost[1])
m, n = Cost.shape

ofertas = {o: oi for o, oi in zip(O, Oi)}
demandas = {d: dj for d, dj in zip(D, Dj)}



adj_matrix = [(list(None for _ in range(len(O) + len(D) + 2))) for _ in range(len(O) + len(D) + 2)]

for i in range(len(O) + len(D) + 2):
    for j in range(len(O) + len(D) + 2):
        if i == 0 and (j in O):
            adj_matrix[i][j] = pulp.LpVariable(f"edge_{i}_to_{j}", 0, None, pulp.LpInteger)
        elif (i in D) and (j == len(O) + len(D) + 1):
            adj_matrix[i][j] = pulp.LpVariable(f"edge_{i}_to_{j}", 0, None, pulp.LpInteger)
        #elif i == (len(O) + len(D) + 1) and j == 0:
            #adj_matrix[i][j] = pulp.LpVariable(f"edge_{i}_to_{j}", 0, None, pulp.LpInteger)
        elif (i in O) and (j in D):
            adj_matrix[i][j] = pulp.LpVariable(f"edge_{i}_to_{j}", 0, None, pulp.LpInteger)
        else:
            adj_matrix[i][j] = None

prob = pulp.LpProblem("transport_network_flows", pulp.LpMinimize)

# Objective function
eq = 0
for i in range(len(O) + len(D) + 2):
    for j in range(len(O) + len(D) + 2):
        if (i in O) and (j in D):
            eq += adj_matrix[i][j] * indexed_cost[(i,j)]

prob += eq, "total cost for transport"

# Restrição de equulibrio

for i in O + D:
    sum1 = 0
    sum2 = 0
    for j in range(len(O) + len(D) + 2):
        if adj_matrix[i][j] != None:
            sum1 += adj_matrix[i][j]
    for k in range(len(O) + len(D) + 2):
        if adj_matrix[k][i] != None:
            sum2 += adj_matrix[k][i]
    prob += (sum1 - sum2 == 0)

for i in V:
    if adj_matrix[0][i] != None:
        prob += adj_matrix[0][i] <= ofertas[i]

for j in V:
    if adj_matrix[j][len(O) + len(D) + 1] != None:
        prob += adj_matrix[j][len(O) + len(D) + 1] == demandas[j]

#prob += adj_matrix[len(O) + len(D) + 1][0] == sum(Oi)     problemas quando oferta não é igual demanda

prob.writeLP("Transport_networkflow.lp")

prob.solve()

print("Status:", pulp.LpStatus[prob.status])


print("Total Cost = ", pulp.value(prob.objective))