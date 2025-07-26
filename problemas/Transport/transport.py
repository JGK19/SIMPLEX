import numpy as np
import pulp
from generator import gen_transport_problem, gen_transport_problem_notint
from canto_noroeste import canto_noroeste, calcular_custo_total
from vogel_method import vogel_method
import pandas as pd


Oi, Dj, Cost = gen_transport_problem(202, 902, seed=42)
m = len(Cost)
n = len(Cost[1])


"""df = pd.read_csv("problema_201x901_[1,100]_seed42", header=None)

num_ofertas = int(df.iloc[0, 0])
num_demandas = int(df.iloc[0, 1])

Oi = df.iloc[1, :num_ofertas].to_numpy()
Dj = df.iloc[2, :num_demandas].to_numpy()

Cost = df.iloc[3:3+num_ofertas, :num_demandas].to_numpy()
m, n = Cost.shape"""

matrix_var = [(list(None for _ in range(n))) for _ in range(m)]

prob = pulp.LpProblem("transport_classical", pulp.LpMinimize)

# Creating each variable i-j
for i in range(m):
    for j in range(n):
        matrix_var[i][j] = pulp.LpVariable(f"edge_{i}_to_{j}", 0, None, pulp.LpInteger)

# Objective function
eq = 0
for i in range(m):
    for j in range(n):
        eq += matrix_var[i][j] * Cost[i][j]

prob += eq, "total cost for transport"


# for each "oferta" node the sum of all edges going out needs to be the total offer in the node
for i in range(m):
    res = matrix_var[i][0]
    for j in range(1,n):
        res += matrix_var[i][j]
    prob += res <= Oi[i] #importante menor ou igual, para permitir sobras
    

#for each "demanda" node the sum of all edges coming in needs to be equals the total demand in the node
for j in range(n):
    res = matrix_var[0][j]
    for i in range(1,m):
        res += matrix_var[i][j]
    prob += res >= Dj[j]


prob.writeLP("Transport.lp")

prob.solve(pulp.PULP_CBC_CMD(msg=True))

print("Status:", pulp.LpStatus[prob.status])

"""for v in prob.variables():
    print(v.name, "=", v.varValue)"""

print("Total Cost = ", pulp.value(prob.objective))