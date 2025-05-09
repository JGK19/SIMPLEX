import numpy as np
import pulp

Cost = [[3, 4, 5],[6, 7, 8]]
#Cost = np.array(Cost)

m = len(Cost)
n = len(Cost[1])

Oi = [100, 200]
Dj = [100, 100, 100]

matrix_var = [(list(None for _ in range(n))) for _ in range(m)]

prob = pulp.LpProblem("transport_classical", pulp.LpMinimize)

# Creating each variable i-j
for i in range(m):
    for j in range(n):
        matrix_var[i][j] = pulp.LpVariable(f" edge {i} to  {j}", 0, None, pulp.LpInteger)

# Objective function
eq = None
for i in range(m):
    for j in range(n):
        eq += matrix_var[i][j] * Cost[i][j]

prob += eq, "total cost for transport"

# for each "oferta" node the sum of all edges going out needs to be the total offer in the node
for i in range(m):
    res = matrix_var[i][0]
    for j in range(1,n):
        res += matrix_var[i][j]
    prob += res == Oi[i]
    

#for each "demanda" node the sum of all edges coming in needs to be equals the total demand in the node
for j in range(n):
    res = matrix_var[0][j]
    for i in range(1,m):
        res += matrix_var[i][j]
    prob += res == Dj[j]


prob.writeLP("Transport.lp")

prob.solve()

print("Status:", pulp.LpStatus[prob.status])

for v in prob.variables():
    print(v.name, "=", v.varValue)

print("Total Cost = ", pulp.value(prob.objective))