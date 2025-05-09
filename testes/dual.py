import pulp
import numpy as np

# primal problem
prob = pulp.LpProblem("test_primal", pulp.LpMinimize)

# primal costs
cost = np.array([2, 3, 5, 2, 3])

# primal A matrix
A = np.array([[1, 1, 2, 1, 3], [2, -2, 3, 1, 1]])

# primal left side 
b = np.array([4, 3])

# primal variables
X = []
for i in range(len(cost)):
    X.append(pulp.LpVariable(f"var_{i}", 0, None, pulp.LpContinuous))

X = np.array(X)

# objetive function for primal problem cost (dot) X
objective = np.dot(cost, X)
prob += objective

# restrições Ax = b
subject = np.dot(A, X)
for r, bi in zip(subject, b):
    prob += r >= bi

prob.writeLP("primal_test.lp")

prob.solve()

print("Status:", pulp.LpStatus[prob.status])

for v in prob.variables():
    print(v.name, "=", v.varValue)

print("Total Cost = ", pulp.value(prob.objective))

print("////////////////////////////////////////////////////////////////////////////////////\n/////////////////////////////////////////////")

# dual problem
prob_dual = pulp.LpProblem("test_dual", pulp.LpMaximize)

# custos e restrições do problema dual
dual_cost = b[:]
dual_b = cost[:]
dual_A = A.transpose()

# dual variables
W = []
for i in range(len(dual_cost)):
    W.append(pulp.LpVariable(f"var_{i}", 0, None, pulp.LpContinuous))


dual_objective = np.dot(dual_cost, W)
prob_dual += dual_objective

subject_dual = np.dot(dual_A, W) #  W (dot) A == A^T (dot) W

# primal: >=
# dual: <=
for r, bi in zip(subject_dual, dual_b):
    prob_dual += r <= bi

prob_dual.writeLP("dual_test.lp")


prob_dual.solve()

print("Status:", pulp.LpStatus[prob_dual.status])

for v in prob_dual.variables():
    print(v.name, "=", v.varValue)

print("Total Cost = ", pulp.value(prob_dual.objective))


