import pulp

prob = pulp.LpProblem("The Whiskas Problem", pulp.LpMinimize)

x1 = pulp.LpVariable("ChickenPercent", 0, None, pulp.LpInteger)
x2 = pulp.LpVariable("BeefPercent", 0)

eq = 0.013 * x1 + 0.008 * x2, "Total Cost of Ingredients per can"

prob += eq

# The five constraints are entered
prob += x1 + x2 == 100, "PercentagesSum"
prob += 0.100 * x1 + 0.200 * x2 >= 8.0, "ProteinRequirement"
prob += 0.080 * x1 + 0.100 * x2 >= 6.0, "FatRequirement"
prob += 0.001 * x1 + 0.005 * x2 <= 2.0, "FibreRequirement"
prob += 0.002 * x1 + 0.005 * x2 <= 0.4, "SaltRequirement"

prob.writeLP("WhiskasModel.lp")

prob.solve()

print("Status:", pulp.LpStatus[prob.status])

for v in prob.variables():
    print(v.name, "=", v.varValue)

print("Total Cost of Ingredients per can = ", pulp.value(prob.objective))
