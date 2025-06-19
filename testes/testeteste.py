import pulp


# Carregar o arquivo MPS
mps_filename = "C:/Users/jgkon/Documents/projects-programa/SIMPLEX/testes/bal8x12.mps"

"""with open(mps_filename) as reader:
    for line in reader:
        print(line[0])"""


lpvars, model = pulp.LpProblem.fromMPS(mps_filename, sense=1)

status = model.solve(pulp.PULP_CBC_CMD(msg=False))

print("Status:", pulp.LpStatus[model.status])

for v in model.variables():
    print(v.name, "=", v.varValue)

print("Total Cost = ", pulp.value(model.objective))
# Prints: 'Optimal'
model.writeLP("problemao.lp")