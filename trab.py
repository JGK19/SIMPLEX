import pulp

prob = pulp.LpProblem("Assignment", pulp.LpMinimize)

matrix = [[4,8,1],[10,4,3],[7,5,5]]
n_people = len(matrix)
n_jobs = len(matrix[0])

matrix_var = [(list(None for _ in range(n_jobs))) for _ in range(n_people)]

# creating each variable person i does job j
objective = 0
for i in range(n_people):
    for j in range(n_jobs):
        matrix_var[i][j] = pulp.LpVariable(f"person {i} does job {j}", 0, 1, pulp.LpInteger)

# objective function
eq = matrix_var[0][0] * matrix[0][0]
for i in range(1,n_people):
    for j in range(0,n_jobs):
        eq += matrix_var[i][j] * matrix[i][j]

prob += eq, "total cost for jobs"

# each person does one job
for i in range(n_people):
    res = matrix_var[i][0]
    for j in range(1,n_jobs):
        res += matrix_var[i][j]
    prob += res == 1

#each job is done by one person
for j in range(n_jobs):
    res = matrix_var[0][j]
    for i in range(1,n_people):
        res += matrix_var[i][j]
    prob += res == 1

prob.writeLP("Assignment.lp")

prob.solve()

print("Status:", pulp.LpStatus[prob.status])

for v in prob.variables():
    print(v.name, "=", v.varValue)

print("Total Cost = ", pulp.value(prob.objective))