def canto_noroeste(supply, demand, costs):
    n_rows = len(supply)
    n_cols = len(demand)

    allocation = [[0 for _ in range(n_cols)] for _ in range(n_rows)]

    i = 0
    j = 0

    supply = supply.copy()
    demand = demand.copy()

    while i < n_rows and j < n_cols:
        qty = min(supply[i], demand[j])
        allocation[i][j] = qty
        supply[i] -= qty
        demand[j] -= qty

        if supply[i] == 0 and demand[j] == 0:
            if i + 1 < n_rows and j + 1 < n_cols:
                i += 1
                j += 1
            elif j + 1 < n_cols:
                j += 1
            elif i + 1 < n_rows:
                i += 1
            else:
                break
        elif supply[i] == 0:
            i += 1
        else:
            j += 1

    return allocation

def calcular_custo_total(allocation, costs):
    total = 0
    for i in range(len(allocation)):
        for j in range(len(allocation[0])):
            total += allocation[i][j] * costs[i][j]
    return total