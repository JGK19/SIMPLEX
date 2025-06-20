def salvar_dados_ampl_transport(Oi, Dj, Cost, filename="dados.dat"):
    m = len(Oi)
    n = len(Dj)
    
    with open(filename, "w") as f:
        # Conjuntos
        f.write("set I := " + " ".join(f"i{i}" for i in range(m)) + ";\n")
        f.write("set J := " + " ".join(f"j{j}" for j in range(n)) + ";\n\n")
        
        # Parâmetro O
        f.write("param O :=\n")
        for i in range(m):
            f.write(f"i{i} {Oi[i]}\n")
        f.write(";\n\n")

        # Parâmetro D
        f.write("param D :=\n")
        for j in range(n):
            f.write(f"j{j} {Dj[j]}\n")
        f.write(";\n\n")

        # Parâmetro C
        f.write("param C : " + " ".join(f"j{j}" for j in range(n)) + " :=\n")
        for i in range(m):
            row = " ".join(str(Cost[i][j]) for j in range(n))
            f.write(f"i{i} {row}\n")
        f.write(";\n")

import numpy as np

max_n = 100
min_n = 1

def gen_transport_problem(oferta, demanda, seed=None):
    if seed is not None:
        np.random.seed(seed)

    custos = np.random.randint(min_n, max_n, size=(oferta, demanda))

    A = np.random.randint(min_n, max_n, size=oferta)
    sum_A = np.sum(A)

    raw_B = np.random.rand(demanda)
    B = (raw_B / raw_B.sum()) * (sum_A * np.random.uniform(0.5, 1.0))
    #B = (raw_B / raw_B.sum()) * (sum_A)
    B = np.floor(B).astype(int)

    return A, B, custos

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