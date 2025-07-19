from amplpy import AMPL
from helpf import salvar_dados_ampl_transport, gen_transport_problem, canto_noroeste, calcular_custo_total
import time
import pandas as pd
import numpy as np




"""Cost = [[3, 4, 5],[6, 7, 8]]
Oi = [100, 200]
Dj = [100, 100, 100]"""



"""df = pd.read_csv("transportP - mnist8.csv")"""


"""Oi = np.array(df.iloc[0])
Oi = Oi[Oi != 0]
Dj = np.array(df.iloc[1])
Cost = np.array(df.iloc[2:178, 0:210])"""

Oi, Dj, Cost = gen_transport_problem(100, 100, seed=42)

salvar_dados_ampl_transport(Oi, Dj, Cost, filename="transportDat.dat")

#solucao_noroeste = canto_noroeste(Oi, Dj, Cost)
#custo_noroeste = calcular_custo_total(solucao_noroeste, Cost)

# Cria instância do AMPL
ampl = AMPL()

# Lê o modelo e os dados
ampl.read("transportMod.mod")
ampl.read_data("transportDat.dat")

# Escolhe o solver (pode ser "cbc", "highs", "gurobi", etc.)
ampl.option["solver"] = "gurobi"
ampl.option["show_stats"] = 1

ampl.eval("option times 1;")

"""
m = len(Oi)
n = len(Dj)
x = ampl.get_variable("x")
for i in range(m):
    for j in range(n):
        val = solucao_noroeste[i][j]
        if val is not None:
            ampl_index_i = f"i{i}"
            ampl_index_j = f"j{j}"
            x[ampl_index_i, ampl_index_j].set_value(val)"""

start = time.time()

# Resolve o problema
ampl.solve()

print(f"Tempo total (Python): {time.time() - start:.3f} segundos")

# Acessa e imprime resultado
x = ampl.get_variable("x")
total_cost = ampl.get_objective("Total_Cost")

print("Custo total:", total_cost.value())
"""
for i in ampl.get_set("I"):
    for j in ampl.get_set("J"):
        val = x[i, j].value()
        if val > 0:
            print(f"x[{i},{j}] = {val}")
"""