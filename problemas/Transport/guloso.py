import numpy as np
import pandas as pd
from generator import gen_transport_problem

def main():
    """
    Cost = [[3, 4, 5],[6, 7, 8]]
    Oi = [100, 200]
    Dj = [100, 100, 100]
    """
    Oi, Dj, Cost = gen_transport_problem(100, 100, seed=42)
    sol = metodo_guloso(Oi, Dj, Cost)
    print(sol)
    print(calcular_custo_total(sol, Cost))
    print(verifica(Oi, Dj, sol))


def metodo_guloso(supply, demand, costs):
    n_rows = len(supply)
    n_cols = len(demand)

    allocation = np.zeros((n_rows, n_cols), dtype=int)
    supply = supply.copy()
    demand = demand.copy()

    positions = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    positions.sort(key=lambda pos: costs[pos[0]][pos[1]])

    for i, j in positions:
        if supply[i] == 0 or demand[j] == 0:
            continue
        qty = min(supply[i], demand[j])
        allocation[i, j] = qty
        supply[i] -= qty
        demand[j] -= qty

    return allocation

def calcular_custo_total(allocation, costs):
    return int(np.sum(allocation * costs))

def verifica(supply, demand, allocation):
    """
    Verifica se uma solução de transporte desbalanceada é válida.
    Considera que oferta total <= demanda total.
    
    Parâmetros:
    - supply: lista ou array 1D de ofertas (comprimento m)
    - demand: lista ou array 1D de demandas (comprimento n)
    - allocation: matriz (m x n) com alocações feitas

    Retorna:
    - True se a solução é válida, False caso contrário.
    """
    supply = np.array(supply)
    demand = np.array(demand)
    allocation = np.array(allocation)

    # Verifica dimensões
    if allocation.shape != (len(supply), len(demand)):
        print("Dimensões inválidas na matriz de alocação.")
        return False

    # Verifica não-negatividade
    if np.any(allocation < 0):
        print("A matriz de alocação contém valores negativos.")
        return False

    # Verifica se a oferta não foi ultrapassada
    oferta_usada = allocation.sum(axis=1)
    if not np.all(oferta_usada <= supply):
        print("Alguma oferta foi ultrapassada.")
        return False

    # Verifica se a demanda não foi ultrapassada
    demanda_atendida = allocation.sum(axis=0)
    if not np.all(demanda_atendida <= demand):
        print("Alguma demanda foi ultrapassada.")
        return False

    return True



main()