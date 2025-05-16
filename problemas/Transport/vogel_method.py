import numpy as np

def vogel_method(supply, demand, costs):
    """
    Método de Aproximação de Vogel otimizado com NumPy.

    Parâmetros:
    - supply: lista ou array 1D de ofertas (comprimento m)
    - demand: lista ou array 1D de demandas (comprimento n)
    - costs: lista de listas ou array 2D de custos (m x n)

    Retorna:
    - allocation: array 2D (m x n) com as quantidades alocadas
    """
    # Converter para arrays NumPy
    supply = np.array(supply, dtype=float)
    demand = np.array(demand, dtype=float)
    costs = np.array(costs, dtype=float)
    m, n = costs.shape

    # Matriz de alocação
    allocation = np.zeros((m, n), dtype=float)

    # Flags para linhas/colunas concluídas
    row_done = np.zeros(m, dtype=bool)
    col_done = np.zeros(n, dtype=bool)

    while True:
        # Identificar linhas e colunas ativas
        active_rows = np.where(~row_done)[0]
        active_cols = np.where(~col_done)[0]
        if active_rows.size == 0 or active_cols.size == 0:
            break

        # Calcular penalidades de linhas
        row_penalties = np.full(m, -np.inf)
        for i in active_rows:
            # Custos nas colunas ainda ativas com demanda > 0
            valid = [j for j in active_cols if demand[j] > 0]
            if not valid:
                continue
            costs_row = costs[i, valid]
            if costs_row.size >= 2:
                two_smallest = np.partition(costs_row, 1)[:2]
                row_penalties[i] = two_smallest[1] - two_smallest[0]
            else:
                row_penalties[i] = costs_row[0]

        # Calcular penalidades de colunas
        col_penalties = np.full(n, -np.inf)
        for j in active_cols:
            valid = [i for i in active_rows if supply[i] > 0]
            if not valid:
                continue
            costs_col = costs[valid, j]
            if costs_col.size >= 2:
                two_smallest = np.partition(costs_col, 1)[:2]
                col_penalties[j] = two_smallest[1] - two_smallest[0]
            else:
                col_penalties[j] = costs_col[0]

        # Selecionar maior penalidade
        if row_penalties.max() >= col_penalties.max():
            # escolher linha
            i = np.argmax(row_penalties)
            # coluna de menor custo nessa linha
            valid = [j for j in active_cols if demand[j] > 0]
            j = min(valid, key=lambda j_: costs[i, j_])
        else:
            # escolher coluna
            j = np.argmax(col_penalties)
            # linha de menor custo nessa coluna
            valid = [i for i in active_rows if supply[i] > 0]
            i = min(valid, key=lambda i_: costs[i_, j])

        # Alocar o máximo possível
        qty = min(supply[i], demand[j])
        allocation[i, j] = qty
        supply[i] -= qty
        demand[j] -= qty

        # Marcar linha/coluna se zeradas
        if supply[i] <= 0:
            row_done[i] = True
        if demand[j] <= 0:
            col_done[j] = True

    return allocation