import numpy as np

def reparar_solucao(solucao_raw, a, b, c):
    m, n = len(a), len(b)
    solucao = np.array(solucao_raw).reshape((m, n)).astype(float)
    solucao = np.maximum(np.floor(solucao), 0).astype(int)

    for i in range(m):
        while solucao[i].sum() > a[i]:
            col_indices = np.where(solucao[i] > 0)[0]
            if len(col_indices) == 0:
                break
            custos_linha = c[i, col_indices]
            idx_max_custo = col_indices[np.argmax(custos_linha)]
            solucao[i, idx_max_custo] = 0

    for j in range(n):
        while solucao[:, j].sum() > b[j]:
            row_indices = np.where(solucao[:, j] > 0)[0]
            if len(row_indices) == 0:
                break
            custos_coluna = c[row_indices, j]
            idx_max_custo = row_indices[np.argmax(custos_coluna)]
            solucao[idx_max_custo, j] = 0

    oferta_rest = a - solucao.sum(axis=1)
    demanda_rest = b - solucao.sum(axis=0)
    oferta_rest = np.maximum(oferta_rest, 0)
    demanda_rest = np.maximum(demanda_rest, 0)

    def metodo_guloso(supply, demand, costs):
        n_rows, n_cols = len(supply), len(demand)
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

    complemento = metodo_guloso(oferta_rest, demanda_rest, c)
    return solucao + complemento
