import numpy as np

def hungarian_algorithm(cost_matrix):
    cost_matrix = cost_matrix.copy()
    n = cost_matrix.shape[0]

    # Passo 1: Subtrair o mínimo de cada linha
    cost_matrix -= cost_matrix.min(axis=1)[:, np.newaxis]

    # Passo 2: Subtrair o mínimo de cada coluna
    cost_matrix -= cost_matrix.min(axis=0)

    # Máscaras
    zero_mask = (cost_matrix == 0)
    row_covered = np.zeros(n, dtype=bool)
    col_covered = np.zeros(n, dtype=bool)
    starred_zeros = np.zeros_like(cost_matrix, dtype=bool)
    primed_zeros = np.zeros_like(cost_matrix, dtype=bool)

    # Estrela zeros não cobertos (passo inicial)
    for i in range(n):
        for j in range(n):
            if zero_mask[i, j] and not row_covered[i] and not col_covered[j]:
                starred_zeros[i, j] = True
                row_covered[i] = True
                col_covered[j] = True

    row_covered[:] = False
    col_covered[:] = False

    def cover_columns_with_starred_zeros():
        for j in range(n):
            if np.any(starred_zeros[:, j]):
                col_covered[j] = True

    cover_columns_with_starred_zeros()

    def find_uncovered_zero():
        for i in range(n):
            for j in range(n):
                if zero_mask[i, j] and not row_covered[i] and not col_covered[j]:
                    return i, j
        return None

    def find_star_in_row(row):
        for j in range(n):
            if starred_zeros[row, j]:
                return j
        return None

    def find_star_in_col(col):
        for i in range(n):
            if starred_zeros[i, col]:
                return i
        return None

    def find_prime_in_row(row):
        for j in range(n):
            if primed_zeros[row, j]:
                return j
        return None

    def augment_path(path):
        for i, j in path:
            if starred_zeros[i, j]:
                starred_zeros[i, j] = False
            else:
                starred_zeros[i, j] = True

    def clear_covers_and_primes():
        row_covered[:] = False
        col_covered[:] = False
        primed_zeros[:, :] = False

    while col_covered.sum() < n:
        while True:
            z = find_uncovered_zero()
            if z is None:
                # Passo de ajuste: achar o menor valor não coberto
                minval = np.min(cost_matrix[~row_covered][:, ~col_covered])
                cost_matrix[~row_covered] -= minval
                cost_matrix[:, col_covered] += minval
                zero_mask = (cost_matrix == 0)
            else:
                i, j = z
                primed_zeros[i, j] = True
                star_col = find_star_in_row(i)
                if star_col is None:
                    # Constrói caminho alternado e aumenta
                    path = [(i, j)]
                    while True:
                        star_row = find_star_in_col(path[-1][1])
                        if star_row is None:
                            break
                        path.append((star_row, path[-1][1]))
                        prime_col = find_prime_in_row(star_row)
                        path.append((star_row, prime_col))
                    augment_path(path)
                    clear_covers_and_primes()
                    cover_columns_with_starred_zeros()
                    break
                else:
                    row_covered[i] = True
                    col_covered[star_col] = False

    result = np.argwhere(starred_zeros)
    return result[:, 0], result[:, 1]
