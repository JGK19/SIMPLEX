import numpy as np
import pandas as pd

def gen_transport_problem(oferta, demanda, seed=None, min_val=1, max_val=100):
    if seed is not None:
        np.random.seed(seed)

    custos = np.random.randint(min_val, max_val + 1, size=(oferta, demanda))

    A = np.random.randint(min_val, max_val + 1, size=oferta)
    sum_A = np.sum(A)

    raw_B = np.random.rand(demanda)
    B = (raw_B / raw_B.sum()) * (sum_A * np.random.uniform(0.5, 1.0))
    B = np.floor(B).astype(int)

    return A, B, custos

def save_transport_problem_to_csv(filename, oferta, demanda, seed=None, min_val=1, max_val=100):
    A, B, C = gen_transport_problem(oferta, demanda, seed, min_val, max_val)

    rows = []

    first_row = [oferta, demanda] + [""] * (max(oferta, demanda) - 2)
    rows.append(first_row)

    oferta_row = list(A) + [""] * (max(oferta, demanda) - len(A))
    rows.append(oferta_row)

    demanda_row = list(B) + [""] * (max(oferta, demanda) - len(B))
    rows.append(demanda_row)

    for i in range(oferta):
        cost_row = list(C[i]) + [""] * (max(oferta, demanda) - len(C[i]))
        rows.append(cost_row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False, header=False)



def main():
    m, n = 1, 1
    max_m, max_n = 10000, 10000
    seed = 42
    min_val=1
    max_val=100
    for i in range(m, max_m+1, 1000):
        for j in range(n, max_n+1, 1000):
            print(f"instancias/problema_{i}x{j}_[{min_val},{max_val}]_seed{seed}")
            save_transport_problem_to_csv(f"instancias/problema_{i}x{j}_[{min_val},{max_val}]_seed{seed}", 
            oferta=i, demanda=j, seed=seed, 
            min_val=min_val, max_val=max_val)
main()