import numpy as np

max_n = 100
min_n = 1

def gen_transport_problem(oferta, demanda):
    custos = np.random.randint(min_n, max_n, size=(oferta, demanda))

    A = np.random.randint(min_n, max_n, size=oferta)
    sum_A = np.sum(A)

    raw_B = np.random.rand(demanda)
    B = (raw_B / raw_B.sum()) * (sum_A * np.random.uniform(0.5, 1.0))
    #B = (raw_B / raw_B.sum()) * (sum_A)
    B = np.floor(B).astype(int)

    return A, B, custos
