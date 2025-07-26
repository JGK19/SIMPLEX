import numpy as np


def gen_transport_problem(oferta, demanda, seed=None, min_val=1, max_val=100):
    if seed is not None:
        np.random.seed(seed)

    custos = np.random.randint(min_val, max_val, size=(oferta, demanda))

    A = np.random.randint(min_val, max_val, size=oferta)
    sum_A = np.sum(A)

    raw_B = np.random.rand(demanda)
    B = (raw_B / raw_B.sum()) * (sum_A * np.random.uniform(0.5, 1.0))
    #B = (raw_B / raw_B.sum()) * (sum_A)
    B = np.floor(B).astype(int)

    return A, B, custos 


def gen_transport_problem_notint(oferta, demanda, seed=None, min_val=1, max_val=100):
    """
    Gera um problema de transporte com valores reais (não inteiros).
    
    Parâmetros:
    - oferta: número de fontes
    - demanda: número de destinos
    - seed: valor opcional para reprodutibilidade
    - min_val, max_val: intervalo dos valores aleatórios

    Retorna:
    - A: vetor de oferta (tamanho oferta)
    - B: vetor de demanda (tamanho demanda)
    - custos: matriz de custos reais (oferta x demanda)
    """
    if seed is not None:
        np.random.seed(seed)

    custos = np.random.uniform(min_val, max_val, size=(oferta, demanda))

    A = np.random.uniform(min_val, max_val, size=oferta)
    sum_A = np.sum(A)

    raw_B = np.random.rand(demanda)
    B = (raw_B / raw_B.sum()) * (sum_A * np.random.uniform(0.5, 1.0))

    return A, B, custos