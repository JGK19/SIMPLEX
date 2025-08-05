import numpy as np

def gen_classrom_problem(n_disciplinas, n_salas, 
                          min_alunos=10, max_alunos=60, 
                          min_cap=30, max_cap=70,
                          min_d=1, max_d=10,
                          seed=None):

    assert n_disciplinas <= n_salas, "Número de disciplinas deve ser menor ou igual ao de salas para garantir viabilidade."

    if seed is not None:
        np.random.seed(seed)

    disciplinas = [f"D{i+1}" for i in range(n_disciplinas)]
    salas = [f"S{j+1}" for j in range(n_salas)]

    C = np.random.randint(min_cap, max_cap + 1, size=n_salas)

    N = []
    for i in range(n_disciplinas):
        sala_suf = np.random.choice(n_salas)
        cap_suf = C[sala_suf]

        alunos = np.random.randint(min_alunos, cap_suf + 1)
        N.append(alunos)

    D = np.random.randint(min_d, max_d, size=(n_disciplinas, n_salas))

    return disciplinas, salas, N, C, D


if __name__ == "__main__":
    # Exemplo de uso:
    disc, salas, N, C, D = gerar_instancia_viavel(8, 10, seed=123)
    print("Disciplinas:", disc)
    print("Salas:", salas)
    print("N alunos:", N)
    print("Capacidades:", C)
    print("Deslocamentos D (matriz):")
    for linha in D:
        print(linha)
