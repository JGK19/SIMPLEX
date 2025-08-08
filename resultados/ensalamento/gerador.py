from helpf import salvar_instancia_csv, ler_instancia_csv
import numpy as np

instances_folder = "instancias/"

def main():
    for j in [1001]: #[10**k for k in range(1,5)]:
        for i in [1001]: #range(1, j, int(j * (1/10))):
            n_disciplinas = i
            n_salas = i
            min_alunos = 10
            max_alunos = 90
            min_cap = 30
            max_cap = 90
            min_d = 1
            max_d = 10
            seed = 42

            disc, salas, N, C, D = gen_classrom_problem(n_disciplinas, n_salas, seed=42)
            salvar_instancia_csv(instances_folder + f"ensalamento_D{n_disciplinas}_S{n_salas}_[{min_alunos},{max_alunos}]_[{min_cap},{max_cap}]_[{min_d},{max_d}]_seed{seed}.csv", disc, salas, N, C, D)

    #disc_lida, salas_lida, N_lido, C_lido, D_lido = ler_instancia_csv(instances_folder + "instancia_exemplo.csv")


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
    main()