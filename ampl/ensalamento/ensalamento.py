import os
from amplpy import AMPL, Environment
from gerador import gerar_instancia_viavel


def main():
    # -----------------------------
    # Dados do problema
    # -----------------------------
    n = 10
    disciplinas, salas, N, C, D = gen_classrom_problem(n, n, seed=42)
    # Construir matriz D[i][j] com deslocamento da origem da disciplina i para sala j
    #D = [D_base[origem[i]] for i in range(len(disciplinas))]

    #escrever_modelo()
    escrever_dados(disciplinas, salas, N, C, D)

    ampl = AMPL()
    ampl.read("modelo.mod")
    ampl.readData("dados.dat")

    ampl.option["solver"] = "gurobi"

    ampl.solve()

    print("\nStatus:", ampl.get_value("solve_result"))
    print("Deslocamento total ótimo:", ampl.get_objective("Total_Deslocamento").value())


    x = ampl.get_variable("x")
    print("\nAlocação ótima (Disciplina → Sala):")
    for (i, j), val in x.getValues().to_dict().items():
        if val > 0.5:
            print(f"  {i} → {j}")


def escrever_modelo(path="modelo.mod"):
    with open(path, "w") as f:
        f.write(r"""
set DISCIPLINAS;
set SALAS;

param N {DISCIPLINAS};          
param C {SALAS};                
param D {DISCIPLINAS, SALAS};   

var x {DISCIPLINAS, SALAS} binary;

minimize Total_Deslocamento:
    sum {i in DISCIPLINAS, j in SALAS} N[i] * D[i,j] * x[i,j];

subject to Alocar_Uma_Sala {i in DISCIPLINAS}:
    sum {j in SALAS} x[i,j] = 1;

subject to Uma_Disciplina_Por_Sala {j in SALAS}:
    sum {i in DISCIPLINAS} x[i,j] <= 1;

subject to Capacidade_Salas {i in DISCIPLINAS, j in SALAS}:
    N[i] * x[i,j] <= C[j];
""")

def escrever_dados(disciplinas, salas, N, C, D, path="dados.dat"):
    with open(path, "w") as f:
        f.write("set DISCIPLINAS := " + " ".join(disciplinas) + ";\n")
        f.write("set SALAS := " + " ".join(salas) + ";\n\n")

        f.write("param N :=\n")
        for d, n in zip(disciplinas, N):
            f.write(f"{d} {n}\n")
        f.write(";\n\n")

        f.write("param C :=\n")
        for s, c in zip(salas, C):
            f.write(f"{s} {c}\n")
        f.write(";\n\n")

        f.write("param D : " + " ".join(salas) + " :=\n")
        for i, d in enumerate(disciplinas):
            f.write(d + " " + " ".join(str(D[i][j]) for j in range(len(salas))) + "\n")
        f.write(";\n")


if __name__ == "__main__":
    main()
