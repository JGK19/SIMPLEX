import os
import argparse
from amplpy import AMPL
import time
from helpf import ler_instancia_csv, escrever_modelo, escrever_dados, salvar_resultado

def resolver_com_ampl(solver, caminho_csv, nome_instancia):
    disciplinas, salas, N, C, D = ler_instancia_csv(caminho_csv)
    escrever_modelo()
    escrever_dados(disciplinas, salas, N, C, D, nome_instancia)

    ampl = AMPL()
    ampl.read("ampl/modelo.mod")
    ampl.readData(f"ampl/dados_{nome_instancia}.dat")
    ampl.option["solver"] = solver

    inicio = time.time()
    ampl.solve()
    fim = time.time()

    tempo_execucao = fim - inicio
    status = ampl.get_value("solve_result")
    objetivo = ampl.get_objective("Total_Deslocamento").value()

    script_path = __file__
    script_name = os.path.basename(script_path)
    codigo = os.path.splitext(script_name)[0]
    codigo = codigo + f"_{solver}"
    salvar_resultado(status, objetivo, tempo_execucao, codigo, nome_instancia)

    print("\nStatus:", status)
    print("Deslocamento total ótimo:", objetivo)
    print("Tempo de execução:", tempo_execucao)

def main():
    parser = argparse.ArgumentParser(description="Resolve problema de ensalamento a partir de CSV.")
    parser.add_argument("-i", type=int, help="Número de disciplinas")
    parser.add_argument("-j", type=int, help="Número de salas")
    parser.add_argument("--min_alunos", type=int, default=10, help="Número mínimo de alunos por disciplina")
    parser.add_argument("--max_alunos", type=int, default=90, help="Número máximo de alunos por disciplina")
    parser.add_argument("--min_cap", type=int, default=30, help="Capacidade mínima das salas")
    parser.add_argument("--max_cap", type=int, default=90, help="Capacidade máxima das salas")
    parser.add_argument("--min_d", type=int, default=1, help="Custo mínimo de distância")
    parser.add_argument("--max_d", type=int, default=10, help="Custo máximo de distância")
    parser.add_argument("--seed", type=int, default=42, help="Seed usada na geração da instância")
    parser.add_argument("--folder", type=str, default="instancias", help="Pasta onde estão as instâncias")
    parser.add_argument("--solver", type=str, default="highs", help="Solver AMPL (ex: highs, cbc, gurobi)")

    args = parser.parse_args()

    nome_arquivo = f"ensalamento_D{args.i}_S{args.j}_[{args.min_alunos},{args.max_alunos}]_"\
                  f"[{args.min_cap},{args.max_cap}]_[{args.min_d},{args.max_d}]_seed{args.seed}.csv"
    
    caminho_csv = os.path.join(args.folder, nome_arquivo)

    if not os.path.exists(caminho_csv):
        print(f"Arquivo {caminho_csv} não encontrado.")
        return

    nome_instancia = f"D{args.i}_S{args.j}_"\
                     f"[{args.min_alunos},{args.max_alunos}]_"\
                     f"[{args.min_cap},{args.max_cap}]_"\
                     f"[{args.min_d},{args.max_d}]_seed{args.seed}"

    resolver_com_ampl(args.solver, caminho_csv, nome_instancia)


if __name__ == "__main__":
    main()
