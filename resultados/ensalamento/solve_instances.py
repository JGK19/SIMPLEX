import os
import sys
import subprocess
import random
import pandas as pd

# Parâmetros fixos
python_exec = sys.executable

tamanhos = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001, 2001]
solvers = ["highs", "cbc", "gurobi", "xpress", "cplex", "mosek", "copt", "gcg", "scip"]
#solvers = ["hungaro", "guloso"]
K = 10

min_alunos = 10
max_alunos = 90
min_cap = 30
max_cap = 90
min_d = 1
max_d = 10
seed = 42

def contar_execucoes_existentes(nome_instancia, solver):
    path_csv = f"solutions/resultado_{nome_instancia}.csv"
    print(path_csv)
    if not os.path.exists(path_csv):
        return 0
    try:
        df = pd.read_csv(path_csv)
        return len(df[df["codigo"] == f"ensalamento_ampl_{solver}"])
    except Exception as e:
        print(f"[ERRO] ao ler {path_csv}: {e}")
        return 0

def rodar_execucao(tam, solver):
    cmd = [
        python_exec, "ensalamento_ampl.py",
        "-i", str(tam),
        "-j", str(tam),
        "--solver", solver
    ]
    subprocess.run(cmd, check=True)

def executar(tam, solver):
    cmd = [
        python_exec, f"{solver}.py",
        "-i", str(tam),
        "-j", str(tam),
    ]
    subprocess.run(cmd, check=True)


def main():
    for tam in tamanhos:
        nome_instancia = f"D{tam}_S{tam}_[{min_alunos},{max_alunos}]_[{min_cap},{max_cap}]_[{min_d},{max_d}]_seed{seed}"
        print(f"\n>> Verificando instância: {nome_instancia}")

        # Construir fila de execuções necessárias
        fila_execucao = []
        for solver in solvers:
            existentes = contar_execucoes_existentes(nome_instancia, solver)
            faltam = K - existentes
            if faltam > 0:
                fila_execucao.extend([solver] * faltam)

        if not fila_execucao:
            print(f"[OK] Já há {K} execuções para todos os solvers em {nome_instancia}")
            continue

        # Embaralhar solvers e executar
        random.shuffle(fila_execucao)
        print(f"[INFO] Rodando {len(fila_execucao)} execuções pendentes...")
        
        for solver in fila_execucao:
            try:
                print(f"  -> Executando {solver} para {nome_instancia}")
                rodar_execucao(tam, solver)
                #executar(tam, solver)
            except subprocess.CalledProcessError as e:
                print(f"[ERRO] na execução com {solver} para {nome_instancia}")

if __name__ == "__main__":
    main()