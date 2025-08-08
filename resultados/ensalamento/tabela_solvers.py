import os
import pandas as pd

tamanhos = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001, 2001]
solvers = ["highs", "cbc", "gurobi", "xpress", "cplex", "mosek", "copt"]
K = 10

min_alunos = 10
max_alunos = 90
min_cap = 30
max_cap = 90
min_d = 1
max_d = 10
seed = 42

saida_path = "tabelas/tempo_execucao.tex"

os.makedirs("tabelas", exist_ok=True)

dados_tabela = []

for n in tamanhos:
    nome_instancia = f"D{n}_S{n}_[{min_alunos},{max_alunos}]_[{min_cap},{max_cap}]_[{min_d},{max_d}]_seed{seed}"
    caminho_csv = f"solutions/resultado_{nome_instancia}.csv"

    if not os.path.exists(caminho_csv):
        print(f"[AVISO] Arquivo não encontrado: {caminho_csv}")
        linha = [nome_instancia] + ["--"] * len(solvers)
        dados_tabela.append(linha)
        continue

    df = pd.read_csv(caminho_csv)
    linha = [f"{n}"]

    for solver in solvers:
        nome = f"ensalamento_ampl_{solver}"
        tempos = df[df['codigo'] == nome]['tempo_execucao'].tolist()

        if len(tempos) < K:
            print(f"[AVISO] Apenas {len(tempos)} execuções para {solver} em {nome_instancia}")
            linha.append("--")
        else:
            media = sum(tempos[:K]) / K
            linha.append(f"{media:.2f}")

    dados_tabela.append(linha)

# Gerar conteúdo LaTeX
with open(saida_path, "w", encoding="utf-8") as f:
    f.write(r"\begin{table}[htbp]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{Tempo médio de execução (em segundos)}" + "\n")
    f.write(r"\label{tab:tempo_execucao}" + "\n")
    f.write(r"\begin{tabular}{l" + "c" * len(solvers) + "}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write("Tamanho & " + " & ".join(solvers) + r" \\" + "\n")
    f.write(r"\midrule" + "\n")

    for linha in dados_tabela:
        f.write(" & ".join(linha) + r" \\" + "\n")

    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")
    f.write(r"\end{table}" + "\n")

print(f"Tabela salva em: {saida_path}")
