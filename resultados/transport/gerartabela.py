import pandas as pd
import numpy as np
import os

def gerar_tabela_latex(solvers, tamanhos, min_val=1, max_val=100, seed=42, rep=10):
    dados = []

    for n in tamanhos:
        nome_csv = f"solutions/problema_{n}x{n}_[{min_val},{max_val}]_seed{seed}_resultado.csv"
        if not os.path.exists(nome_csv):
            print(f"Aviso: Arquivo não encontrado: {nome_csv}")
            continue

        df = pd.read_csv(nome_csv)
        df["tempo_execucao"] = df["tempo_execucao"].astype(float)

        linha = {"Tamanho": f"{n}x{n}"}

        for solver in solvers:
            # código pode ser 'tClassico_ampl_<solver>'
            codigo_solver = f"tClassico_ampl_{solver}"
            tempos = df[df["codigo"] == codigo_solver]["tempo_execucao"].head(rep).to_numpy()

            if len(tempos) > 0:
                linha[solver] = np.mean(tempos)
            else:
                linha[solver] = None

        dados.append(linha)

    # Criar DataFrame final
    tabela_df = pd.DataFrame(dados)
    tabela_df.set_index("Tamanho", inplace=True)

    # Gerar LaTeX
    latex = "\\begin{table}[htbp]\n\\centering\n"
    latex += "\\caption{Tempo médio de execução (em segundos)}\n"
    latex += "\\label{tab:tempo_execucao}\n"
    latex += "\\begin{tabular}{" + "l" + "c" * len(solvers) + "}\n"
    latex += "\\toprule\n"
    latex += "Tamanho & " + " & ".join(solvers) + " \\\\\n"
    latex += "\\midrule\n"

    for idx, row in tabela_df.iterrows():
        tempos_formatados = [
            f"{row[solver]:.2f}" if pd.notna(row[solver]) else "—"
            for solver in solvers
        ]
        latex += f"{idx} & " + " & ".join(tempos_formatados) + " \\\\\n"

    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}"

    return latex


# ==== CONFIGURAÇÃO ====
tamanhos = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001, 2001]
solvers = ["highs", "cbc", "gurobi", "xpress", "cplex", "mosek"]
min_val = 1
max_val = 100
seed = 42
rep = 10

# ==== EXECUÇÃO ====
latex_output = gerar_tabela_latex(solvers, tamanhos, min_val, max_val, seed, rep)

# ==== SALVAR EM ARQUIVO ====
with open("tabela_comparacao.tex", "w", encoding="utf-8") as f:
    f.write(latex_output)

# ==== MOSTRAR ====
print(latex_output)
