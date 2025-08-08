import os
import pandas as pd
import numpy as np

# Configurações
tamanhos = [101,201,301,401,501,601,701,801,901,1001,2001]
K = 10

min_alunos = 10
max_alunos = 90
min_cap = 30
max_cap = 90
min_d = 1
max_d = 10
seed = 42

solvers = ["hungaro", "gulosa"]
solver_otimo = "ensalamento_ampl_gurobi"

os.makedirs("tabelas", exist_ok=True)

# --- Preparar dados para tempo médio ---
dados_tempo = []
for n in tamanhos:
    nome_instancia = f"D{n}_S{n}_[{min_alunos},{max_alunos}]_[{min_cap},{max_cap}]_[{min_d},{max_d}]_seed{seed}"
    caminho_csv = f"solutions/resultado_{nome_instancia}.csv"

    if not os.path.exists(caminho_csv):
        print(f"[AVISO] Arquivo não encontrado: {caminho_csv}")
        linha = [f"{n}"] + ["--"] * (len(solvers) + 1)
        dados_tempo.append(linha)
        continue

    df = pd.read_csv(caminho_csv)
    linha = [f"{n}"]

    # Tempo heurísticas + ótimo
    for solver in solvers + [solver_otimo]:
        linhas_solver = df[df["codigo"] == solver]
        tempos = linhas_solver["tempo_execucao"].tolist()
        if len(tempos) < K:
            print(f"[AVISO] Menos de {K} execuções para solver {solver} em {nome_instancia}")
            linha.append("--")
        else:
            media = np.mean(tempos[:K])
            linha.append(f"{media:.2f}")

    dados_tempo.append(linha)

# --- Preparar dados para gap percentual ---
dados_gap = []
for n in tamanhos:
    nome_instancia = f"D{n}_S{n}_[{min_alunos},{max_alunos}]_[{min_cap},{max_cap}]_[{min_d},{max_d}]_seed{seed}"
    caminho_csv = f"solutions/resultado_{nome_instancia}.csv"

    if not os.path.exists(caminho_csv):
        linha = [f"{n}"] + ["--"] * len(solvers)
        dados_gap.append(linha)
        continue

    df = pd.read_csv(caminho_csv)

    # Média custo ótimo
    df_otimo = df[df["codigo"] == solver_otimo]
    custos_otimo = df_otimo["custo"].tolist()
    if len(custos_otimo) < K:
        print(f"[AVISO] Menos de {K} execuções para solver ótimo em {nome_instancia}")
        linha = [f"{n}"] + ["--"] * len(solvers)
        dados_gap.append(linha)
        continue
    custo_otimo_medio = np.mean(custos_otimo[:K])

    linha = [f"{n}"]
    for solver in solvers:
        df_solv = df[df["codigo"] == solver]
        custos = df_solv["custo"].tolist()
        if len(custos) < K:
            print(f"[AVISO] Menos de {K} execuções para solver {solver} em {nome_instancia}")
            linha.append("--")
        else:
            custo_medio = np.mean(custos[:K])
            gap = 100.0 * (custo_medio - custo_otimo_medio) / custo_otimo_medio
            linha.append(f"{gap:.2f}")

    dados_gap.append(linha)

# --- Função para escrever tabela LaTeX ---
def salvar_tabela_latex(dados, colunas, caption, label, caminho):
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{" + caption + "}\n")
        f.write(r"\label{" + label + "}\n")
        f.write(r"\begin{tabular}{l" + "c" * (len(colunas)-1) + "}\n")
        f.write(r"\toprule" + "\n")
        f.write(" & ".join(colunas) + r" \\" + "\n")
        f.write(r"\midrule" + "\n")
        for linha in dados:
            f.write(" & ".join(linha) + r" \\" + "\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

# Colunas para cada tabela
colunas_tempo = ["Tamanho"] + solvers + [solver_otimo]
colunas_gap = ["Tamanho"] + solvers

# Salvar arquivos
salvar_tabela_latex(
    dados_tempo,
    colunas_tempo,
    caption="Tempo médio de execução (em segundos)",
    label="tab:tempo_execucao_heuristicas",
    caminho="tabelas/tempo_execucao_heuristica.tex"
)

salvar_tabela_latex(
    dados_gap,
    colunas_gap,
    caption="Gap percentual médio em relação à solução ótima (\\%)",
    label="tab:gap_percentual_heuristica",
    caminho="tabelas/gap_percentual_heuristica.tex"
)

print("Tabelas LaTeX salvas em 'tabelas/'")
