import pandas as pd
import numpy as np
import os

def gerar_tabela_heuristicas_latex(heuristicas, tamanhos, min_val=1, max_val=100, seed=42, rep=10):
    linhas_tempo = []
    linhas_gap = []

    for n in tamanhos:
        nome_csv = f"solutions/problema_{n}x{n}_[{min_val},{max_val}]_seed{seed}_resultado.csv"
        if not os.path.exists(nome_csv):
            print(f"Aviso: Arquivo não encontrado: {nome_csv}")
            continue

        df = pd.read_csv(nome_csv)
        df["tempo_execucao"] = df["tempo_execucao"].astype(float)
        df["custo"] = df["custo"].astype(float)

        # Identificar o custo ótimo para a instância (pela solução ótima)
        custos_otimos = df[df["status"] == "Optimal"]["custo"]
        if custos_otimos.empty:
            print(f"Aviso: Nenhuma solução ótima encontrada para {n}x{n}")
            continue

        custo_otimo = custos_otimos.min()
        tempo_linha = {"Tamanho": f"{n}x{n}"}
        gap_linha = {"Tamanho": f"{n}x{n}"}

        for heur in heuristicas:
            tempos = df[df["codigo"] == heur]["tempo_execucao"].head(rep).to_numpy()
            custos = df[df["codigo"] == heur]["custo"].head(rep).to_numpy()

            if len(tempos) > 0 and len(custos) > 0:
                tempo_medio = np.mean(tempos)
                gap_percentual = np.mean(100 * (custos - custo_otimo) / custo_otimo)

                tempo_linha[heur] = tempo_medio
                gap_linha[heur] = gap_percentual
            else:
                tempo_linha[heur] = None
                gap_linha[heur] = None

        linhas_tempo.append(tempo_linha)
        linhas_gap.append(gap_linha)

    # Converter em DataFrames
    df_tempo = pd.DataFrame(linhas_tempo).set_index("Tamanho")
    df_gap = pd.DataFrame(linhas_gap).set_index("Tamanho")

    # Gerar LaTeX
    def gerar_latex(df, caption, label, format_str="{:.2f}"):
        latex = "\\begin{table}[htbp]\n\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += "\\begin{tabular}{" + "l" + "c" * len(heuristicas) + "}\n"
        latex += "\\toprule\n"
        latex += "Tamanho & " + " & ".join(heuristicas) + " \\\\\n"
        latex += "\\midrule\n"

        for idx, row in df.iterrows():
            valores_formatados = [
                format_str.format(row[heur]) if pd.notna(row[heur]) else "—"
                for heur in heuristicas
            ]
            latex += f"{idx} & " + " & ".join(valores_formatados) + " \\\\\n"

        latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        return latex

    tabela_tempo = gerar_latex(df_tempo, "Tempo médio de execução das heurísticas (s)", "tab:tempo_heuristicas")
    tabela_gap = gerar_latex(df_gap, "Gap percentual médio das heurísticas em relação à solução ótima", "tab:gap_heuristicas")

    return tabela_tempo + "\n\n" + tabela_gap


# ==== CONFIGURAÇÃO ====
tamanhos = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001, 2001]
heuristicas = ["tCantoNoroeste", "tVogel", "tGuloso"]
min_val = 1
max_val = 100
seed = 42
rep = 10

# ==== EXECUÇÃO ====
latex_output = gerar_tabela_heuristicas_latex(heuristicas, tamanhos, min_val, max_val, seed, rep)

# ==== SALVAR EM ARQUIVO ====
with open("tabela_heuristicas.tex", "w", encoding="utf-8") as f:
    f.write(latex_output)

# ==== MOSTRAR NO TERMINAL ====
print(latex_output)
