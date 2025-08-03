import argparse
import pandas as pd
import numpy as np
import time
import os
from amplpy import AMPL
from helpf import salvar_dados_ampl_transport

def ler_instancia_csv(caminho_csv):
    df = pd.read_csv(caminho_csv, header=None)
    i, j = int(df.iloc[0, 0]), int(df.iloc[0, 1])
    Oi = df.iloc[1, :i].to_numpy()
    Dj = df.iloc[2, :j].to_numpy()
    Cost = df.iloc[3:3+i, :j].to_numpy()
    assert Cost.shape == (i, j), "Dimensão da matriz de custos inválida"
    return i, j, Oi, Dj, Cost

def resolver_ampl(i, j, Oi, Dj, Cost, solver, nome_arquivo_csv):
    ampl_dir = "ampl"
    solutions_dir = "solutions"
    os.makedirs(ampl_dir, exist_ok=True)
    os.makedirs(solutions_dir, exist_ok=True)

    dat_path = os.path.join(ampl_dir, f"transportDat_{i}x{j}_{os.path.basename(nome_arquivo_csv)}.dat")
    salvar_dados_ampl_transport(Oi, Dj, Cost, filename=dat_path)

    ampl = AMPL()
    ampl.option["solver"] = solver
    ampl.option["show_stats"] = 1

    mod_path = os.path.join(ampl_dir, "transportMod.mod")
    ampl.read(mod_path)
    ampl.read_data(dat_path)

    start = time.time()
    ampl.solve()
    elapsed = time.time() - start

    status = ampl.get_value("solve_result")
    custo_total = ampl.get_objective("Total_Cost").value()

    script_path = __file__
    script_name = os.path.basename(script_path)
    codigo = f"{os.path.splitext(script_name)[0]}_{solver}"

    resultado_csv = nome_arquivo_csv.replace("instancias/", "solutions/") + "_resultado.csv"
    escrever_cabecalho = not os.path.exists(resultado_csv)

    with open(resultado_csv, "a", newline="") as f:
        import csv
        writer = csv.writer(f)
        if escrever_cabecalho:
            writer.writerow(["status", "custo", "tempo_execucao", "codigo"])
        writer.writerow([status, custo_total, f"{elapsed:.6f}", codigo])

    print("Problema resolvido via AMPL.")
    print(f"Status: {status}")
    print(f"Custo total: {custo_total}")
    print(f"Tempo: {elapsed:.6f} s")


def main():
    parser = argparse.ArgumentParser(description="Resolve problema de transporte com AMPL a partir de CSV.")
    parser.add_argument("i", type=int, help="Número de ofertas")
    parser.add_argument("j", type=int, help="Número de demandas")
    parser.add_argument("--min_val", type=int, default=1)
    parser.add_argument("--max_val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver", type=str, default="highs", help="Solver AMPL (ex: highs, cbc, gurobi)")
    args = parser.parse_args()

    nome_csv = f"instancias/problema_{args.i}x{args.j}_[{args.min_val},{args.max_val}]_seed{args.seed}"
    if not os.path.exists(nome_csv):
        print(f"Arquivo {nome_csv} não encontrado.")
        return

    i, j, Oi, Dj, Cost = ler_instancia_csv(nome_csv)
    resolver_ampl(i, j, Oi, Dj, Cost, args.solver, nome_csv)

if __name__ == "__main__":
    main()
