import os
import argparse
import time
import numpy as np
from helpf import ler_instancia_csv, salvar_resultado
from heuristica import heuristica_gulosa_ensalamento, verificar_solucao_valida

def resolver_com_gulosa(caminho_csv, nome_instancia):
    # Lê os dados da instância
    disciplinas, salas, N, C, D = ler_instancia_csv(caminho_csv)

    inicio = time.time()
    alocacao, custo = heuristica_gulosa_ensalamento(disciplinas, salas, N, C, D)
    fim = time.time()

    tempo_execucao = fim - inicio

    if alocacao is not None and verificar_solucao_valida(alocacao, disciplinas, salas, N, C):
        status = "aproximada"
        print("[INFO] Solução viável encontrada.")
    else:
        status = "falha"
        custo = -1  # ou float('inf'), se preferir
        print("[ERRO] Solução inválida retornada pela heurística gulosa.")

    codigo = "gulosa"
    salvar_resultado(status, custo, tempo_execucao, codigo, nome_instancia)

    print("\nStatus:", status)
    print("Deslocamento total:", custo if custo != -1 else "N/A")
    print("Tempo de execução:", tempo_execucao)

def main():
    parser = argparse.ArgumentParser(description="Resolve problema de ensalamento com heurística gulosa.")
    parser.add_argument("-i", type=int, help="Número de disciplinas")
    parser.add_argument("-j", type=int, help="Número de salas")
    parser.add_argument("--min_alunos", type=int, default=10)
    parser.add_argument("--max_alunos", type=int, default=90)
    parser.add_argument("--min_cap", type=int, default=30)
    parser.add_argument("--max_cap", type=int, default=90)
    parser.add_argument("--min_d", type=int, default=1)
    parser.add_argument("--max_d", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folder", type=str, default="instancias")

    args = parser.parse_args()

    nome_arquivo = f"ensalamento_D{args.i}_S{args.j}_[{args.min_alunos},{args.max_alunos}]_"
    nome_arquivo += f"[{args.min_cap},{args.max_cap}]_[{args.min_d},{args.max_d}]_seed{args.seed}.csv"

    caminho_csv = os.path.join(args.folder, nome_arquivo)

    if not os.path.exists(caminho_csv):
        print(f"Arquivo {caminho_csv} não encontrado.")
        return

    nome_instancia = f"D{args.i}_S{args.j}_[{args.min_alunos},{args.max_alunos}]_"
    nome_instancia += f"[{args.min_cap},{args.max_cap}]_[{args.min_d},{args.max_d}]_seed{args.seed}"

    resolver_com_gulosa(caminho_csv, nome_instancia)

if __name__ == "__main__":
    main()
