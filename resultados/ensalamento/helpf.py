import numpy as np
import pandas as pd
import csv
import os

def salvar_resultado(status, objetivo, tempo_execucao, codigo, nome_instancia):
    output_path = f"solutions/resultado_{nome_instancia}.csv"
    os.makedirs("solutions", exist_ok=True)

    nova_linha = pd.DataFrame([{
        "status": status,
        "custo": objetivo,
        "tempo_execucao": tempo_execucao,
        "codigo": codigo
    }])

    if os.path.exists(output_path):
        df_existente = pd.read_csv(output_path)
        df_final = pd.concat([df_existente, nova_linha], ignore_index=True)
    else:
        df_final = nova_linha

    df_final.to_csv(output_path, index=False)

def salvar_instancia_csv(filepath, disciplinas, salas, N, C, D):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(disciplinas)
        writer.writerow(salas)
        writer.writerow(N)
        writer.writerow(C)
        writer.writerows(D)

def ler_instancia_csv(filepath):
    with open(filepath, 'r', newline='') as f:
        reader = list(csv.reader(f))
        
        disciplinas = reader[0]
        salas = reader[1]
        N = list(map(int, reader[2]))
        C = list(map(int, reader[3]))
        D = np.array([list(map(int, row)) for row in reader[4:]])

    return disciplinas, salas, N, C, D

def escrever_modelo(path="ampl/modelo.mod"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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

import os

def escrever_dados(disciplinas, salas, N, C, D, nome_instancia):
    path = f"ampl/dados_{nome_instancia}.dat"
    
    # Se os dados já existem, não salva novamente
    if os.path.exists(path):
        print(f"[INFO] Dados para a instância '{nome_instancia}' já existem. Pulando escrita.")
        return
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
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

    print(f"[INFO] Dados salvos em: {path}")
