import pandas as pd
from scipy.stats import ttest_rel

# Carregar os dados
df = pd.read_csv("solutions/problema_1001x1001_[1,100]_seed42_resultado.csv")  # Substitua pelo caminho correto do seu CSV

# Filtrar os tempos de execução para cada código
tempos_classico = df[df["codigo"] == "tClassico"]["tempo_execucao"].reset_index(drop=True)
tempos_restrito = df[df["codigo"] == "tRestrito"]["tempo_execucao"].reset_index(drop=True)

# Verificar se os pares estão corretamente alinhados
assert len(tempos_classico) == len(tempos_restrito), "As listas não têm o mesmo número de observações!"

# Teste t pareado
stat, p_valor = ttest_rel(tempos_classico, tempos_restrito)

# Exibir resultados
print(f"Estatística t: {stat}")
print(f"p-valor: {p_valor}")

alpha = 0.01
if p_valor < alpha:
    print(f"Diferença significativa entre os tempos (rejeita H0 com {1 - alpha} de confiança)")
else:
    print("Sem diferença significativa (não rejeita H0)")
