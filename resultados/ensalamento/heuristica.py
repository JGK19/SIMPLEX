def heuristica_gulosa_ensalamento(disciplinas, salas, N, C, D):
    n_disciplinas = len(disciplinas)
    n_salas = len(salas)

    capacidade_disponivel = {sala: capacidade for sala, capacidade in zip(salas, C)}
    alocacao = {}  # disciplina -> sala
    disciplinas_alocadas = set()

    # Lista com (distância, disciplina, sala)
    combinacoes = []
    for i, d in enumerate(disciplinas):
        for j, s in enumerate(salas):
            combinacoes.append((D[i][j], d, s))

    # Ordenar por menor distância
    combinacoes.sort()

    # Tentar alocar as melhores combinações viáveis
    for dist, d, s in combinacoes:
        i = disciplinas.index(d)
        demanda = N[i]

        if d not in alocacao and capacidade_disponivel[s] >= demanda:
            alocacao[d] = s
            capacidade_disponivel[s] -= demanda
            disciplinas_alocadas.add(d)

        if len(disciplinas_alocadas) == n_disciplinas:
            break

    # Verifica se a solução é viável
    if len(alocacao) < n_disciplinas:
        print("[ERRO] Não foi possível alocar todas as disciplinas com a heurística gulosa.")
        return None, None

    # Custo real: distância ponderada pelo número de alunos
    custo_total = sum(
        D[disciplinas.index(d)][salas.index(s)] * N[disciplinas.index(d)]
        for d, s in alocacao.items()
    )

    return alocacao, custo_total

def verificar_solucao_valida(alocacao, disciplinas, salas, N, C):
    if alocacao is None:
        return False

    # Verifica se todas as disciplinas foram alocadas
    if set(alocacao.keys()) != set(disciplinas):
        print("[ERRO] Nem todas as disciplinas foram alocadas.")
        return False

    # Inicializa consumo de capacidade por sala
    uso_por_sala = {sala: 0 for sala in salas}

    for d, s in alocacao.items():
        if s not in salas:
            print(f"[ERRO] Sala '{s}' inválida.")
            return False

        if d not in disciplinas:
            print(f"[ERRO] Disciplina '{d}' inválida.")
            return False

        idx = disciplinas.index(d)
        uso_por_sala[s] += N[idx]

    # Verifica se alguma sala excede sua capacidade
    for s in salas:
        if uso_por_sala[s] > C[salas.index(s)]:
            print(f"[ERRO] Sala '{s}' excedeu a capacidade: usado {uso_por_sala[s]}, capacidade {C[salas.index(s)]}")
            return False

    return True