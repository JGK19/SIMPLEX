import numpy as np

def gen_matrix_I(n):
    return np.identity(n)

def gen_cost_0(n):
    return np.zeros(n)

def gen_B_and_N(matrix_A, basic_vars, nonbasic_vars):
    matrix_B = []
    matrix_N = []
    for var in basic_vars:
        matrix_B.append(matrix_A[:, var])
    for var in nonbasic_vars:
        matrix_N.append(matrix_A[:, var])
    matrix_B = np.array(matrix_B).T #escolhendo variaveis basicas
    matrix_N = np.array(matrix_N).T #escolhendo variaveis não basicas

    return matrix_B, matrix_N

def gen_cost_B_and_N(cost, basic_vars, nonbasic_vars):
    cost_B = []
    cost_N = []
    for var in basic_vars:
        cost_B.append(cost[var])
    for var in nonbasic_vars:
        cost_N.append(cost[var])
    cost_B = np.array(cost_B)
    cost_N = np.array(cost_N)

    return cost_B, cost_N

def simplex(cost, matrix_R, b):
    rows_R = len(matrix_R)

    matrix_I =  gen_matrix_I(rows_R)#coeficientes das variaveis de folga em cada linha do sistema

    matrix_A = np.hstack((matrix_R, matrix_I)) #concatenando matrizes horizontalmente
    cost = np.concatenate((cost, gen_cost_0(rows_R)), axis=None) #adicionando coeficientes das variaveis de folga na função objetivo

    basic_vars = list(range(len(matrix_R[0]), len(matrix_A[0]))) # x2, x3, x4
    nonbasic_vars = list(range(0, len(matrix_R[0]))) #x0, x1

    matrix_B, matrix_N = gen_B_and_N(matrix_A, basic_vars, nonbasic_vars)

    cost_B, cost_N = gen_cost_B_and_N(cost, basic_vars, nonbasic_vars) #coeficicentes das variaveis basicas e nao basicas na função objetivo
    
    have_positive = True
    while(have_positive):
        #verificar se esta em uma solução otima
        B_inv = np.linalg.inv(matrix_B)
        resultados = [float("-inf") for _ in range(len(nonbasic_vars) + len(basic_vars))]
        resultados = np.array(resultados)
        for i in nonbasic_vars:
            temp = np.dot(cost_B, B_inv)
            Zi = np.dot(temp, matrix_A[:,i])
            Ci = cost[i]
            resultados[i] = Zi - Ci

        # verifica se existe algum resultado maior que zero
        function_vec = np.vectorize(lambda x: x > 0) 
        boolean = function_vec(resultados)
        have_positive = True in boolean
        if(not have_positive):
            break

        #seleciona nova variavel para entrar nas variaveis basicas
        index_of_max = resultados.argmax(axis=0)
        new_basic_var = nonbasic_vars[index_of_max]

        # calcula valores para encontrar quem tirar das soluções basicas
        x_solutions = np.dot(B_inv, b)
        y_values = np.dot(B_inv, matrix_A[:,new_basic_var])

        # seleciona menor dos valores e encontra variavel parar tirar das variaveis basicas
        resultados2 = np.divide(x_solutions, y_values)
        index_of_min = resultados2.argmin(axis=0)
        old_basic_var = basic_vars[index_of_min]

        #faz a troca das variaveis basicas e não basicas
        basic_vars[index_of_min] = new_basic_var
        nonbasic_vars[index_of_max] = old_basic_var
        
        #calcula novamente matrizes das variaveis basicas e não basicas // não precisava era so trocar mas nao pensei em como ainda
        matrix_B = np.array([matrix_A[:, basic_vars[0]],matrix_A[:, basic_vars[1]],matrix_A[:, basic_vars[2]]]).T
        matrix_N = np.array([matrix_A[:, nonbasic_vars[0]],matrix_A[:, nonbasic_vars[1]]]).T



    
    B_inv = np.linalg.inv(matrix_B)
    solution = np.dot(B_inv, b)
    
    return (basic_vars, solution, nonbasic_vars)
