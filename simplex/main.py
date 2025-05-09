import numpy as np
from simplex import *

def main():
    cost = np.array([-1.2, -1.7]) #coeficientes da função para ser maximizada //x0 e x1
    matrix_R = np.array([[1,1],[1,0],[0,1]]) # coeficientes das variaveis x0 e x1 no sistema de restrições
    b = np.array([5000, 3000, 4000]) #lado direito das restrições

    basic_vars, solution, nonbasic_vars = simplex(cost, matrix_R, b)
    print("x:",basic_vars, " = ", solution)
    print("x:",nonbasic_vars, " = ", 0)

    print()

    #nao precisa multiplica restrições por menos 1? nao erapra ser igual? primal e dual deu certo?
    cost = np.array([3000, 4000, 5000]) #coeficientes da função para ser maximizada //x0 e x1
    matrix_R = np.array([[-1, 0, -1],[0, -1, -1]]) # coeficientes das variaveis x0 e x1 no sistema de restrições
    b = np.array([-1.2, -1.7]) #lado direito das restrições
    basic_vars, solution, nonbasic_vars = simplex(cost, matrix_R, b)
    print("x:",basic_vars, " = ", solution)
    print("x:",nonbasic_vars, " = ", 0)

    print()



    return

main()