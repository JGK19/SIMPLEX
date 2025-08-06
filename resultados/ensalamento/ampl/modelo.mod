
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
