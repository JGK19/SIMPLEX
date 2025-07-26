set I;
set J;

param O {i in I};
param D {j in J};
param C {i in I, j in J};

var x {i in I, j in J} >= 0, integer;

minimize Total_Cost:
    sum {i in I, j in J} C[i,j] * x[i,j];

subject to Oferta {i in I}:
    sum {j in J} x[i,j] <= O[i];

subject to Demanda {j in J}:
    sum {i in I} x[i,j] >= D[j];
