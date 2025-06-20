from amplpy import AMPL, Environment

# Cria o ambiente AMPL
ampl = AMPL()

# Dados (comidas, custo e nutrientes)
ampl.eval("""
set FOODS;
param cost {FOODS};
param min_nutrient;
param max_nutrient;
param nutrient {FOODS};

var Buy {i in FOODS} >= 0;

minimize Total_Cost:
    sum {i in FOODS} cost[i] * Buy[i];

subject to MinNutrition:
    sum {i in FOODS} nutrient[i] * Buy[i] >= min_nutrient;

subject to MaxNutrition:
    sum {i in FOODS} nutrient[i] * Buy[i] <= max_nutrient;
""")

# Carregar dados diretamente do Python
foods = ["bread", "milk", "cheese"]
cost = {"bread": 0.5, "milk": 0.8, "cheese": 2.0}
nutrient = {"bread": 1.0, "milk": 3.0, "cheese": 4.0}
min_nutr = 5.0
max_nutr = 10.0

# Enviar dados para AMPL
ampl.set["FOODS"] = foods
ampl.param["cost"] = cost
ampl.param["nutrient"] = nutrient
ampl.param["min_nutrient"] = min_nutr
ampl.param["max_nutrient"] = max_nutr

# Resolver com o solver instalado (ex: highs)
ampl.option["solver"] = "cbc"
ampl.solve()

# Mostrar resultado
print("Custo total:", ampl.obj["Total_Cost"].value())
buy = ampl.get_variable("Buy")
for food in foods:
    print(f"{food}: {buy[food].value():.2f}")