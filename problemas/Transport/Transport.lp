\* transport_classical *\
Minimize
total_cost_for_transport: 3 edge_0_to_0 + 4 edge_0_to_1 + 5 edge_0_to_2
 + 6 edge_1_to_0 + 7 edge_1_to_1 + 8 edge_1_to_2
Subject To
_C1: edge_0_to_0 + edge_0_to_1 + edge_0_to_2 = 100
_C2: edge_1_to_0 + edge_1_to_1 + edge_1_to_2 = 200
_C3: edge_0_to_0 + edge_1_to_0 = 100
_C4: edge_0_to_1 + edge_1_to_1 = 100
_C5: edge_0_to_2 + edge_1_to_2 = 100
Bounds
 0 <= edge_0_to_0
 0 <= edge_0_to_1
 0 <= edge_0_to_2
 0 <= edge_1_to_0
 0 <= edge_1_to_1
 0 <= edge_1_to_2
Generals
edge_0_to_0
edge_0_to_1
edge_0_to_2
edge_1_to_0
edge_1_to_1
edge_1_to_2
End
