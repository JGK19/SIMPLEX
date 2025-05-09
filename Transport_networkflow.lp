\* transport_network_flows *\
Minimize
total_cost_for_transport: 3 edge_1_to_3 + 4 edge_1_to_4 + 5 edge_1_to_5
 + 6 edge_2_to_3 + 7 edge_2_to_4 + 8 edge_2_to_5
Subject To
_C1: - edge_0_to_1 + edge_1_to_3 + edge_1_to_4 + edge_1_to_5 = 0
_C10: edge_3_to_6 = 100
_C11: edge_4_to_6 = 100
_C12: edge_5_to_6 = 100
_C13: edge_6_to_0 = 300
_C2: - edge_0_to_2 + edge_2_to_3 + edge_2_to_4 + edge_2_to_5 = 0
_C3: - edge_1_to_3 - edge_2_to_3 + edge_3_to_6 = 0
_C4: - edge_1_to_4 - edge_2_to_4 + edge_4_to_6 = 0
_C5: - edge_1_to_5 - edge_2_to_5 + edge_5_to_6 = 0
_C6: edge_0_to_1 + edge_0_to_2 - edge_6_to_0 = 0
_C7: - edge_3_to_6 - edge_4_to_6 - edge_5_to_6 + edge_6_to_0 = 0
_C8: edge_0_to_1 = 100
_C9: edge_0_to_2 = 200
Bounds
 0 <= edge_0_to_1
 0 <= edge_0_to_2
 0 <= edge_1_to_3
 0 <= edge_1_to_4
 0 <= edge_1_to_5
 0 <= edge_2_to_3
 0 <= edge_2_to_4
 0 <= edge_2_to_5
 0 <= edge_3_to_6
 0 <= edge_4_to_6
 0 <= edge_5_to_6
 0 <= edge_6_to_0
Generals
edge_0_to_1
edge_0_to_2
edge_1_to_3
edge_1_to_4
edge_1_to_5
edge_2_to_3
edge_2_to_4
edge_2_to_5
edge_3_to_6
edge_4_to_6
edge_5_to_6
edge_6_to_0
End
