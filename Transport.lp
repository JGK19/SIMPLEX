\* transport_classical *\
Minimize
total_cost_for_transport: 3 _edge_0_to__0 + 4 _edge_0_to__1 + 5 _edge_0_to__2
 + 6 _edge_1_to__0 + 7 _edge_1_to__1 + 8 _edge_1_to__2
Subject To
_C1: _edge_0_to__0 + _edge_0_to__1 + _edge_0_to__2 = 100
_C2: _edge_1_to__0 + _edge_1_to__1 + _edge_1_to__2 = 200
_C3: _edge_0_to__0 + _edge_1_to__0 = 100
_C4: _edge_0_to__1 + _edge_1_to__1 = 100
_C5: _edge_0_to__2 + _edge_1_to__2 = 100
Bounds
 0 <= _edge_0_to__0
 0 <= _edge_0_to__1
 0 <= _edge_0_to__2
 0 <= _edge_1_to__0
 0 <= _edge_1_to__1
 0 <= _edge_1_to__2
Generals
_edge_0_to__0
_edge_0_to__1
_edge_0_to__2
_edge_1_to__0
_edge_1_to__1
_edge_1_to__2
End
