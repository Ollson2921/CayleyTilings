from gridded_cayley_permutations import Tiling, GriddedCayleyPerm
from cayley_permutations import CayleyPermutation
from mapplings import Parameter, MappedTiling, MTFactor, ParameterPlacement
from gridded_cayley_permutations.row_col_map import RowColMap

base_obs = [
    GriddedCayleyPerm(CayleyPermutation([0, 2, 1]), ((0, 0), (0, 0), (0, 0))),
    GriddedCayleyPerm(CayleyPermutation([0, 2, 1]), ((0, 1), (0, 1), (0, 1))),
    GriddedCayleyPerm(CayleyPermutation([0, 2, 1]), ((1, 0), (1, 0), (1, 0))),
    GriddedCayleyPerm(CayleyPermutation([0, 2, 1]), ((1, 1), (1, 1), (1, 1))),
    GriddedCayleyPerm(CayleyPermutation([0, 2, 1]), ((2, 0), (2, 0), (2, 0))),
    GriddedCayleyPerm(CayleyPermutation([0, 2, 1]), ((2, 1), (2, 1), (2, 1))),
]

point_obs = [
    GriddedCayleyPerm(CayleyPermutation([0]), ((0, 1),)),
    GriddedCayleyPerm(CayleyPermutation([0]), ((1, 0),)),
    GriddedCayleyPerm(CayleyPermutation([0]), ((2, 0),)),
]

base_tiling = Tiling(
    base_obs + point_obs,
    [[GriddedCayleyPerm(CayleyPermutation((0,)), [(2, 1)])]],
    (3, 2),
)
# print(base_tiling)

extra_obs = [
    GriddedCayleyPerm(CayleyPermutation([0, 2, 1]), ((3, 0), (3, 0), (3, 0))),
    GriddedCayleyPerm(CayleyPermutation([0, 2, 1]), ((3, 1), (3, 1), (3, 1))),
]
ghost_tiling = Tiling(
    base_obs + extra_obs,
    [[GriddedCayleyPerm(CayleyPermutation([0]), ((3, 1),))]],
    (4, 2),
)

# print(ghost_tiling)

P3 = Parameter(base_tiling, RowColMap({0: 0, 1: 1, 2: 2}, {0: 0, 1: 1}))

P1 = Parameter(
    ghost_tiling.add_obstructions(
        [
            # GriddedCayleyPerm(CayleyPermutation([0]), ((0, 1),)),
            GriddedCayleyPerm(CayleyPermutation([0]), ((1, 0),)),
            # GriddedCayleyPerm(CayleyPermutation([0]), ((0, 1),)),
            GriddedCayleyPerm(CayleyPermutation([0]), ((2, 0),)),
            GriddedCayleyPerm(CayleyPermutation([0]), ((3, 0),)),
        ]
    ).add_requirement_list([GriddedCayleyPerm(CayleyPermutation([0]), ((0, 1),))]),
    RowColMap({0: 0, 1: 1, 2: 1, 3: 2}, {0: 0, 1: 1}),
)
P2 = Parameter(
    ghost_tiling.add_obstructions(
        [
            GriddedCayleyPerm(CayleyPermutation([0]), ((0, 1),)),
            GriddedCayleyPerm(CayleyPermutation([0]), ((1, 0),)),
            # GriddedCayleyPerm(CayleyPermutation([0]), ((1, 1),)),
            GriddedCayleyPerm(CayleyPermutation([0]), ((2, 0),)),
            GriddedCayleyPerm(CayleyPermutation([0]), ((3, 0),)),
        ]
    ),
    RowColMap({0: 0, 1: 0, 2: 1, 3: 2}, {0: 0, 1: 1}),
)


M = MappedTiling(base_tiling, [P1, P2], [], [])
# M = MappedTiling(base_tiling, [P2, P1], [], [])
M = MappedTiling(base_tiling, [P3], [], [])
# M.reap_contradictory_ghosts()
#print(M)
#print("FACTORS:")

#for factor in MTFactor(M).find_factors():
#    print("-------------------------------------")
#    print(factor)
    # print(base_tiling.sub_tiling(factor))
# print(M.find_factors())

#print(MTFactor(M).is_factorable2(MTFactor(M).find_factor_cells()))


point = CayleyPermutation((0,))
cay = CayleyPermutation((0,0))
asc2 = CayleyPermutation((0,1))
des2 = CayleyPermutation((1,0))
asc3 = CayleyPermutation((0,1,2))
obstructions = [
    GriddedCayleyPerm(point,((0,0),)),
    GriddedCayleyPerm(point,((0,1),)),
    GriddedCayleyPerm(point,((0,2),)),
    GriddedCayleyPerm(point,((0,3),)),
    GriddedCayleyPerm(point,((1,0),)),
    GriddedCayleyPerm(point,((2,0),)),
    GriddedCayleyPerm(point,((3,0),)),
    GriddedCayleyPerm(point,((1,2),)),
    GriddedCayleyPerm(point,((1,3),)),
    GriddedCayleyPerm(point,((2,1),)),
    GriddedCayleyPerm(point,((3,1),)),
    GriddedCayleyPerm(point,((2,2),)),
    GriddedCayleyPerm(point,((2,3),)),
    GriddedCayleyPerm(point,((3,2),)),
    GriddedCayleyPerm(point,((4,3),)),
    GriddedCayleyPerm(point,((3,4),)),
    GriddedCayleyPerm(point,((4,1),)),
    GriddedCayleyPerm(point,((1,4),)),
    GriddedCayleyPerm(point,((4,4),)),
    GriddedCayleyPerm(asc2,((2,4),(2,4))),
    GriddedCayleyPerm(asc2,((4,2),(4,2))),
    GriddedCayleyPerm(asc3,((0,4),(0,4),(0,4))),
    GriddedCayleyPerm(asc3,((0,4),(0,4),(2,4))),
    GriddedCayleyPerm(asc3,((4,0),(4,0),(4,0))),
    GriddedCayleyPerm(asc3,((4,0),(4,0),(4,2))),

    GriddedCayleyPerm(des2,((1,1),(1,1))),
    GriddedCayleyPerm(asc2,((1,1),(1,1))),
    GriddedCayleyPerm(cay,((1,1),(1,1))),
    GriddedCayleyPerm(des2,((3,3),(3,3))),
    GriddedCayleyPerm(asc2,((3,3),(3,3))),
    GriddedCayleyPerm(cay,((3,3),(3,3))),
    ]

requirements = [[GriddedCayleyPerm(point,((1,1),))],[GriddedCayleyPerm(point,((3,3),))]]

T1 = Tiling(obstructions,requirements,(5,5))
#print(T1)
P0 = Parameter(T1,RowColMap({0:0,1:0,2:0,3:0,4:0},{0:0,1:0,2:0,3:0,4:0}))

new_obstructions = [
    GriddedCayleyPerm(point,((4,0),)),
    GriddedCayleyPerm(point,((4,2),)),
    GriddedCayleyPerm(point,((4,3),)),
    GriddedCayleyPerm(point,((5,0),)),
    GriddedCayleyPerm(point,((5,2),)),
    GriddedCayleyPerm(point,((5,4),)),
    GriddedCayleyPerm(point,((6,3),)),
    GriddedCayleyPerm(point,((6,4),)),
    GriddedCayleyPerm(des2,((5,3),(5,3))),
    GriddedCayleyPerm(asc2,((5,3),(5,3))),
    GriddedCayleyPerm(cay,((5,3),(5,3)))
]
new_requirements = [[GriddedCayleyPerm(point,((5,3),))]]

P1 = Parameter(Tiling(new_obstructions,new_requirements,(7,7)),RowColMap({0:0,1:1,2:2,3:3,4:4,5:4,6:4},{0:0,1:1,2:2,3:2,4:2,5:3,6:4}))
P1 = P1.back_map_obs_and_reqs(T1)
#print(P1.ghost)

P2 = Parameter(Tiling([GriddedCayleyPerm(point,((5,6),)),GriddedCayleyPerm(point,((7,6),))],[],(9,9)),RowColMap({4:0,5:1,6:2,7:3,8:4},{0:0,1:1,2:2,3:3,4:4,5:4}))
P2 = Parameter(P2.back_map_obs_and_reqs(T1).ghost,RowColMap({0:0,1:1,2:2,3:3,4:4,5:4,6:4,7:4,8:4},{0:0,1:0,2:0,3:0,4:0,5:1,6:2,7:3,8:4}))
P2 = P2.back_map_obs_and_reqs(T1)
#print(P2.ghost)

T0 = Tiling([GriddedCayleyPerm(asc3,((0,0),(0,0),(0,0)))],[[GriddedCayleyPerm(point,((0,0),))]],(1,1))

M0 = MappedTiling(T0,[],[[P0]],[])

M1 = MappedTiling(T1,[P1,P2],[],[])


#print(M1)
print(MTFactor(M1).is_factorable(MTFactor(M1).find_factor_cells()))

#for factor in MTFactor(M1).find_factors():
#    print("-------------------------------------")
#    print(factor)









#T = Tiling([GriddedCayleyPerm(CayleyPermutation([0,1,2]),((0,0),(0,0),(0,0)))],[[GriddedCayleyPerm(point,((0,0),))]])

#M5 = MappedTiling(T,[],[[P1]],[])

#ParameterPlacement(M5,P1,(0,0))

