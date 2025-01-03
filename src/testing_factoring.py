from gridded_cayley_permutations import Tiling, GriddedCayleyPerm
from cayley_permutations import CayleyPermutation
from mapplings import Parameter, MappedTiling
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

P1 = Parameter(
    ghost_tiling.add_obstructions(
        [
            # GriddedCayleyPerm(CayleyPermutation([0]), ((0, 1),)),
            GriddedCayleyPerm(CayleyPermutation([0]), ((1, 0),)),
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
            # GriddedCayleyPerm(CayleyPermutation([0]), ((1, 1),)),
            GriddedCayleyPerm(CayleyPermutation([0]), ((2, 0),)),
            GriddedCayleyPerm(CayleyPermutation([0]), ((3, 0),)),
        ]
    ),
    RowColMap({0: 0, 1: 0, 2: 1, 3: 2}, {0: 0, 1: 1}),
)


M = MappedTiling(base_tiling, [P1, P2])
M = MappedTiling(base_tiling, [P2, P1])
M.reap_contradictory_ghosts()
print(M)
# print("FACTORS:")
# for factor in M.find_factors():
#     print("-------------------------------------")
#     print(factor)
