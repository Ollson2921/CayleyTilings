from gridded_cayley_permutations import Tiling, GriddedCayleyPerm
from cayley_permutations import CayleyPermutation
from mapplings import Parameter, MappedTiling
from gridded_cayley_permutations.row_col_map import RowColMap
from mapplings import MTRequirementPlacement

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
    # [[GriddedCayleyPerm(CayleyPermutation((0,)), [(2, 1)])]],
    [],
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

new_ghost_tiling = Tiling(
    base_obs + extra_obs,
    [[GriddedCayleyPerm(CayleyPermutation([0]), ((3, 1),))]],
    (5, 2),
)
P4 = Parameter(
    new_ghost_tiling.add_obstructions(
        [
            # GriddedCayleyPerm(CayleyPermutation([0]), ((0, 1),)),
            GriddedCayleyPerm(CayleyPermutation([0]), ((1, 0),)),
            # GriddedCayleyPerm(CayleyPermutation([0]), ((0, 1),)),
            GriddedCayleyPerm(CayleyPermutation([0]), ((2, 0),)),
            GriddedCayleyPerm(CayleyPermutation([0]), ((3, 0),)),
        ]
    ),
    RowColMap({0: 0, 1: 1, 2: 1, 3: 1, 4: 2}, {0: 0, 1: 1}),
)


# M = MappedTiling(base_tiling, [P1, P2], [[P1]], [])
M = MappedTiling(base_tiling, [], [[]], [[P4], [P1]])

print(M)


cell = (1, 1)
gcps = (
    GriddedCayleyPerm(
        CayleyPermutation([0]),
        [cell],
    ),
)
indices = (0,)
direction = 1
for mt in MTRequirementPlacement(M).point_placement(gcps, indices, 0):
    print(mt)
    print("-" * 20)
# print(
#     MTRequirementPlacement(M).point_placement_in_cell(gcps, indices, 6, (0, 0))
#     == MTRequirementPlacement(M).directionless_point_placement((0, 0))
# )
