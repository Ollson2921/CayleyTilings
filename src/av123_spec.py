# from tilescope import TileScope, TileScopePack, strategies
from tilescope_folder import TileScope, TileScopePack, strategies
from comb_spec_searcher.rule_db import RuleDBForest
from gridded_cayley_permutations import Tiling, GriddedCayleyPerm
from cayley_permutations import CayleyPermutation
from gridded_cayley_permutations.point_placements import (
    PointPlacement,
    PartialPointPlacements,
    Directions,
    Right_bot,
    Left,
    Right,
    Left_bot,
    Left_top,
    Right_top,
)


basis = "012_00"
basis_patterns = [CayleyPermutation.standardise(p) for p in basis.split("_")]

# rules = []
# tiling = Tiling(
#     [GriddedCayleyPerm(p, [(0, 0) for _ in p]) for p in basis_patterns],
#     [],
#     (1, 1),
# )


# print("START:")
# print(tiling)

# for strategy in strategies.RowInsertionFactory()(tiling):
#     print(strategy)
#     print(strategy(tiling))
#     print("")

# tiling = Tiling(
#     [GriddedCayleyPerm(p, [(1, 1) for _ in p]) for p in basis_patterns],
#     [],
#     (3, 3),
# )

tiling = Tiling(
    [
        GriddedCayleyPerm(CayleyPermutation([0, 1]), [(0, 1), (0, 1)]),
        GriddedCayleyPerm(CayleyPermutation([0, 1]), [(1, 1), (1, 1)]),
        GriddedCayleyPerm(CayleyPermutation([0, 1]), [(1, 2), (1, 2)]),
        GriddedCayleyPerm(CayleyPermutation([0, 1]), [(0, 2), (0, 2)]),
        GriddedCayleyPerm(CayleyPermutation([0, 1]), [(0, 1), (0, 2)]),
        GriddedCayleyPerm(CayleyPermutation([0, 1]), [(1, 1), (1, 2)]),
        GriddedCayleyPerm(CayleyPermutation([0, 1]), [(0, 1), (0, 3)]),
        GriddedCayleyPerm(CayleyPermutation([0, 1]), [(0, 2), (0, 3)]),
        GriddedCayleyPerm(CayleyPermutation([2, 1, 0]), [(0, 3), (0, 1), (0, 0)]),
        GriddedCayleyPerm(CayleyPermutation([2, 1, 0]), [(0, 3), (0, 2), (0, 0)]),
        # GriddedCayleyPerm(CayleyPermutation([0, 1, 2]), [(0, 0), (0, 1), (0, 2)]),
    ],
    [],
    (2, 4),
)
print(tiling)
# print(tiling.sub_tiling([(1, 1), (1, 2)]))
print(tiling.fuse(1, 1))
print(tiling.fuse(1, 1).obstructions)
# gcp = GriddedCayleyPerm(CayleyPermutation([0, 1]), [(0, 1), (0, 1)])
# for g in gcp.shifts(1, 0):
#     print(g)
