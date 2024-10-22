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

rules = []
tiling = Tiling(
    [GriddedCayleyPerm(p, [(0, 0) for _ in p]) for p in basis_patterns],
    [],
    (1, 1),
)


print("START:")
print(tiling)

# for strategy in strategies.RowInsertionFactory()(tiling):
#     print(strategy)
#     print(strategy(tiling))
#     print("")

strategy = strategies.RequirementPlacementStrategy(
    (GriddedCayleyPerm(CayleyPermutation([0]), [(0, 0)]),), (0,), 4
)
print(strategy)
print(strategy(tiling))
rule = strategy(tiling)
tiling1 = rule.children[1]

strategy = strategies.FactorStrategy()
rule2 = strategy(tiling1)
print(rule2)
tiling2 = rule2.children[0]

strategy = strategies.RemoveEmptyRowsAndColumnsStrategy()
rule3 = strategy(tiling2)
print(rule3)
tiling3 = rule3.children[0]


strategy = strategies.RequirementPlacementStrategy(
    (
        GriddedCayleyPerm(CayleyPermutation([0]), [(0, 0)]),
        GriddedCayleyPerm(CayleyPermutation([0]), [(1, 0)]),
    ),
    (0, 0),
    4,
)
print(strategy)
rule4 = strategy(tiling3)
print(rule4)
tiling4, tiling5, tiling6 = rule4.children

strategy = strategies.RemoveEmptyRowsAndColumnsStrategy()
rule5 = strategy(tiling5)
print(rule5)
tiling7 = rule5.children[0]

strategy = strategies.RemoveEmptyRowsAndColumnsStrategy()
rule6 = strategy(tiling6)
print(rule6)
tiling8 = rule6.children[0]

strategy = strategies.FactorStrategy()
rule7 = strategy(tiling7)
print(rule7)
tiling9 = rule7.children[0]

strategy = strategies.RemoveEmptyRowsAndColumnsStrategy()
rule8 = strategy(tiling9)
print(rule8)
tiling10 = rule8.children[0]

# strategy = strategies.FusionStrategy(0, 1)
# rule9 = strategy(tiling10)
# print(rule9)

for strategy in strategies.FusionFactory()(tiling10):
    print(strategy)
    print(strategy(tiling10))
    print("")


# strategy = strategies.FusionStrategy(1, 0)(tiling)
# print(strategy)

# tiling = Tiling(
#     [GriddedCayleyPerm(p, [(1, 1) for _ in p]) for p in basis_patterns],
#     [],
#     (3, 3),
# )

# tiling = Tiling(
#     [
#         GriddedCayleyPerm(CayleyPermutation([0, 1]), [(0, 1), (0, 1)]),
#         GriddedCayleyPerm(CayleyPermutation([0, 1]), [(1, 1), (1, 1)]),
#         GriddedCayleyPerm(CayleyPermutation([0, 1]), [(1, 2), (1, 2)]),
#         GriddedCayleyPerm(CayleyPermutation([0, 1]), [(0, 2), (0, 2)]),
#         GriddedCayleyPerm(CayleyPermutation([0, 1]), [(0, 1), (0, 2)]),
#         GriddedCayleyPerm(CayleyPermutation([0, 1]), [(1, 1), (1, 2)]),
#         GriddedCayleyPerm(CayleyPermutation([0, 1]), [(0, 1), (0, 3)]),
#         GriddedCayleyPerm(CayleyPermutation([0, 1]), [(0, 2), (0, 3)]),
#         GriddedCayleyPerm(CayleyPermutation([2, 1, 0]), [(0, 3), (0, 1), (0, 0)]),
#         GriddedCayleyPerm(CayleyPermutation([2, 1, 0]), [(0, 3), (0, 2), (0, 0)]),
#         # GriddedCayleyPerm(CayleyPermutation([0, 1, 2]), [(0, 0), (0, 1), (0, 2)]),
#     ],
#     [],
#     (2, 4),
# )
# print(tiling)
# # print(tiling.sub_tiling([(1, 1), (1, 2)]))
# print(tiling.fuse(1, 1))
# print(tiling.fuse(1, 1).obstructions)
# gcp = GriddedCayleyPerm(CayleyPermutation([0, 1]), [(0, 1), (0, 1)])
# for g in gcp.shifts(1, 0):
#     print(g)

# strategy = strategies.FusionStrategy(1, 1)(tiling)
# print(strategy)


# for strategy in strategies.FusionFactory()(tiling):
#     print(strategy)
#     print(strategy(tiling))
#     print("")
