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

GCP = GriddedCayleyPerm(CayleyPermutation((0,1,1,2,0)) , ((0,0),(0,1),(1,1),(1,1),(2,0)))

print("Column 0 Shifts")
for g in GCP.shifts(0,0):
    print(g)
print("Row 0 Shifts")
for g in GCP.shifts(0,1):
    print(g)