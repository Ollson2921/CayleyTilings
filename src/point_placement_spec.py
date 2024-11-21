from tilescope_folder import TileScope, TileScopePack
from comb_spec_searcher.rule_db import RuleDBForest
from gridded_cayley_permutations import Tiling, GriddedCayleyPerm
from cayley_permutations import CayleyPermutation
from gridded_cayley_permutations.mapped_tiling import Parameter, MappedTiling
from gridded_cayley_permutations.row_col_map import RowColMap




P1 = Parameter(
    Tiling(
        [
            GriddedCayleyPerm(CayleyPermutation([0, 1]), [(1, 0), (1, 0)]),
            GriddedCayleyPerm(CayleyPermutation([0, 1]), [(0, 0), (0, 0)]),
            GriddedCayleyPerm(CayleyPermutation([0, 1]), [(1, 0), (1, 0)]),
        ],
        [],
        (3, 1),
    ),
    RowColMap({0: 0, 1: 0, 2: 1}, {0: 0}),
)
P2 = Parameter(
    Tiling(
        [GriddedCayleyPerm(CayleyPermutation([1, 0]), [(0, 0), (1, 0)])], [], (2, 1)
    ),
    RowColMap({0: 0}, {0: 0, 1: 0}),
)

M = MappedTiling(
    Tiling(
        [GriddedCayleyPerm(CayleyPermutation([0, 1, 2]), [(0, 0), (0, 0), (0, 0)])],
        [],
        (2, 1),
    ),
    [P1],
)



tiling = Tiling(
    [GriddedCayleyPerm(CayleyPermutation([0]), [(0,0)]),GriddedCayleyPerm(CayleyPermutation([0]), [(1,1)])],
    [],
    (2,2)
)
ghost = Tiling([],[],(2,2))
#tiling = tiling.add_requirements([[GriddedCayleyPerm(CayleyPermutation([0]), [(0,1)])],[GriddedCayleyPerm(CayleyPermutation([0]), [(1,0)])]])
M2 = MappedTiling(ghost,[Parameter(tiling, RowColMap({0:0,1:1},{0:0,1:1}))])
# print(
#     M.add_obstructions(
#         [GriddedCayleyPerm(CayleyPermutation([0, 2, 1]), [(0, 0), (0, 0), (0, 0)])]
#     ).parameters
# )
#print(M)

for factor in M2.find_factors():
    print('-------------------------------------')
    print(factor)
# for param in M.parameters:
#     print(param.param)
#     print(param.map)

# placed_point = M.point_placement((0, 0), 0).remove_empty_rows_and_columns()
# print(placed_point)
# for param in placed_point.parameters:
#     print(param.param)
#     print(param.map)

# print(M)
# basis = "000"

# basis_patterns = [CayleyPermutation.standardise(p) for p in basis.split("_")]

# rules = []
# tiling = Tiling(
#     [GriddedCayleyPerm(p, [(0, 0) for _ in p]) for p in basis_patterns],
#     [],
#     (1, 1),
# )
# ruledb = RuleDBForest(reverse=False)
# scope = TileScope(tiling, TileScopePack.point_placement(), debug=False, ruledb=ruledb)
# spec = scope.auto_search()
# print(spec)
# spec.show()
# # for i in range(10):
# #     print(spec.count_objects_of_size(i))
# # print(spec.get_genf())
