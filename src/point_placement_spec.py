from tilescope_folder import TileScope, TileScopePack
from comb_spec_searcher.rule_db import RuleDBForest
from gridded_cayley_permutations import Tiling, GriddedCayleyPerm
from cayley_permutations import CayleyPermutation


basis = "1234"
basis_patterns = [CayleyPermutation.standardise(p) for p in basis.split("_")]

rules = []
tiling = Tiling(
    [GriddedCayleyPerm(p, [(0, 0) for _ in p]) for p in basis_patterns],
    [],
    (1, 1),
)
ruledb = RuleDBForest(reverse=False)
scope = TileScope(tiling, TileScopePack.point_placement(), debug=False, ruledb=ruledb)
spec = scope.auto_search()
print(spec)
spec.show()
# for i in range(10):
#     print(spec.count_objects_of_size(i))
# print(spec.get_genf())
