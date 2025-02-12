from mapplings import ParameterPlacement, MappedTiling, Parameter
from gridded_cayley_permutations import Tiling, GriddedCayleyPerm
from cayley_permutations import CayleyPermutation
from gridded_cayley_permutations.row_col_map import RowColMap

obs = [GriddedCayleyPerm(CayleyPermutation([0, 1, 2]), ((0, 0), (0, 0), (0, 0)))]
obs = []

base_tiling = Tiling(obs, [], (1, 2))

mesh_pattern = Tiling.from_vincular(CayleyPermutation((0, 1, 2)), (0,))

param = Parameter(
    mesh_pattern,
    RowColMap(
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
    ),
)

mappling = MappedTiling(base_tiling, [], [[param]], [])

# print(mappling)
# print("=" * 150)

cell = (0, 0)
output = ParameterPlacement(mappling, param, cell).param_placement(3, 0)
print(output)

# for n in range(1, 8):
#     param_placed_count = output.get_objects(n)
#     map_count = mappling.get_objects(n)
#     print(map_count)
#     print(param_placed_count)
#     print("Are they equal?", map_count == param_placed_count)
