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
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1},
    ),
)

mappling = MappedTiling(base_tiling, [], [[param]], [])

print(mappling)
# print("=" * 150)

cell = (0, 0)
output = ParameterPlacement(mappling, param, cell).param_placement(3, 0)
print(output)

# print(output.tiling)

# from gridded_cayley_permutations.point_placements import PointPlacement

# print(
#     PointPlacement(output.tiling)
#     .directionless_point_placement((1, 1))
#     .remove_empty_rows_and_columns()
# )


# gcps = mesh_pattern.requirements
# # print(gcps)

# print(PointPlacement(output.tiling).point_placement_in_cell(gcps, [0, 0, 0], 0, (0, 0)))
