from gridded_cayley_permutations import Tiling, GriddedCayleyPerm
from cayley_permutations import CayleyPermutation
from mapplings import Parameter, MappedTiling
from gridded_cayley_permutations.row_col_map import RowColMap
from itertools import combinations_with_replacement

all_obs = []
for i in [1, 3, 5]:
    for j in range(7):
        if i == j:
            continue
        all_obs.append(GriddedCayleyPerm(CayleyPermutation([0]), ((i, j),)))
for i in range(7):
    all_obs.append(GriddedCayleyPerm(CayleyPermutation([0]), ((i, 4),)))

reqs = []
for i in [1, 3, 5]:
    all_obs.append(GriddedCayleyPerm(CayleyPermutation([0, 0]), ((i, i), (i, i))))
    all_obs.append(GriddedCayleyPerm(CayleyPermutation([0, 1]), ((i, i), (i, i))))
    all_obs.append(GriddedCayleyPerm(CayleyPermutation([1, 0]), ((i, i), (i, i))))
    reqs.append([GriddedCayleyPerm(CayleyPermutation([0]), ((i, i),))])

for i in [1, 3, 5]:
    cells_in_row = []
    for j in range(7):
        cells_in_row.append((j, i))
    for subset in combinations_with_replacement(cells_in_row, 2):
        all_obs.append(GriddedCayleyPerm(CayleyPermutation([0, 1]), subset))
        all_obs.append(GriddedCayleyPerm(CayleyPermutation([1, 0]), subset))

# print(all_obs)
ghost = Tiling(all_obs, reqs, (7, 7))
# print(ghost)
avoiding_parameters = [
    Parameter(ghost, RowColMap({i: 0 for i in range(7)}, {i: 0 for i in range(7)}))
]
mappling = MappedTiling(
    Tiling(
        [
            GriddedCayleyPerm(CayleyPermutation([0, 2, 1]), ((0, 0), (0, 0), (0, 0))),
            GriddedCayleyPerm(CayleyPermutation((0, 0)), [(0, 0), (0, 0)]),
        ],
        [],
        (1, 1),
    ),
    avoiding_parameters,
    [],
    [],
)

new_mappling = mappling.point_placement((0, 0), 0)
print(new_mappling)

# print(mappling)
for i in range(10):
    print(mappling.get_terms(i))
    # for gcp in mappling.get_objects(i)[tuple()]:
    #     print(gcp.pattern)

# gcp = GriddedCayleyPerm(
#     CayleyPermutation((2, 0, 1, 3)), [(0, 0), (0, 0), (0, 0), (0, 0)]
# )
# for preimage in avoiding_parameters[0].preimage_of_gcp(gcp):
#     print(preimage)
