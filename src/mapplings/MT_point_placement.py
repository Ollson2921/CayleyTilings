from typing import Dict, Tuple, List, Iterator
from cayley_permutations import CayleyPermutation
from gridded_cayley_permutations.point_placements import (
    PointPlacement,
    Left,
    Right,
)
from gridded_cayley_permutations import GriddedCayleyPerm
from cayley_permutations import CayleyPermutation
from mapplings import MappedTiling, Parameter
from gridded_cayley_permutations import GriddedCayleyPerm
from gridded_cayley_permutations.point_placements import (
    MultiplexMap,
    PartialMultiplexMap,
    PointPlacement,
)


class MTRequirementPlacement:
    def __init__(self, mappling: MappedTiling) -> None:
        self.mappling = mappling

    def req_placement(
        self,
        requirement_list: Tuple[GriddedCayleyPerm, ...],
        indices: Tuple[int, ...],
        direction: int,
    ) -> Tuple[MappedTiling, ...]:
        cells = []
        for idx, gcp in zip(indices, requirement_list):
            cells.append(gcp.positions[idx])
        cells = sorted(set(cells))
        return tuple(
            self.req_placement_in_cell(requirement_list, indices, direction, cell)
            for cell in cells
        )

    def req_placement_in_cell(
        self,
        requirement_list: Tuple[GriddedCayleyPerm, ...],
        indices: Tuple[int, ...],
        direction: int,
        cell: Tuple[int, int],
    ) -> MappedTiling:
        base_tiling = PointPlacement(self.mappling.tiling).point_placement_in_cell(
            requirement_list, indices, direction, cell
        )
        new_containing_parameters = self.req_placement_param_list(
            self.mappling.containing_parameters,
            requirement_list,
            indices,
            direction,
            cell,
        )
        new_enumeration_parameters = self.req_placement_param_list(
            self.mappling.enumeration_parameters,
            requirement_list,
            indices,
            direction,
            cell,
        )
        return MappedTiling(
            base_tiling,
            self.mappling.avoiding_parameters,
            new_containing_parameters,
            new_enumeration_parameters,
        )

    def req_placement_param_list(
        self, param_lists, requirement_list, indices, direction, cell
    ):
        """For a given list of lists of parameters, maps each individual
        parameter to a a new list of parameters with the requirement list
        placed in the cell given (where cell, requirement_list, etc have
        been mapped to the parameter)."""
        new_parameters = []
        for param_list in param_lists:
            new_param_list = set()
            for parameter in param_list:
                n, m = parameter.ghost.dimensions
                param_requirement_list, param_indices = (
                    self.map_requirement_list_to_parameter(
                        requirement_list, indices, parameter
                    )
                )
                for param_cell in parameter.map.preimage_of_cell(cell):
                    new_ghost = PointPlacement(parameter.ghost).point_placement_in_cell(
                        param_requirement_list, param_indices, direction, param_cell
                    )
                    if new_ghost.is_empty():
                        continue
                    new_n, new_m = new_ghost.dimensions
                    new_map = parameter.expand_row_col_map_at_index(
                        new_n - n, new_m - m, param_cell[0], param_cell[1]
                    )
                    new_param_list.add(Parameter(new_ghost, new_map))
            new_parameters.append(list(new_param_list))
        return new_parameters

    def map_requirement_list_to_parameter(self, requirement_list, indices, parameter):
        """Maps each requirement in a requirement list to a new requirement based on
        parameter.map and creates new requirement list for the parameter. Also turns
        indices into a list length len(new_requirement_list), with one occurrence of each
        index for each requirement in new_requirement_list."""
        new_requirement_list = []
        new_indices = []
        for idx, gcp in zip(indices, requirement_list):
            for stretched_gcp in parameter.map.preimage_of_gridded_cperm(gcp):
                new_requirement_list.append(stretched_gcp)
                new_indices.append(idx)
        return new_requirement_list, new_indices


class PartialPointPlacements(MTRequirementPlacement):
    """TODO: update for mapplings"""

    DIRECTIONS = [Left, Right]

    def point_obstructions_and_requirements(
        self, cell: Tuple[int], direction: int
    ) -> Tuple[
        Tuple[GriddedCayleyPerm, ...] | Tuple[Tuple[GriddedCayleyPerm, ...], ...]
    ]:
        cell = self.placed_cell(cell)
        _, y = self.new_dimensions()
        col_obs = [
            GriddedCayleyPerm(CayleyPermutation([0]), [(cell[0], i)])
            for i in range(y)
            if i != cell[1]
        ]
        return [
            [
                GriddedCayleyPerm(CayleyPermutation((0, 1)), [cell, cell]),
                GriddedCayleyPerm(CayleyPermutation((0, 0)), [cell, cell]),
                GriddedCayleyPerm(CayleyPermutation((1, 0)), [cell, cell]),
            ]
            + col_obs,
            [[GriddedCayleyPerm(CayleyPermutation([0]), [cell])]],
        ]

    def placed_cell(self, cell: Tuple[int]) -> Tuple[int]:
        return (cell[0] + 1, cell[1])

    def multiplex_map(self, cell: Tuple[int]) -> MultiplexMap:
        return PartialMultiplexMap(cell, self.mappling.tiling.dimensions)

    def new_dimensions(self):
        return (
            self.mappling.tiling.dimensions[0] + 2,
            self.mappling.tiling.dimensions[1],
        )
