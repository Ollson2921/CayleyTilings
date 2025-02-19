from typing import Dict, Tuple, List, Iterator
from cayley_permutations import CayleyPermutation
from gridded_cayley_permutations.point_placements import (
    PointPlacement,
    Left,
    Right,
)
from gridded_cayley_permutations import GriddedCayleyPerm, Tiling
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

    def point_placement(
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
            self.point_placement_in_cell(requirement_list, indices, direction, cell)
            for cell in cells
        )

    def point_placement_in_cell(
        self,
        requirement_list: Tuple[GriddedCayleyPerm, ...],
        indices: Tuple[int, ...],
        direction: int,
        cell: Tuple[int, int],
    ) -> MappedTiling:
        base_tiling = PointPlacement(self.mappling.tiling).point_placement_in_cell(
            requirement_list, indices, direction, cell
        )
        new_avoiding_parameters = self.req_placement_in_list(
            self.mappling.avoiding_parameters,
            requirement_list,
            indices,
            direction,
            cell,
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
            new_avoiding_parameters,
            new_containing_parameters,
            new_enumeration_parameters,
        )

    def req_placement_param_list(
        self, param_lists, requirement_list, indices, direction, cell
    ):
        """Point placement in a list of lists of parameters."""
        new_parameters = []
        for param_list in param_lists:
            new_parameters.append(
                self.req_placement_in_list(
                    param_list, requirement_list, indices, direction, cell
                )
            )
        return new_parameters

    def req_placement_in_list(
        self, param_list, requirement_list, indices, direction, cell
    ):
        """Point placement in a single list of parameters.
        For a given list of lists of parameters, maps each individual
        parameter to a a new list of parameters with the requirement list
        placed in the cell given (where cell, requirement_list, etc have
        been mapped to the parameter)."""
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
        return list(new_param_list)

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

    ### Directionless point placement ###

    def directionless_point_placement(self, cell: Tuple[int, int]) -> MappedTiling:
        """Place a directionless point in the tiling and all parameters and update maps."""
        new_tiling = PointPlacement(self.mappling.tiling).directionless_point_placement(
            cell
        )
        new_avoiding_parameters = self.update_param_list(
            self.mappling.avoiding_parameters, cell
        )
        new_containing_parameters = self.update_list_of_param_lists(
            self.mappling.containing_parameters, cell
        )
        new_enumeration_parameters = self.update_list_of_param_lists(
            self.mappling.enumeration_parameters, cell
        )
        return MappedTiling(
            new_tiling,
            new_avoiding_parameters,
            new_containing_parameters,
            new_enumeration_parameters,
        )

    def update_list_of_param_lists(self, param_lists, cell):
        """Doing directionless point placements in a list of parameter lists and updating maps."""
        new_param_lists = []
        for param_list in param_lists:
            new_param_lists.append(self.update_param_list(param_list, cell))
        return new_param_lists

    def update_param_list(self, param_list, cell):
        """Doing directionless point placements in parameter list and updating maps."""
        new_param_list = []
        for parameter in param_list:
            new_cells = parameter.map.preimage_of_cell(cell)
            #print(parameter)
            for new_cell in new_cells:
                new_ghost = PointPlacement(parameter.ghost).directionless_point_placement(
                new_cell
                )
                if new_ghost.is_empty():
                    continue
                new_param = self.new_parameter_from_point_placed_tiling(
                    parameter, new_ghost, new_cell
                )
                new_param_list.append(new_param)
        return new_param_list

    def new_parameter_from_point_placed_tiling(
        self, parameter: Parameter, new_ghost: Tiling, cell: Tuple[int, int]
    ) -> Parameter:
        """For a given parameter and a tiling after a point has been placed
        in the cell of the parameter, returns a new parameter with the new tiling
        and correct map."""
        n, m = parameter.ghost.dimensions
        new_n, new_m = new_ghost.dimensions
        new_map = parameter.expand_row_col_map_at_index(
            new_n - n, new_m - m, cell[0], cell[1]
        )
        return Parameter(new_ghost, new_map)


class MTPartialPointPlacements(MTRequirementPlacement):
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
