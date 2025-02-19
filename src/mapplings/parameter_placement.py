from mapplings import MappedTiling, Parameter
from gridded_cayley_permutations import Tiling, GriddedCayleyPerm
from cayley_permutations import CayleyPermutation
from typing import Tuple, Dict
from gridded_cayley_permutations.point_placements import PointPlacement
from gridded_cayley_permutations.row_col_map import RowColMap
from .MT_point_placement import MTRequirementPlacement


class ParameterPlacement:
    """For a given mappling and containing parameter, places the parameter in the base tiling of the mappling."""

    def __init__(
        self, mappling: MappedTiling, param: Parameter, cell: Tuple[int, int]
    ) -> None:
        """cell is the cell in the tiling which the parameter will be placed into"""
        self.mappling = mappling
        self.param = param
        self.cell = cell

    def param_placement(self, direction: int, index_of_pattern: int) -> MappedTiling:
        """Place a parameter in the tiling."""
        """index_of_pattern is the index of the pattern that is placed in the tiling and is 0 based."""
        temp_containing_parameters = self.mappling.containing_parameters.copy()
        if [self.param] in temp_containing_parameters:
            temp_containing_parameters.remove([self.param])
        new_mappling = MTRequirementPlacement(
            MappedTiling(self.mappling.tiling, 
                         self.mappling.avoiding_parameters, 
                         temp_containing_parameters, 
                         self.mappling.enumeration_parameters)
            ).directionless_point_placement(self.cell)
        #print(new_mappling)
        new_avoiding_parameters = (
            new_mappling.avoiding_parameters
            + self.find_new_avoiding_parameters(direction, index_of_pattern)
        )
        
        new_containing_parameters = self.update_containing_parameters(
            index_of_pattern, new_mappling.containing_parameters
        )
        new_enumeration_parameters = new_mappling.enumeration_parameters

        return MappedTiling(
            new_mappling.tiling,
            new_avoiding_parameters,
            new_containing_parameters,
            new_enumeration_parameters,
        )

    def find_new_avoiding_parameters(self, direction: int, index_of_pattern: int):
        """Return a list of new avoiding parameters for the new mappling."""
        cells_to_insert_in = self.cells_to_insert_point_in(direction, index_of_pattern)
        new_avoiding_parameters = []
        for cell in cells_to_insert_in:
            new_ghost = PointPlacement(self.param.ghost).directionless_point_placement(
                cell
            )
            new_avoiding_parameters.append(
                MTRequirementPlacement(
                    self.mappling
                ).new_parameter_from_point_placed_tiling(self.param, new_ghost, cell)
            )
        return new_avoiding_parameters

    def new_cell_block_in_tiling(self):
        return [
            (i, j)
            for i in range(self.cell[0], self.cell[0] + 3)
            for j in range(self.cell[1], self.cell[1] + 3)
        ]

    def cells_to_insert_point_in(self, direction, index_of_pattern):
        """Returns a list of cells in the parameter which a point can be
        placed into for the resulting tiling to be an avoiding parameter (cells
        which are not further in the given direction so that the pattern in the
        parameter will be, therefore it must be avoided.)"""
        all_cells = [
            cell
            for cell in self.cells_in_parameter()
            if GriddedCayleyPerm(CayleyPermutation([0]), [cell])
            not in self.param.ghost.obstructions
        ]
        cell_of_point_being_placed = self.cell_of_inserted_point_in_param(
            index_of_pattern
        )
        cells_to_insert_in = [
            cell
            for cell in all_cells
            if PointPlacement(self.param.ghost).farther(
                cell_of_point_being_placed, cell, direction
            )
        ]
        return cells_to_insert_in

    def place_point_in_base_tiling(self) -> MappedTiling:
        """Place a directionless point in the base tiling in self.cell."""
        return PointPlacement(self.mappling.tiling).directionless_point_placement(
            self.cell
        )

    def cells_in_parameter(self):
        """The cells in the parameter that are being placed in the
        tiling at the cell specified."""
        return self.param.map.preimage_of_cell(self.cell)

    def update_containing_parameters(
        self, index_of_pattern: int, containing_parameters
    ):
        """Remove [self.param] from containing parameters and add new
        containing parameter list (one that is the identity)."""
        new_map = self.new_containing_param_map(index_of_pattern)
        containing_param = [Parameter(self.param.ghost, new_map)]
        containing_parameters.append(containing_param)
        return containing_parameters

    def new_containing_param_map(self, index_of_pattern: int):
        """Return a new RowColMap for the containing parameter."""
        middle_cell = self.cell_of_inserted_point_in_param(index_of_pattern)
        row_map = self.param.map.row_map.copy()
        col_map = self.param.map.col_map.copy()

        new_row_map = self.adjust_dict_in_param(middle_cell, 1, row_map)
        new_col_map = self.adjust_dict_in_param(middle_cell, 0, col_map)
        return RowColMap(new_col_map, new_row_map)

    def adjust_dict_in_param(
        self, middle_cell: Tuple[int, int], row_or_col: int, new_map: Dict[int, int]
    ):
        vals_in_param = set(cell[row_or_col] for cell in self.cells_in_parameter())
        middle_val = middle_cell[row_or_col]
        for val in range(self.param.ghost.dimensions[row_or_col]):
            if val not in vals_in_param:
                if val > middle_val:
                    new_map[val] += 2
            elif val < middle_val:
                new_map[val] = self.cell[row_or_col]
            elif val > middle_val:
                new_map[val] = self.cell[row_or_col] + 2
            else:
                new_map[val] = self.cell[row_or_col] + 1
        return new_map

    def cell_of_inserted_point_in_param(self, index_of_pattern: int):
        for cell in self.cells_in_parameter():
            if (
                cell[0] == index_of_pattern * 2 + 1
                and GriddedCayleyPerm(CayleyPermutation([0]), [cell])
                not in self.param.ghost.obstructions
            ):
                return cell
