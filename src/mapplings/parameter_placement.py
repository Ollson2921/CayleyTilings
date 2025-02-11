from mapplings import MappedTiling, Parameter
from gridded_cayley_permutations import Tiling, GriddedCayleyPerm
from cayley_permutations import CayleyPermutation
from typing import Tuple, Dict
from gridded_cayley_permutations.point_placements import PointPlacement
from gridded_cayley_permutations.row_col_map import RowColMap


class ParameterPlacement:
    """For a given mappling and containing parameter, places the parameter in the base tiling of the mappling.
    TODO: need to decide what happens to the other parameters in the mappling (also do point placements?).
    """

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
        new_base_tiling = self.place_point_in_base_tiling()

        cells_to_insert_in = self.cells_to_insert_point_in(direction)

        new_avoiding_parameters = []
        new_containing_parameters = self.update_containing_parameters(index_of_pattern)
        new_enumeration_parameters = []

        return MappedTiling(
            new_base_tiling,
            new_avoiding_parameters,
            new_containing_parameters,
            new_enumeration_parameters,
        )

    def new_cell_block_in_tiling(self):
        return [
            (i, j)
            for i in range(self.cell[0], self.cell[0] + 3)
            for j in range(self.cell[1], self.cell[1] + 3)
        ]

    def cells_to_insert_point_in(self, direction):
        """Returns a list of cells which the point in the parameter at index_of_pattern
        can be placed in for the pattern to be farther in the given direction than the
        point in the center (used for finding avoiding parameters)."""
        all_cells = self.new_cell_block_in_tiling()
        all_cells.remove((self.cell[0] + 1, self.cell[1] + 2))
        all_cells.remove((self.cell[0] + 1, self.cell[1]))
        cells_to_insert_in = [
            cell
            for cell in all_cells
            if PointPlacement(self.mappling.tiling).farther(
                cell, (self.cell[0] + 1, self.cell[1] + 1), direction
            )
        ]
        return cells_to_insert_in

    def place_point_in_base_tiling(self) -> MappedTiling:
        """Place a directionless point in the base tiling in self.cell."""
        return PointPlacement(self.mappling.tiling).directionless_point_placement(
            self.cell
        )

    def cells_in_parameter(self):
        return self.param.map.preimage_of_cell(self.cell)

    def update_containing_parameters(self, index_of_pattern: int):
        """Remove [self.param] from containing parameters and add new containing parameter list (one that is the identity)."""
        new_cont_params = [
            param_list
            for param_list in self.mappling.containing_parameters
            if param_list != [self.param]
        ]
        new_map = self.new_containing_param_map(index_of_pattern)
        new_cont_params.append([Parameter(self.param.ghost, new_map)])
        return new_cont_params

    def new_containing_param_map(self, index_of_pattern: int):
        """Return a new RowColMap for the containing parameter.
        TODO: Also sort this so that rest of row col map is correct (smaller vals are the same,
        anything higher increase by 2)."""
        middle_cell = self.cell_of_inserted_point_in_param(index_of_pattern)
        row_map = self.param.map.row_map.copy()
        col_map = self.param.map.col_map.copy()

        new_row_map = self.adjust_dict_in_param(middle_cell, 1, row_map)
        new_col_map = self.adjust_dict_in_param(middle_cell, 0, col_map)

        # cols_in_param = set(cell[0] for cell in self.cells_in_parameter())
        # rows_in_param = set(cell[1] for cell in self.cells_in_parameter())
        # print("cols_in_param", cols_in_param)
        # print("rows_in_param", rows_in_param)
        # other_cols_in_param = set(
        #     i for i in range(self.param.ghost.dimensions[0]) if i not in cols_in_param
        # )
        # other_rows_in_param = set(
        #     i for i in range(self.param.ghost.dimensions[1]) if i not in rows_in_param
        # )
        # print("other_cols_in_param", other_cols_in_param)
        # print("other_rows_in_param", other_rows_in_param)

        # for cell in self.cells_in_parameter():
        #     if cell[0] < middle_cell[0]:
        #         col_map[cell[0]] = self.cell[0]
        #     elif cell[0] > middle_cell[0]:
        #         col_map[cell[0]] = self.cell[0] + 2
        #     else:
        #         col_map[cell[0]] = self.cell[0] + 1

        #     if cell[1] < middle_cell[1]:
        #         row_map[cell[1]] = self.cell[1]
        #     elif cell[1] > middle_cell[1]:
        #         row_map[cell[1]] = self.cell[1] + 2
        #     else:
        #         row_map[cell[1]] = self.cell[1] + 1

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
