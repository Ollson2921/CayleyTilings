from itertools import combinations
from typing import Dict, Tuple
from cayley_permutations import CayleyPermutation
from .row_col_map import RowColMap
from .tilings import Tiling
from .gridded_cayley_perms import GriddedCayleyPerm
from .row_col_map import OBSTRUCTIONS, REQUIREMENTS


Right = 0
Right_top = 1
Left_top = 2
Left = 3
Left_bot = 4
Right_bot = 5

Directions = [Right, Right_top, Left_top, Left, Left_bot, Right_bot]


class MultiplexMap(RowColMap):
    """
        A special class that maps
        + - + - + - +
        | A |   | A | \
        + - + - + - +    + - +
        |   | o |   | -  | A |
        + - + - + - +    + - +
        | A |   | A | /
        + - + - + - +
        where the preimage does not place points in the empty cells.
        """

    def __init__(self, cell: Tuple[int, int], dimensions: Tuple[int, int]):
        """Cell is the cell that is expanded to a three by three grid
        and dimensions are the dimenstions of the grid before it was expanded."""
        self.cell = cell
        self.dimensions = dimensions
        super().__init__(self.col_map(), self.row_map())

    def row_map(self) -> Dict[int, int]:
        """Return the row map."""
        row_width = self.dimensions[1] + 2
        row_map_dict = dict()
        for i in range(row_width):
            if i < self.cell[1]:
                row_map_dict[i] = i
            elif self.cell[1] <= i <= self.cell[1] + 2:
                row_map_dict[i] = self.cell[1]
            else:
                row_map_dict[i] = i - 2
        return row_map_dict

    def col_map(self) -> Dict[int, int]:
        """Return the col map."""
        col_width = self.dimensions[0] + 2
        col_map_dict = {}
        for i in range(col_width):
            if i < self.cell[0]:
                col_map_dict[i] = i
            elif self.cell[0] <= i <= self.cell[0] + 2:
                col_map_dict[i] = self.cell[0]
            else:
                col_map_dict[i] = i - 2
        return col_map_dict


class PartialMultiplexMap(MultiplexMap):
    def row_map(self) -> Dict[int, int]:
        """Return the row map."""
        return {i: i for i in range(self.dimensions[1])}


class PointPlacement:
    DIRECTIONS = [Right, Right_top, Left_top, Left, Left_bot, Right_bot]

    def __init__(self, tiling: Tiling) -> None:
        self.tiling = tiling

    def point_obstructions_and_requirements(
        self, cell: Tuple[int, int]
    ) -> Tuple[OBSTRUCTIONS, REQUIREMENTS]:
        cell = self.placed_cell(cell)
        x, y = self.new_dimensions()
        col_obs = [
            GriddedCayleyPerm(CayleyPermutation([0]), [(cell[0], i)])
            for i in range(y)
            if i != cell[1]
        ]
        row_obs = []
        row = cell[1]
        for col in range(x):
            row_obs.append(
                GriddedCayleyPerm(CayleyPermutation([0, 1]), [(col, row), (col, row)])
            )
            row_obs.append(
                GriddedCayleyPerm(CayleyPermutation([1, 0]), [(col, row), (col, row)])
            )
        for col1, col2 in combinations(range(x), 2):
            row_obs.append(
                GriddedCayleyPerm(CayleyPermutation([0, 1]), [(col1, row), (col2, row)])
            )
            row_obs.append(
                GriddedCayleyPerm(CayleyPermutation([1, 0]), [(col1, row), (col2, row)])
            )

        return [
            [
                GriddedCayleyPerm(CayleyPermutation((0, 1)), [cell, cell]),
                GriddedCayleyPerm(CayleyPermutation((0, 0)), [cell, cell]),
                GriddedCayleyPerm(CayleyPermutation((1, 0)), [cell, cell]),
            ]
            + col_obs
            + row_obs,
            [[GriddedCayleyPerm(CayleyPermutation([0]), [cell])]],
        ]

    def placed_cell(self, cell: Tuple[int, int]) -> Tuple[int, int]:
        return (cell[0] + 1, cell[1] + 1)

    def multiplex_map(self, cell: Tuple[int, int]) -> MultiplexMap:
        return MultiplexMap(cell, self.tiling.dimensions)

    def forced_obstructions(
        self,
        cell: Tuple[int, int],
        requirement_list: Tuple[GriddedCayleyPerm, ...],
        indices: Tuple[int, ...],
        direction: int,
    ) -> OBSTRUCTIONS:
        multiplex_map = self.multiplex_map(cell)
        cell = self.placed_cell(cell)
        obstructions = []
        for idx, gcp in zip(indices, requirement_list):
            for stretched_gcp in multiplex_map.preimage_of_gridded_cperm(gcp):
                if self.farther(stretched_gcp.positions[idx], cell, direction):
                    obstructions.append(stretched_gcp)
        return obstructions

    @staticmethod
    def farther(cell1: Tuple[int, int], cell2: Tuple[int, int], direction: int) -> bool:
        """Return True if cell1 is farther in the given direction than cell2."""
        if direction == Right:
            return cell1[0] > cell2[0]
        if direction == Right_top:
            return cell1[1] > cell2[1] or (cell1[1] == cell2[1] and cell1[0] > cell2[0])
        if direction == Left_top:
            return cell1[1] > cell2[1] or (cell1[1] == cell2[1] and cell1[0] < cell2[0])
        if direction == Left:
            return cell1[0] < cell2[0]
        if direction == Left_bot:
            return cell1[1] < cell2[1] or (cell1[1] == cell2[1] and cell1[0] < cell2[0])
        if direction == Right_bot:
            return cell1[1] < cell2[1] or (cell1[1] == cell2[1] and cell1[0] > cell2[0])

    def point_placement(
        self,
        requirement_list: Tuple[GriddedCayleyPerm, ...],
        indices: Tuple[int, ...],
        direction: int,
    ) -> Tuple[Tiling, ...]:
        if direction not in self.DIRECTIONS:
            raise ValueError(f"Direction {direction} is not a valid direction.")
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
    ) -> Tiling:
        multiplex_map = self.multiplex_map(cell)
        multiplex_obs, multiplex_reqs = multiplex_map.preimage_of_tiling(self.tiling)
        point_obs, point_reqs = self.point_obstructions_and_requirements(cell)
        forced_obs = self.forced_obstructions(
            cell, requirement_list, indices, direction
        )
        obstructions = multiplex_obs + point_obs + forced_obs
        requirements = multiplex_reqs + point_reqs
        return Tiling(obstructions, requirements, self.new_dimensions())

    def new_dimensions(self):
        return (self.tiling.dimensions[0] + 2, self.tiling.dimensions[1] + 2)

    def directionless_point_placement(self, cell: Tuple[int, int]) -> Tiling:
        multiplex_map = self.multiplex_map(cell)
        multiplex_obs, multiplex_reqs = multiplex_map.preimage_of_tiling(self.tiling)
        point_obs, point_reqs = self.point_obstructions_and_requirements(cell)
        obstructions = multiplex_obs + point_obs
        requirements = multiplex_reqs + point_reqs
        return Tiling(obstructions, requirements, self.new_dimensions())


class PartialPointPlacements(PointPlacement):
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
        return PartialMultiplexMap(cell, self.tiling.dimensions)

    def new_dimensions(self):
        return (self.tiling.dimensions[0] + 2, self.tiling.dimensions[1])
