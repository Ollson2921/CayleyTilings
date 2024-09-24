from cayley_permutations import CayleyPermutation
from typing import Tuple, Iterable, List, Iterator
from itertools import combinations
from math import floor, ceil


class MeshPattern:
    def __init__(
        self,
        pattern: CayleyPermutation,
        shaded_cells: Iterable[Tuple[Tuple[int, int], Tuple[int, int]]],
    ):
        """Shaded regions is an iterable of shaded cells, where each cell is
        defined by a tuple.
        All cells with odd indices contain a point, so must automatically be
        shaded. To simplfy shadings from the pattern, we have not allowed shadings
        in odd indices."""
        self.pattern = pattern
        self.shaded_cells = shaded_cells
        for cell in self.shaded_cells:
            if cell[0] % 2 != 0:
                print(self.pattern, self.shaded_cells)
                print(self.ascii_plot())
                raise ValueError("Shaded cell must have even indices.")

    @classmethod
    def from_regions(
        cls,
        cperm: CayleyPermutation,
        regions: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    ):
        """Returns a mesh pattern from a Cayley permutation and a
        list of regions to be shaded rather than cells.

        Regions is an list of shaded regions, where each shaded
        region is defined by two tuples, the first defines the indices that
        the shaded region lies between and the second the values it lies between
        (on a plot where there is an empty column and row between each point).
        """
        cells = cls.regions_to_cells(regions)
        return MeshPattern(cperm, cells)

    def regions_to_cells(
        regions: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ) -> List[Tuple[int, int]]:
        """Takes an input of regions and returns them as cells."""
        cells = []
        for reg in regions:
            i1, i2 = reg[0]
            v1, v2 = reg[1]
            for i in range(i1, i2):
                for j in range(v1, v2):
                    cells.append((i, j))
        return cells

    def avoids(self, basis: List["MeshPattern"]) -> bool:
        """Returns true if the mesh pattern avoids the basis of mesh patterns."""
        for mesh_patt in basis:
            if self.contains(mesh_patt):
                return False
        return True

    def contains(self, mesh_patt: "MeshPattern") -> bool:
        """Returns true if the mesh pattern self contains the
        input mesh pattern."""
        for occ in mesh_patt.occurrences_of_pattern(self.pattern):
            if self.is_shaded_areas_map_correct(mesh_patt, occ):
                return True
        return False

    def is_shaded_areas_map_correct(self, mesh_patt: "MeshPattern", occ: List[int]):
        """Maps the shaded areas in the smaller mesh pattern to the
        larger one and returns true if all the same areas are shaded."""
        cperm_occurrence = [self.pattern.cperm[idx] for idx in occ]
        # map each cell to a range of indices and values in the larger mesh pattern
        for cell in mesh_patt.shaded_cells:
            if cell[0] == 0:
                idx1 = 0
                idx2 = occ[0] * 2 + 1
            if cell[0] == len(occ) * 2:
                idx1 = occ[-1] * 2 + 2
                idx2 = len(self.pattern) * 2 + 1
            if cell[0] != 0 and cell[0] != len(occ) * 2:
                idx1 = occ[int(cell[0] / 2) - 1] * 2 + 2
                idx2 = occ[int(cell[0] / 2)] * 2 + 1

            if cell[1] == 0:
                val1 = 0
                val2 = min(cperm_occurrence) * 2 + 1
            if cell[1] == (max(mesh_patt.pattern.cperm) + 1) * 2:
                val1 = max(cperm_occurrence) * 2 + 1
                val2 = max(self.pattern.cperm) * 2 + 1
            if cell[1] != 0 and cell[1] != (max(mesh_patt.pattern.cperm) + 1) * 2:
                for idx, val in enumerate(mesh_patt.pattern.cperm):
                    if val == ceil(cell[1] / 2) - 1:
                        new_idx = idx
                if ceil(cell[1] / 2) != int(cell[1] / 2):
                    val1 = cperm_occurrence[new_idx] * 2 + 1
                    val2 = val1 + 1
                else:
                    if new_idx != len(cperm_occurrence) - 1:
                        val1 = cperm_occurrence[new_idx] * 2 + 2
                        val2 = cperm_occurrence[new_idx + 1] * 2 + 1
                    else:
                        val1 = cperm_occurrence[new_idx - 1] * 2 + 1
                        val2 = cperm_occurrence[new_idx] * 2 + 2
            # check that cells which should be in the shaded region are in there
            for i in range(idx1, idx2, 2):
                for j in range(val1, val2):
                    if (i, j) not in self.shaded_cells:
                        return False
        return True

    def is_not_contained_in_cperm(self, cperm: CayleyPermutation) -> bool:
        """Returns true if the Cayely permutation avoids the mesh pattern."""
        return not self.is_contained_in_cperm(cperm)

    def is_contained_in_cperm(self, cperm: CayleyPermutation) -> bool:
        """Returns true if the Cayely permutation contains the mesh pattern.

        TODO: For words etc., I think we want the cperm to be a word and
        the mesh pattern to still be a cperm mesh pattern so this is fine (just change the input).
        """
        for occ in self.occurrences_of_pattern(cperm):
            if self.is_valid_occurrence(occ, cperm):
                return True
        return False

    def is_valid_occurrence(self, occ: List[int], cperm: CayleyPermutation) -> bool:
        """Occurrence is valid if shaded cells don't contain points.
        Returns true if this is the case."""
        for cell in self.shaded_cells:
            # Map index of cell to range of indices in cperm
            if floor(cell[0] / 2) - 1 < 0:
                new_idx1 = -1
            else:
                new_idx1 = occ[floor(cell[0] / 2) - 1]
            if floor(cell[0] / 2) >= len(occ):
                new_idx2 = len(cperm) + 1
            else:
                new_idx2 = occ[floor(cell[0] / 2)]
            # Map value of cell to range of values in cperm
            cperm_occurrence = [cperm.cperm[idx] for idx in occ]
            new_v1 = self.new_v(cell[1] - 1, cperm_occurrence, cperm)
            new_v2 = self.new_v(cell[1], cperm_occurrence, cperm)
            # Check that there are no points in the region mapped across from the cell
            for idx, val in enumerate(cperm.cperm):
                if new_idx1 < idx < new_idx2:
                    if new_v1 <= val < new_v2:
                        return False
        return True

    def new_v(self, v: int, cperm_occurrence: List[int], cperm: CayleyPermutation):
        """Maps the value of a cell to the value in the Cayley permutation."""
        if v < 0:
            return -1
        if v >= (max(self.pattern) + 1) * 2:
            return max(cperm.cperm) + 2
        for idx, val in enumerate(self.pattern):
            if val == floor(v / 2):
                idx1 = idx
        new_v = cperm_occurrence[idx1]
        if floor(v / 2) != v / 2:
            new_v += 0.5
        return new_v

    def occurrences_of_pattern(self, cperm: CayleyPermutation) -> List[List[int]]:
        """Returns the indices of all occurrences of the pattern
        of the mesh pattern in the Cayley permutation."""
        size = len(cperm)
        indices_of_occurrences = []
        for indices in combinations(range(size), len(self.pattern)):
            occ = [cperm.cperm[idx] for idx in indices]
            stand = cperm.standardise(occ)
            if stand == self.pattern:
                indices_of_occurrences.append(indices)
        return indices_of_occurrences

    def sub_mesh_patterns(self, max_size) -> Iterator["MeshPattern"]:
        """Returns all sub mesh patterns of the mesh pattern up to length max_size."""
        for n in range(max_size + 1):
            if n == 0:
                if len(self.pattern) == 0:
                    yield MeshPattern(self.pattern, [])
                else:
                    yield MeshPattern(self.pattern, [(0, 0)])
                continue
            for indices in combinations(range(len(self.pattern)), n):
                sub_cperm = CayleyPermutation.standardise(
                    [self.pattern.cperm[idx] for idx in indices]
                )
                shaded_cells = []
                for i in range(0, len(sub_cperm) * 2 + 1, 2):
                    for j in range(0, (max(sub_cperm) + 1) * 2 + 1):
                        cell = (i, j)
                        if not MeshPattern(sub_cperm, [cell]).is_valid_occurrence(
                            indices, self.pattern
                        ):
                            shaded_cells.append(cell)
                yield MeshPattern(sub_cperm, shaded_cells)

    def complement(self) -> "MeshPattern":
        """Returns the complement of the mesh pattern - anything
        that was shaded now isn't, anything that wasn't now is
        apart from columns which currently remain unshaded."""
        mesh_patt = self
        cells_before = mesh_patt.shaded_cells
        cells_after = []
        for i in range(0, len(mesh_patt.pattern) * 2 + 1, 2):
            for j in range(0, (max(mesh_patt.pattern) + 1) * 2 + 1):
                cell = (i, j)
                if cell not in cells_before:
                    cells_after.append(cell)
        return MeshPattern(mesh_patt.pattern, cells_after)

    def ascii_plot(self) -> str:
        """Returns an ascii plot of the mesh pattern.
        Example:
        >>> print(MeshPattern(CayleyPermutation([1, 2]), [(0, 3), (2, 2)]).ascii_plot())
           |   |
        xxx+---●---
           |▒▒▒|
        ---●---+---
           |   |

        """
        empty_cell = "   "
        shaded_cell = "\u2592\u2592\u2592"
        point = "\u25cf"
        normal_row = "---"
        shaded_row = "xxx"
        crossing_lines = "+"
        normal_column = "|"

        if len(self.pattern) == 0:
            return "+---+\n|   |\n+---+\n"
        n = len(self.pattern)
        m = max(self.pattern) + 1
        cperm = self.pattern

        # make point rows
        point_rows = []
        for row_height in range(m):
            # create list of points and crossing lines for the row
            row = []
            for val in cperm.cperm:
                if val == row_height:
                    row.append(point)
                else:
                    row.append(crossing_lines)
            # create list of horizontal lines between the points
            if any(
                cell[1] == 2 * row_height + 1 and cell[0] == 0
                for cell in self.shaded_cells
            ):
                point_row_with_shadings = shaded_row
            else:
                point_row_with_shadings = normal_row
            count = 0
            for idx in range(2, 2 * n + 2, 2):
                point_row_with_shadings += row[count]
                count += 1
                if any(
                    cell[1] == 2 * row_height + 1 and cell[0] == idx
                    for cell in self.shaded_cells
                ):
                    point_row_with_shadings += shaded_row
                else:
                    point_row_with_shadings += normal_row
            point_rows.append(point_row_with_shadings)
        # make empty rows
        empty_rows = []
        for row in range(0, 2 * m + 2, 2):
            if any(cell[0] == 0 and cell[1] == row for cell in self.shaded_cells):
                new_row = shaded_cell
            else:
                new_row = empty_cell
            for idx in range(2, n + 4, 2):
                new_row += normal_column
                if any(cell[0] == idx and cell[1] == row for cell in self.shaded_cells):
                    new_row += shaded_cell
                else:
                    new_row += empty_cell
            empty_rows.append(new_row)
        # make grid
        point_rows_reversed = [x for x in reversed(point_rows)]
        empty_rows_reversed = [x for x in reversed(empty_rows)]
        grid = empty_rows_reversed[0] + "\n"
        for idx, point_row in enumerate(point_rows_reversed):
            grid += point_row + "\n"
            grid += empty_rows_reversed[idx + 1] + "\n"
        return grid

    def __len__(self) -> int:
        return len(self.pattern)

    def __leq__(self, other: "MeshPattern") -> bool:
        return self.pattern <= other.pattern

    def __lt__(self, other: "MeshPattern") -> bool:
        return self.pattern < other.pattern

    def __str__(self) -> str:
        return f"MeshPattern({self.pattern}, {self.shaded_cells})"
