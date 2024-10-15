from typing import Dict, Iterable, List, Tuple, Iterator, TYPE_CHECKING
from itertools import product, chain
from gridded_cayley_permutations import GriddedCayleyPerm

if TYPE_CHECKING:
    from .tilings import Tiling

OBSTRUCTIONS = Tuple[GriddedCayleyPerm, ...]
REQUIREMENTS = Tuple[Tuple[GriddedCayleyPerm, ...], ...]


class RowColMap:
    """
    The pre-image of any value is an interval.
    If a > b then every pre-image of a is to the greater than every pre-image of b.
    """

    def __init__(self, col_map: Dict[int, int], row_map: Dict[int, int]):
        self.row_map = row_map
        self.col_map = col_map

    def map_gridded_cperm(self, gcp: GriddedCayleyPerm) -> GriddedCayleyPerm:
        """
        Map a gridded Cayley permutation according to the row and column maps.
        """
        new_positions = []
        for cell in gcp.positions:
            new_cell = (self.col_map[cell[0]], self.row_map[cell[1]])
            new_positions.append((new_cell))
        return GriddedCayleyPerm(gcp.pattern, new_positions)

    def map_gridded_cperms(
        self, gcps: Iterable[GriddedCayleyPerm]
    ) -> Iterable[GriddedCayleyPerm]:
        """
        Map a gridded Cayley permutation according to the column and row maps.
        """
        return list(self.map_gridded_cperm(gcp) for gcp in gcps)

    def map_requirements(
        self, requirements: Iterable[Iterable[GriddedCayleyPerm]]
    ) -> Iterable[Iterable[GriddedCayleyPerm]]:
        """
        Map a list of requirements according to the column and row maps.
        """
        return list(self.map_gridded_cperms(req) for req in requirements)

    def preimage_of_gridded_cperm(
        self, gcp: GriddedCayleyPerm
    ) -> Iterable[GriddedCayleyPerm]:
        """
        Return the preimages of a gridded Cayley permutation with repect to the map.
        """
        for cols, rows in product(self.product_of_cols(gcp), self.product_of_rows(gcp)):
            new_positions = tuple(zip(cols, rows))
            yield GriddedCayleyPerm(gcp.pattern, new_positions)

    def product_of_rows(self, gcp: GriddedCayleyPerm) -> Iterator[Tuple[int, ...]]:
        row_pos = [cell[1] for cell in gcp.positions]
        preimages_of_gcp = (
            self.preimages_of_row_of_gcp(row, gcp) for row in self.row_codomain()
        )
        codomain = self.row_codomain()
        yield from self.product_of_row_or_columns(row_pos, preimages_of_gcp, codomain)

    def product_of_cols(self, gcp: GriddedCayleyPerm) -> Iterator[Tuple[int, ...]]:
        col_pos = [cell[0] for cell in gcp.positions]
        preimages_of_gcp = (
            self.preimages_of_col_of_gcp(col, gcp) for col in self.col_codomain()
        )
        codomain = self.col_codomain()
        yield from self.product_of_row_or_columns(col_pos, preimages_of_gcp, codomain)

    def product_of_row_or_columns(self, positions, preimages_of_gcp, codomain):
        indices = {}
        for row in codomain:
            indices[row] = [idx for idx, val in enumerate(positions) if val == row]
        working_list = [None] * len(positions)
        for row_values_at_indices in product(*preimages_of_gcp):
            for row, values_at_row in zip(codomain, row_values_at_indices):
                for idx, val in zip(indices[row], values_at_row):
                    working_list[idx] = val
            yield tuple(working_list)

    def row_codomain(self):
        return sorted(set(self.row_map.values()))

    def col_codomain(self):
        return sorted(set(self.col_map.values()))

    def partition(self, n, k):
        """Partition n into k parts"""
        if k == 1:
            yield [n]
            return
        for i in range(n + 1):
            for result in self.partition(n - i, k - 1):
                yield [i] + result

    def preimages_of_row_of_gcp(
        self, row: int, gcp: GriddedCayleyPerm
    ) -> Iterable[int]:
        """Finds all the preimages of the subcayley permutation of gcp in the row.
        Yields tuples of preimages of the values in the row.
        """
        values_in_row = gcp.values_in_row(row)
        pre_image_values = self.preimages_of_row(row)
        yield from self._preimages_of_gcp(values_in_row, pre_image_values)

    def preimages_of_row(self, row: int) -> List[int]:
        """Return the preimages of all values in the row."""
        keys = []
        for key, value in self.row_map.items():
            if value == row:
                keys.append(key)
        return keys

    def preimages_of_col_of_gcp(
        self, col: int, gcp: GriddedCayleyPerm
    ) -> Iterable[int]:
        """Return the preimages of the subcayley permutation of gcp in the column."""
        indices_in_col = gcp.indices_in_col(col)
        pre_image_values = self.preimages_of_col(col)
        yield from self._preimages_of_gcp(indices_in_col, pre_image_values)

    def _preimages_of_gcp(self, values_in_col, pre_image_values):
        if not values_in_col:
            yield tuple()
            return
        number_of_values = max(values_in_col) - min(values_in_col) + 1
        size = len(pre_image_values)
        values_ordered = sorted(set(values_in_col))
        for partition in self.partition(number_of_values, size):
            preimage = [None] * len(values_in_col)
            seen_so_far = 0
            for idx, part in enumerate(partition):
                new_col = pre_image_values[idx]
                vals = values_ordered[seen_so_far : seen_so_far + part]
                seen_so_far += part
                for idx, val in enumerate(values_in_col):
                    if val in vals:
                        preimage[idx] = new_col
            yield tuple(preimage)

    def preimages_of_col(self, col: int) -> List[int]:
        """Return the preimages of all values in the column."""
        keys = []
        for key, value in self.col_map.items():
            if value == col:
                keys.append(key)
        return keys

    def preimage_of_obstructions(
        self, obstructions: Iterable[GriddedCayleyPerm]
    ) -> Iterable[GriddedCayleyPerm]:
        """Return the preimages of the obstructions."""
        return list(
            chain.from_iterable(
                self.preimage_of_gridded_cperm(ob) for ob in obstructions
            )
        )

    def preimage_of_requirements(
        self, requirements: Iterable[Iterable[GriddedCayleyPerm]]
    ) -> Iterable[Iterable[GriddedCayleyPerm]]:
        """Return the preimages of the requirements."""
        return list(self.preimage_of_obstructions(req) for req in requirements)

    def preimage_of_tiling(self, tiling: "Tiling") -> Tuple[OBSTRUCTIONS, REQUIREMENTS]:
        """Return the preimage of the tiling."""
        return self.preimage_of_obstructions(
            tiling.obstructions
        ), self.preimage_of_requirements(tiling.requirements)

    def __str__(self) -> str:
        return f"RowColMap({self.col_map}, {self.row_map})"
