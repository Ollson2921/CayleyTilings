from typing import Iterable, List, Tuple, Set
from gridded_cayley_permutations import GriddedCayleyPerm
from itertools import product
from math import factorial
from collections import defaultdict
from cayley_permutations import CayleyPermutation


def binomial(x, y):
    try:
        return factorial(x) // factorial(y) // factorial(x - y)
    except ValueError:
        return 0


class SimplifyObstructionsAndRequirements:
    def __init__(
        self,
        obstructions: Iterable[GriddedCayleyPerm],
        requirements: Iterable[Iterable[GriddedCayleyPerm]],
        dimensions: Tuple[int, int],
    ):
        self.obstructions = obstructions
        self.requirements = requirements
        self.dimensions = dimensions
        self.sort_obstructions()

    def remove_redundant_gridded_cperms(
        self, gridded_cperms: Iterable[GriddedCayleyPerm]
    ):
        """Remove gcps that are implied by other gcps."""
        redundant_gcps = set()
        new_gridded_cperms = list(gridded_cperms)
        for gcp in gridded_cperms:
            for gcp2 in gridded_cperms:
                if gcp != gcp2 and gcp2.contains_gridded_cperm(gcp):
                    redundant_gcps.add(gcp2)
        for gcps in redundant_gcps:
            new_gridded_cperms.remove(gcps)
        return tuple(new_gridded_cperms)

    def remove_redundant_obstructions(self):
        """Remove obstructions that are implied by other obstructions."""
        self.obstructions = self.remove_redundant_gridded_cperms(self.obstructions)

    def remove_redundant_requirements(self):
        """Remove requirements that are implied by other requirements in the same list."""
        self.requirements = tuple(
            self.remove_redundant_gridded_cperms(
                tuple(req for req in req_list if req.avoids(self.obstructions))
            )
            for req_list in self.requirements
        )

    def remove_redundant_lists_requirements(self):
        """Remove requirements lists that are implied by other requirements lists."""
        indices = []
        for i in range(len(self.requirements)):
            for j in range(len(self.requirements)):
                if i != j and j not in indices:
                    req_list_1 = self.requirements[i]
                    req_list_2 = self.requirements[j]
                    if any(req.contains(req_list_2) for req in req_list_1):
                        indices.append(i)
        self.requirements = tuple(
            req for i, req in enumerate(self.requirements) if i not in indices
        )

    def simplify(self):
        """Simplify the obstructions and requirements."""
        curr_obs = None
        curr_reqs = None
        while curr_obs != self.obstructions or curr_reqs != self.requirements:
            curr_obs = self.obstructions
            curr_reqs = self.requirements
            self.simplify_once()
            self.sort_requirements()
            self.sort_obstructions()

    def simplify_once(self):
        self.remove_redundant_obstructions()
        self.remove_redundant_requirements()
        self.remove_redundant_lists_requirements()
        self.remove_factors_from_obstructions()

    def sort_requirements(self):
        """Orders the requirements and removes duplicates."""
        self.requirements = tuple(
            sorted(set(tuple(sorted(set(req_list))) for req_list in self.requirements))
        )

    def sort_obstructions(self):
        """Orders the obstructions and removes duplicates."""
        self.obstructions = tuple(sorted(set(self.obstructions)))

    def remove_factors_from_obstructions(self):
        """Removes factors from all of the obstructions."""
        self.obstructions = tuple(
            self.remove_factors_from_obstruction(ob) for ob in self.obstructions
        )

    def remove_factors_from_obstruction(
        self, ob: GriddedCayleyPerm
    ) -> GriddedCayleyPerm:
        """Removes factors from a single obstruction:
        Splits an obstruction into its factors and removes the factors that are implied by the requirements.
        """
        cells = ob.find_active_cells()
        for factor in ob.find_factors(self.point_rows()):
            if self.implied_by_requirements(factor):
                cells.difference_update(factor.find_active_cells())
        return ob.sub_gridded_cayley_perm(cells)

    def point_rows(self) -> Set[int]:
        """Returns the point rows of the tiling."""
        point_rows = set()
        counter_dict = defaultdict(int)
        for ob in self.obstructions:
            if ob.pattern in (CayleyPermutation([0, 1]), CayleyPermutation([1, 0])):
                if ob.positions[0][1] == ob.positions[1][1]:
                    counter_dict[ob.positions[0][1]] += 1
        for row, count in counter_dict.items():
            n = len(self.cells_in_row(row))
            if 2 * binomial(n, 2) + 2 * n == count:
                point_rows.add(row)
        return point_rows

    def cells_in_row(self, row: int) -> Set[Tuple[int, int]]:
        """Returns the set of active cells in the given row."""
        cells = set()
        for cell in self.active_cells():
            if cell[1] == row:
                cells.add(cell)
        return cells

    def active_cells(self) -> Set[Tuple[int, int]]:
        """Returns the set of active cells in the tiling.
        (Cells are active if they do not contain a point obstruction.)"""
        active_cells = set(
            product(range(self.dimensions[0]), range(self.dimensions[1]))
        )
        for ob in self.obstructions:
            if len(ob) == 1:
                active_cells.discard(ob.positions[0])
        return active_cells

    def implied_by_requirement(
        self, gcp: GriddedCayleyPerm, req_list: List[GriddedCayleyPerm]
    ) -> bool:
        """Check whether a gridded Cayley permutation is implied by a requirement."""
        return all(req.contains_gridded_cperm(gcp) for req in req_list)

    def implied_by_requirements(self, gcp: GriddedCayleyPerm) -> bool:
        """Check whether a gridded Cayley permutation is implied by the requirements."""
        return any(
            self.implied_by_requirement(gcp, req_list) for req_list in self.requirements
        )
