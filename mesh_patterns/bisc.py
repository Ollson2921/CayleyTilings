from mesh_patterns.mesh_patts import MeshPattern
from cayley_permutations import CayleyPermutation
from collections import defaultdict
from typing import Iterable, Optional
from itertools import chain
from logzero import logger
from tqdm import tqdm
import time


class BiSC:
    def __init__(
        self,
        max_patt_size: int,
        avoiders: Iterable[CayleyPermutation],
        containers: Optional[Iterable[CayleyPermutation]] = None,
    ):
        self.max_patt_size = max_patt_size
        self.avoiders = defaultdict(set)
        self.containers = defaultdict(set)
        for cperm in avoiders:
            self.avoiders[len(cperm)].add(cperm)
        if containers is not None:
            for cperm in containers:
                self.containers[len(cperm)].add(cperm)
        else:
            for i in range(max(self.avoiders.keys()) + 1):
                self.containers[i] = (
                    set(CayleyPermutation.of_size(i)) - self.avoiders[i]
                )

    def all_avoiders(self, start: int = 0) -> Iterable[CayleyPermutation]:
        return frozenset(
            chain.from_iterable(
                (self.avoiders[i] for i in range(start, max(self.avoiders.keys()) + 1))
            )
        )

    def all_containers(self) -> Iterable[CayleyPermutation]:
        return frozenset(
            chain.from_iterable((cperms for cperms in self.containers.values()))
        )

    @staticmethod
    def minimal_patterns(patterns: Iterable[MeshPattern]) -> list[MeshPattern]:
        basis = []
        logger.info("Computing minimal mesh patterns")
        for patt in tqdm(sorted(patterns)):
            if patt.avoids(basis):
                basis.append(patt)
        return basis

    def find_mesh_basis(self):
        """
        Find the minimal patterns that are not contained in the avoiders.

        This can be modeled as a set cover problem on the containers.
        https://math.stackexchange.com/questions/2750531/finding-the-smallest-set-with-non-empty-intercept-with-of-a-collection-of-sets

        First reduce to the minimal set of mesh patterns according to
        mesh pattern containment.

        Then, compute Co(p) for each pattern p in the minimal set,
        in order to find the mesh basis p1, p2, ..., pk such that
        Co(p1) union Co(p2) union ... union Co(pk)
        is equal to the containers.
        """
        minimal_patterns_not_contained = list(
            self.find_minimal_patterns_not_contained()
        )
        basis = self.minimal_patterns(minimal_patterns_not_contained)
        containers = self.all_containers()
        logger.info("Computing container sets for patterns")
        subsets_left = []
        for patt in tqdm(basis):
            patt_containers = set(
                cperm for cperm in containers if patt.is_contained_in_cperm(cperm)
            )
            subsets_left.append((patt, patt_containers))
        logger.info("Searching for set cover")
        subsets_left.sort(key=lambda x: len(x[1]))
        res = []
        while subsets_left:
            patt, patt_containers = subsets_left.pop()
            res.append(patt)
            containers -= patt_containers
            subsets_left = [
                (old_patt, old_patt_containers - patt_containers)
                for old_patt, old_patt_containers in subsets_left
            ]
            subsets_left = [
                (old_patt, old_patt_containers)
                for old_patt, old_patt_containers in subsets_left
                if old_patt_containers
            ]
            subsets_left.sort(key=lambda x: len(x[1]))
        if not containers:
            yield res

    def find_minimal_patterns_not_contained(self):
        """

        https://esham.io/files/2012/09/olympic-colors/pkbuch-chap1.pdf
        page 16 - Algorithm 1.4

        Compute the minimal incomparable set of patterns to those contained in avoiders
        """
        maximal_shaded_patterns = self.find_maximal_shaded_patterns()
        logger.info("Computing minimal patterns avoided")
        for pattern in (
            set(
                chain.from_iterable(
                    CayleyPermutation.of_size(i) for i in range(self.max_patt_size + 1)
                )
            )
            - maximal_shaded_patterns.keys()
        ):
            yield MeshPattern(pattern, [])
        for patt, shadings in tqdm(maximal_shaded_patterns.items()):

            vertices = set()
            edges = set()
            cell_label: dict[tuple[int, int], int] = dict()
            label_cell: dict[int, tuple[int, int]] = dict()
            for shading in shadings:
                edge = []
                for cell in shading:
                    if cell not in cell_label:
                        cell_label[cell] = len(cell_label)
                        label_cell[cell_label[cell]] = cell
                        vertices.add(cell_label[cell])
                    edge.append(cell_label[cell])
                edges.add(tuple(sorted(edge)))
            for shading in BiSC.EnumerateHS(list(vertices), list(edges)):
                cells = [label_cell[v] for v in shading]
                print(MeshPattern(patt, cells))
                yield MeshPattern(patt, cells)

    @staticmethod
    def EnumerateHS(
        vertices: list[int],
        edges: list[tuple[int, ...]],
        non_vertices: Optional[list[int]] = None,
    ) -> set[list[int]]:
        """
        https://esham.io/files/2012/09/olympic-colors/pkbuch-chap1.pdf
        page 16 - Algorithm 1.4, don't care about k here.
        """
        if not edges:
            return [set(non_vertices)]
        if non_vertices is None:
            non_vertices = []
        res = []
        edge = edges[0]
        for v in edge:
            new_vertices = vertices.copy()
            new_vertices.remove(v)
            new_edges = list(e for e in edges[1:] if v not in e)
            new_non_vertices = non_vertices + [v]
            res.extend(BiSC.EnumerateHS(new_vertices, new_edges, new_non_vertices))
        return res

    def find_maximal_shaded_patterns(self):
        """
        For each permutation in the avoiders, find the maximal mesh patterns
        it contains for each pattern size.

        Here maximal means the most shading possible.
        """
        contained_patterns = defaultdict(set)
        logger.info("Computing size minimal patterns")
        for cperm in tqdm(self.all_avoiders()):
            minimal_shadings = self.minimal_shadings(cperm)
            for patt, shadings in minimal_shadings.items():
                for shading in shadings:
                    shading = frozenset(shading)
                    sets_to_remove = set()
                    to_add = True
                    for other_shading in contained_patterns[patt]:
                        if other_shading.issubset(shading):
                            to_add = False
                            break
                        elif shading.issubset(other_shading):
                            sets_to_remove.add(other_shading)
                    if to_add:
                        contained_patterns[patt] -= sets_to_remove
                        contained_patterns[patt].add(frozenset(shading))
        return contained_patterns

    def minimal_shadings(self, cperm: CayleyPermutation) -> dict:
        cperm_as_mesh_patt = MeshPattern(cperm, [])
        min_shadings = dict()
        for mesh_patt in cperm_as_mesh_patt.sub_mesh_patterns(self.max_patt_size):
            if mesh_patt.pattern not in min_shadings:
                min_shadings[mesh_patt.pattern] = set(
                    [
                        frozenset(mesh_patt.shaded_cells),
                    ]
                )
            else:
                min_shadings[mesh_patt.pattern].add(frozenset(mesh_patt.shaded_cells))
        return min_shadings
