""" This module contains the class Av which
generates Cayley permutations avoiding a given basis. """

from typing import List, Dict, Tuple, Set, FrozenSet

from .cayley import CayleyPermutation


class Av:
    """
    Generates Cayley permutations avoiding the input.
    """

    def __init__(self, basis: List[CayleyPermutation]):
        """Cache is a list of dictionaries. The nth dictionary contains the Cayley
        permutations of size n which avoid the basis and a tuple of lists.
        The  first list is the indices where a new maximum can be inserted
        and the second is the indices where the same maximum can be inserted."""
        self.basis = basis
        self.cache: List[Dict[CayleyPermutation, Tuple[List[int], List[int]]]] = [
            {CayleyPermutation([]): ([0], [])}
        ]

    def in_class(self, cperm: CayleyPermutation) -> bool:
        """
        Returns True if the Cayley permutation avoids the basis.

        Examples:
        >>> av = Av([CayleyPermutation([0, 1]), CayleyPermutation([1, 0])])
        >>> av.in_class(CayleyPermutation([0, 0, 0]))
        True
        >>> av = Av([CayleyPermutation([0, 1]), CayleyPermutation([1, 0])])
        >>> av.in_class(CayleyPermutation([0, 1, 0]))
        False
        """
        return not cperm.contains(self.basis)

    def generate_cperms(self, size: int) -> List[CayleyPermutation]:
        """Generate Cayley permutations of size 'size' which
        avoid the basis by checking avoidance at each step.

        Examples:
        >>> Av([CayleyPermutation([0, 1]), CayleyPermutation([1, 0])]).generate_cperms(3)
        [CayleyPermutation([0, 0, 0])]

        >>> Av([CayleyPermutation([0, 0]), CayleyPermutation([1, 0])]).generate_cperms(4)
        [CayleyPermutation([0, 1, 2, 3])]
        """
        if size == 0:
            return [CayleyPermutation([])]
        cperms = [CayleyPermutation([1])]
        count = 1
        next_cperms: List[CayleyPermutation] = []
        while count < size:
            for cperm in cperms:
                for next_cperm in cperm.add_maximum():
                    if self.in_class(next_cperm):
                        next_cperms.append(next_cperm)
            count += 1
            cperms = next_cperms
            next_cperms = []
        return cperms

    def counter(self, ran: int = 7) -> List[int]:
        """
        Returns a list of the number of cperms for each size in range 'ran'
        starting at size 0 (the empty Cayley permutation).

        Examples:
        >>> print(Av([CayleyPermutation([0, 1]), CayleyPermutation([1, 0])]).counter(3))
        [1, 1, 1, 1]

        >>> print(Av([CayleyPermutation([1, 0])]).counter(4))
        [1, 1, 2, 4, 8]
        """
        count = []
        for size in range(ran + 1):
            count.append(len(self.generate_cperms(size)))
        return count

    # def new_generate_cperms(
    #     self, size: int
    # ) -> dict[CayleyPermutation, tuple[list[int], list[int]]]:
    #     """
    #     TODO: not working
    #     Returns a list of Cayley permutations of size 'size' which avoids the basis.

    #     Examples:
    #     >>> Av([CayleyPermutation([1, 0]), CayleyPermutation([0, 1])]).new_generate_cperms(3)
    #     {CayleyPermutation([0, 0, 0]): ([], [])}

    #     >>> Av([CayleyPermutation([0, 0]), CayleyPermutation([1, 0])]).new_generate_cperms(4)
    #     {CayleyPermutation([0, 1, 2, 3]): ([], [])}
    #     """
    #     self.ensure_level(size)
    #     return self.cache[size]

    # def ensure_level(self, size: int):
    #     """Ensures that the cache has a level for size 'size'."""
    #     maximums = {max(cperm.cperm) for cperm in self.basis}
    #     max_basis_value = max(maximums)
    #     for nplusone in range(len(self.cache), size + 1):
    #         n = nplusone - 1
    #         new_level: Dict[CayleyPermutation, Tuple[List[int], List[int]]] = {}
    #         for cperm, valid_ins in self.cache[n].items():
    #             new_max, same_max = valid_ins
    #             if len(cperm.cperm) == 0:
    #                 max_value = -1
    #             else:
    #                 max_value = max(cperm.cperm)
    #             for index in self.new_max_valid_insertions(cperm, max_basis_value):
    #                 new_cperm = cperm.insert(index, max_value + 1)
    #                 if self.condition() or not new_cperm.contains(self.basis):
    #                     new_level[new_cperm] = ([], [])
    #                     new_max.append(index)
    #             for index in self.same_max_valid_insertions(cperm, max_basis_value):
    #                 new_cperm = cperm.insert(index, max_value)
    #                 if self.condition() or not new_cperm.contains(self.basis):
    #                     new_level[new_cperm] = ([], [])
    #                     same_max.append(index)

    #         self.cache.append(new_level)

    # def new_max_valid_insertions(
    #     self, cperm: CayleyPermutation, max_basis_value: int
    # ) -> FrozenSet[int]:
    #     """Returns a list of indices where a new maximum can be inserted into cperm."""
    #     res = None
    #     if len(cperm) == 0:
    #         return frozenset([0])
    #     if max(cperm) < max_basis_value:
    #         return frozenset(range(len(cperm) + 1))
    #     for index in cperm.indices_above_value(max(cperm.cperm) - max_basis_value):
    #         sub_cperm = cperm.delete_index(index)
    #         indices = self.cache[len(sub_cperm)][sub_cperm][0]
    #         valid_indices = [i for i in indices if i <= index]
    #         valid_indices.extend([i + 1 for i in indices if i >= index])
    #         if res is None:
    #             res = frozenset(valid_indices)
    #         else:
    #             res = res.intersection(valid_indices)
    #         if not res:
    #             break
    #     assert res is not None
    #     return res

    # def same_max_valid_insertions(
    #     self, cperm: CayleyPermutation, max_basis_value: int
    # ) -> FrozenSet[int]:
    #     """Returns a list of indices where the same maximum can be inserted into cperm."""
    #     res = None
    #     if len(cperm) == 0:
    #         return frozenset([])
    #     if max(cperm.cperm) < max_basis_value:
    #         last_entry = cperm.index_rightmost_max()
    #         return frozenset(range(last_entry + 1, len(cperm) + 1))
    #     for index in cperm.indices_above_value(max(cperm.cperm) - max_basis_value):
    #         sub_cperm = cperm.delete_index(index)
    #         if len(sub_cperm.cperm) == 0:
    #             max_sub_cperm = 0
    #         else:
    #             max_sub_cperm = max(sub_cperm.cperm)
    #         if max(cperm.cperm) != max_sub_cperm:
    #             indices = self.cache[len(sub_cperm)][sub_cperm][0]
    #         else:
    #             indices = self.cache[len(sub_cperm)][sub_cperm][1]
    #         valid_indices = [i for i in indices if i <= index]
    #         valid_indices.extend([i + 1 for i in indices if i >= index])
    #         if res is None:
    #             res = frozenset(valid_indices)
    #         else:
    #             res = res.intersection(valid_indices)
    #         if not res:
    #             break
    #     assert res is not None
    #     return res

    def condition(self) -> bool:
        """Returns True if can skip pattern avoidance."""
        return False

    def __str__(self) -> str:
        return f"Av({ ','.join(str(x) for x in self.basis)})"


class CanonicalAv(Av):
    """Generates canonical Cayley permutations avoiding the basis."""

    def in_class(self, cperm: CayleyPermutation) -> bool:
        return not cperm.contains(self.basis) and cperm.is_canonical()

    # def generate_cperms(self, size: int) -> List[CayleyPermutation]:
    #     """Generates canonical Cayley permutations of size 'size' avoiding the basis.

    #     Examples:
    #     >>> CanonicalAv([CayleyPermutation([1, 2])]).generate_cperms(3)
    #     [CayleyPermutation([1, 1, 1])]
    #     >>> for cperm in CanonicalAv([CayleyPermutation([1, 2, 3])]).generate_cperms(3):
    #     ...     print(cperm)
    #     111
    #     112
    #     121
    #     122
    #     """
    #     perms = []
    #     config = CanonicalConfiguration(["🔹"])
    #     for perm in config.cayley_perms(size, self.basis):
    #         if len(perm) == size:
    #             perms.append(perm)
    #     return perms

    def get_canonical_basis(self) -> List[CayleyPermutation]:
        """Turns a basis into canonical form using as_canonical() from the CayleyPermutation class.

        Example:
        >>> print(CanonicalAv([CayleyPermutation([1, 0])]).get_canonical_basis())
        [CayleyPermutation([0, 1, 0])]
        """
        basis: Set[CayleyPermutation] = set()
        for cperm in self.basis:
            basis.update(cperm.as_canonical())
        res: List[CayleyPermutation] = []
        for cperm in sorted(basis, key=len):
            if not cperm.contains(res):
                res.append(cperm)
        return res

    def new_max_valid_insertions(
        self, cperm: CayleyPermutation, max_basis_value: int
    ) -> FrozenSet[int]:
        res = None
        if len(cperm) <= max_basis_value:
            acceptable_indices = []
            for idx in range(len(cperm) + 1):
                if self.new_max_okay(cperm, idx):
                    acceptable_indices.append(idx)
            return frozenset(acceptable_indices)
        for index in cperm.indices_above_value(max(cperm.cperm) - max_basis_value):
            sub_cperm = cperm.delete_index(index)
            indices = self.cache[len(sub_cperm)][sub_cperm][0]
            valid_indices = [i for i in indices if i <= index]
            valid_indices.extend([i + 1 for i in indices if i >= index])
            if res is None:
                res = frozenset(valid_indices)
            else:
                res = res.intersection(valid_indices)
            if not res:
                break
        assert res is not None
        return res

    def new_max_okay(self, cperm: CayleyPermutation, index: int) -> bool:
        """Returns True if the new maximum at index is okay for canonical form."""
        if len(cperm) == 0:
            return True
        for idx, val in enumerate(cperm):
            if idx < index:
                if val == max(cperm):
                    return True
        return False
