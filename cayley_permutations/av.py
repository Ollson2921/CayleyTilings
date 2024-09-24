""" This module contains the class Av which
generates Cayley permutations avoiding a given basis. """

from typing import List, Dict, Tuple, Set

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

    def condition(self) -> bool:
        """Returns True if can skip pattern avoidance."""
        return False

    def __str__(self) -> str:
        return f"Av({ ','.join(str(x) for x in self.basis)})"


class CanonicalAv(Av):
    """Generates canonical Cayley permutations avoiding the basis."""

    def in_class(self, cperm: CayleyPermutation) -> bool:
        return not cperm.contains(self.basis) and cperm.is_canonical()

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
