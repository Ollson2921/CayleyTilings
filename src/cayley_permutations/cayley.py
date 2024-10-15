"""This module contains the CayleyPermutation class and functions for working with them."""

from itertools import combinations
from typing import Iterable, Iterator, Tuple, List


class CayleyPermutation:
    """
    A Cayley Permutation is a list of integers with repeats allowed where
    if n is in the list, every k < n is in the list.

    Examples:
    >>> print(CayleyPermutation([0, 1, 2]))
    012
    >>> print(CayleyPermutation([1, 0, 2, 1]))
    1021
    """

    def __init__(self, cperm: Iterable[int]):
        """
        Checks that the input is a Cayley permutation and converts it to zero based if not already.
        """
        try:
            self.cperm: Tuple[int, ...] = tuple(cperm)
        except TypeError as error:
            raise TypeError(
                "Input to CayleyPermutation must be an iterable of ints."
            ) from error

        if len(self.cperm) != 0:
            for val in range(1, max(self.cperm)):
                if val not in self.cperm:
                    raise ValueError(
                        "Input to CayleyPermutation must be a Cayley permutation."
                    )

            if 0 not in self.cperm:
                self.cperm = tuple(x - 1 for x in self.cperm)

    def __eq__(self, other) -> bool:
        return self.cperm == other.cperm

    def as_one_based(self) -> "List[int]":
        """Returns Cayley permutation as a one based list from zero based.

        Example:
        >>> CayleyPermutation([1, 2, 3, 0])
        CayleyPermutation([1, 2, 3, 0])
        """
        return [x + 1 for x in self.cperm]

    def sub_cperms(self) -> List["CayleyPermutation"]:
        """Returns all sub-Cayley permutations of the Cayley permutation."""
        sub_cperms = set()
        next_cperms = self.remove_one_point(self)
        while next_cperms:
            sub_cperms.update(next_cperms)
            next_cperms = [
                cperm for cperm in next_cperms for cperm in self.remove_one_point(cperm)
            ]
        return sub_cperms

    def remove_one_point(self, cperm: "CayleyPermutation") -> List["CayleyPermutation"]:
        """Returns all sub-Cayley permutations that are the Cayley permutation with one point removed."""
        sub_cperms = set()
        if len(cperm) == 1:
            return sub_cperms
        for i in range(len(cperm)):
            sub_cperms.add(self.standardise((cperm.cperm[:i] + cperm.cperm[i + 1 :])))
        return sub_cperms

    def is_simple(self) -> bool:
        """Returns true if the Cayley permutation is simple."""
        number_of_indices = len(self.cperm)
        for a, b in combinations(range(number_of_indices), 2):
            if len(self.interval(a, b)) < number_of_indices:
                return False
        return True

    def interval(self, idx1: int, idx2: int) -> List[int]:
        """Returns the smallest interval in the Cayley permutation
        that contains the indices idx1 and idx2."""
        indices_in_interval = list(range(idx1, idx2 + 1))
        new_indices_in_interval = self.add_to_interval(indices_in_interval)
        while new_indices_in_interval != list(
            range(min(new_indices_in_interval), max(new_indices_in_interval) + 1)
        ):
            indices_in_interval_no_gaps = list(
                range(min(new_indices_in_interval), max(new_indices_in_interval) + 1)
            )
            new_indices_in_interval = sorted(
                self.add_to_interval(indices_in_interval_no_gaps)
            )
        return new_indices_in_interval

    def add_to_interval(self, indices_in_interval: List[int]) -> List[int]:
        """For any values in the Cayley permutation that are in the range
        of the interval, adds their indices to the list of indices in the interval."""
        subcperm = [self.cperm[idx] for idx in indices_in_interval]
        max_val = max(subcperm)
        min_val = min(subcperm)
        values_in_interval = list(range(min_val, max_val + 1))
        for idx, val in enumerate(self.cperm):
            if not idx in indices_in_interval:
                if val in values_in_interval:
                    indices_in_interval.append(idx)
                    values_in_interval.append(val)
        return indices_in_interval

    def block_decomposition(self) -> List[List[int]]:
        """For a Cayley permutation, breaks it into intervals, begining with
        the leftmost largest interval and returns these as a list.

        Example:
        >>> CayleyPermutation([0, 1, 2, 1, 0]).block_decomposition()
        [[0, 1, 2], [3], [4]]
        """
        blocks = []
        current_index = 0
        while current_index < len(self):
            end_index = len(self.cperm) - 1
            while len(self.interval(current_index, end_index)) == len(self.cperm):
                if current_index == end_index:
                    break
                end_index -= 1
            if current_index == end_index:
                block = [current_index]
            else:
                block = self.interval(current_index, end_index)
            blocks.append(block)
            current_index = max(block) + 1
        return blocks

    def standardisation_of_block(self) -> "CayleyPermutation":
        """Returns the standardisation of the block of the Cayley permutation.
        (is the simple Cayley permutation that was inflated to give the Cayley permutation).
        """
        block_decomposition = self.block_decomposition()
        cperm = []
        for i in range(len(block_decomposition)):
            cperm.append(self.cperm[block_decomposition[i][0]])
        return CayleyPermutation.standardise(cperm)

    @classmethod
    def inflation(
        cls, simple_decomp: Tuple["CayleyPermutation", Tuple["CayleyPermutation", ...]]
    ) -> "CayleyPermutation":
        """Returns the inflation of the Cayley permutation from the simple decomposition.

        Example:
        >>> cperm = CayleyPermutation([0, 1, 2, 1, 0])
        >>> simple_decomp = cperm.simple_decomposition()
        >>> CayleyPermutation.inflation(simple_decomp)
        CayleyPermutation([0, 1, 2, 1, 0])
        """
        simple_cperm, blocks_decomp = simple_decomp
        simple = simple_cperm.cperm
        blocks = [block.cperm for block in blocks_decomp]
        blocks_and_indices = []
        current_max = -1
        for i in range(max(simple) + 1):
            indices = []
            for idx, val in enumerate(simple):
                if val == i:
                    indices.append(idx)
            for idx in indices:
                new_block = [x + current_max + 1 for x in blocks[idx]]
                blocks_and_indices.append((new_block, idx))
            if current_max < max(new_block):
                current_max = max(new_block)
        cperm = []
        for i in range(len(blocks_and_indices)):
            cperm.extend(sorted(blocks_and_indices, key=lambda x: x[1])[i][0])
        return CayleyPermutation(cperm)

    def simple_decomposition(
        self,
    ) -> Tuple["CayleyPermutation", Tuple["CayleyPermutation", ...]]:
        """For a Cayley permutation, returns the tuple of the simple Cayley permutation it
        was inflated from and it's block decomposition.

        Example:
        >>> cperm = CayleyPermutation([0, 1, 2, 1, 0])
        >>> cperm.simple_decomposition()
        (CayleyPermutation([0, 1, 2]), (CayleyPermutation([0, 0]), CayleyPermutation([0])))
        """
        blocks = self.block_decomposition()
        simple_cperm = self.standardisation_of_block()
        cperm_blocks = []
        for block in blocks:
            cperm_blocks.append(
                CayleyPermutation.standardise(self.cperm[block[0] : block[-1] + 1])
            )
        return simple_cperm, tuple(cperm_blocks)

    def sum_decomposable(self) -> bool:
        """Returns true if the Cayley permutation is sum decomposable."""
        for idx in range(len(self.cperm) - 1):
            interval = self.interval(0, idx)
            if len(interval) == len(self.cperm):
                return False
            for i in interval:
                if self.cperm[i] == 0:
                    return True
        return False

    def skew_decomposable(self) -> bool:
        """Returns true if the Cayley permutation is skew decomposable."""
        for idx in range(len(self.cperm) - 1, 0, -1):
            interval = self.interval(idx, len(self.cperm) - 1)
            if len(interval) == len(self.cperm):
                return False
            for i in interval:
                if self.cperm[i] == 0:
                    return True
        return False

    @classmethod
    def of_size(cls, size: int) -> List["CayleyPermutation"]:
        """
        Returns a list of all Cayley permutations of size 'size'.

        Examples:
        >>> CayleyPermutation.of_size(0)
        [CayleyPermutation([])]
        >>> CayleyPermutation.of_size(1)
        [CayleyPermutation([0])]
        >>> CayleyPermutation.of_size(2)
        [CayleyPermutation([1, 0]), CayleyPermutation([0, 1]), CayleyPermutation([0, 0])]
        """
        cperms: List["CayleyPermutation"] = []
        if size == 0:
            return [CayleyPermutation([])]
        if size == 1:
            return [CayleyPermutation([0])]
        for cperm in CayleyPermutation.of_size(size - 1):
            cperms.extend(cperm.add_maximum())
        return cperms

    def insert(self, index, value):
        """Inserts value at index in the Cayley permutation."""
        return CayleyPermutation(self.cperm[:index] + [value] + self.cperm[index:])

    def subperm_from_indices(self, indices: List[int]) -> "CayleyPermutation":
        """Returns the Cayley permutation at the indices."""
        return CayleyPermutation.standardise([self.cperm[idx] for idx in indices])

    def indices_above_value(self, value: int) -> List[int]:
        """Returns a list of the indices of the values that
        are greater than or equal to the input value."""
        above_max_indices = []
        for idx, val in enumerate(self.cperm):
            if val >= value:
                above_max_indices.append(idx)
        return above_max_indices

    def add_maximum(self) -> List["CayleyPermutation"]:
        """Adds a new maximum to the Cayley permutation in every possible way
        (one larger anywhere or the same as the current max at a smaller index).

        Example:
        >>> for cperm in CayleyPermutation([0, 1]).add_maximum():
        ...     print(cperm)
        201
        021
        012
        101
        011
        """
        val = max(self.cperm)
        index = self.cperm.index(val)
        perms = []
        for i in range(len(self.cperm) + 1):
            perms.append(
                CayleyPermutation(
                    [x for x in self.cperm[:i]]
                    + [val + 1]
                    + [x for x in self.cperm[i:]]
                )
            )
        for i in range(index + 1):
            perms.append(
                CayleyPermutation(
                    [x for x in self.cperm[:i]] + [val] + [x for x in self.cperm[i:]]
                )
            )
        return perms

    def contains(self, patterns: Iterable["CayleyPermutation"]) -> bool:
        """
        Input a list of patterns and returns true if contains any of them.

        Examples:
        >>> CayleyPermutation([0, 1, 2]).contains([CayleyPermutation([0, 1])])
        True
        >>> CayleyPermutation([0, 1, 2]).contains([CayleyPermutation([0, 1]),
        ... CayleyPermutation([1, 0])])
        True
        >>> CayleyPermutation([0, 1, 2]).contains([CayleyPermutation([1, 0])])
        False
        """
        return any(self.contains_pattern(pattern) for pattern in patterns)

    def contains_pattern(self, pattern: "CayleyPermutation") -> bool:
        """
        Input one pattern and returns true if the pattern is contained.

        Examples:
        >>> CayleyPermutation([0, 1, 2]).contains_pattern(CayleyPermutation([0, 1]))
        True
        >>> CayleyPermutation([0, 1, 2]).contains_pattern(CayleyPermutation([1, 0]))
        False
        """
        size = len(self)
        for indices in combinations(range(size), len(pattern)):
            occ = [self.cperm[idx] for idx in indices]
            stand = self.standardise(occ)
            if stand == pattern:
                return True
        return False

    def avoids(self, pattern: Iterable["CayleyPermutation"]) -> bool:
        """Returns true if the Cayley permutation avoids any of the patterns."""
        return not self.contains(pattern)

    def avoids_pattern(self, pattern: "CayleyPermutation") -> bool:
        """Returns true if the Cayley permutation avoids the pattern."""
        return not self.contains_pattern(pattern)

    @classmethod
    def standardise(cls, pattern: Iterable[int]) -> "CayleyPermutation":
        """Returns the standardised version of a pattern.

        Example:
        >>> CayleyPermutation.standardise([2, 3])
        CayleyPermutation([0, 1])
        """
        pattern = tuple(pattern)
        key = sorted(set(pattern))
        stand = {}
        for i, v in enumerate(key):
            stand[v] = i
        return CayleyPermutation([stand[pat] for pat in pattern])

    def reverse(self) -> "CayleyPermutation":
        """Returns the reverse of the Cayley permutation."""
        return CayleyPermutation(self.cperm[::-1])

    def first_k_entries(self, k: int) -> List[int]:
        """Returns a list of the indices of the first k numbers
        that were inserted in the Cayley permutation.

        Example:
        >>> CayleyPermutation([2, 0, 1, 2]).first_k_entries(2)
        [1, 2]
        >>> CayleyPermutation([0, 1, 0, 1, 2]).first_k_entries(3)
        [0, 2, 3]
        """
        current_min = 0
        indices: List[int] = []
        while len(indices) < k:
            mindices = []
            for idx, val in enumerate(self.cperm):
                if val == current_min:
                    mindices.append(idx)
            indices.extend(mindices[-(k - len(indices)) :])
            current_min += 1
        return sorted(indices)

    def last_k_entries(self, k: int) -> List[int]:
        """Returns a list of the indices of the last k numbers that were inserted.

        Example:
        >>> CayleyPermutation([2, 0, 1, 2]).last_k_entries(2)
        [0, 3]
        """
        current_max = max(self.cperm)
        indices: List[int] = []
        while len(indices) < k:
            maxindices = []
            for idx, val in enumerate(self.cperm):
                if val == current_max:
                    maxindices.append(idx)
            indices.extend(maxindices[: k - len(indices)])
            current_max -= 1
        return sorted(indices)

    def index_rightmost_max(self) -> int:
        """Returns the index of the rightmost maximum."""
        if len(self) == 0:
            # TODO: WHY?
            return 1
        max_val = max(self)
        for idx, val in reversed(list(enumerate(self))):
            if val == max_val:
                return idx
        raise ValueError("No maximum found.")

    def occurrences(
        self, basis: List["CayleyPermutation"]
    ) -> dict["CayleyPermutation", List[Tuple[int, ...]]]:
        """Returns a dictionary of the occurrences of a pattern in the basis
        and indices of the Cayley permutation where they occur.

        Example:
        >>> basis = [CayleyPermutation([0, 0])]
        >>> CayleyPermutation([0, 1, 2, 1, 2]).occurrences(basis)
        {CayleyPermutation([0, 0]): [(1, 3), (2, 4)]}
        """
        size = len(self)
        dict_of_occ_and_indices: dict["CayleyPermutation", List[Tuple[int, ...]]] = {}
        for pattern in basis:
            dict_of_occ_and_indices[pattern] = []
            for indices in combinations(range(size), len(pattern)):
                occ = [self.cperm[idx] for idx in indices]
                stand = self.standardise(occ)
                if stand == pattern:
                    dict_of_occ_and_indices[pattern].append(indices)
        return dict_of_occ_and_indices

    def avoids_same_after_deleting(
        self, basis: Iterable["CayleyPermutation"], index: int
    ) -> bool:
        """
        Returns true if the Cayley permutation avoids
        the basis still after deleting the index.
        """
        basis = tuple(basis)
        if self.contains(basis):
            cperm_deleted = self.delete_index(index)
            if not cperm_deleted.contains(basis):
                return False
        return True

    def delete_index(self, index: int) -> "CayleyPermutation":
        """Returns a Cayley permutation with the index deleted."""
        return CayleyPermutation.standardise(
            self.cperm[:index] + self.cperm[index + 1 :]
        )

    def is_monotonically_decreasing(self) -> bool:
        """Returns true if the Cayley permutation is monotonicaly decreasing.

        Example:
        >>> CayleyPermutation([2, 1, 0, 0]).is_monotonically_decreasing()
        True
        """
        first_elements = self.cperm[:-1]
        second_elements = self.cperm[1:]
        for first, second in zip(first_elements, second_elements):
            if first < second:
                return False
        return True

    def is_monotonically_increasing(self) -> bool:
        """Returns true if the Cayley permutation is monotonicaly increasing.

        Example:
        >>> CayleyPermutation([0, 1, 2, 2]).is_monotonically_increasing()
        True
        """
        first_elements = self.cperm[:-1]
        second_elements = self.cperm[1:]
        for first, second in zip(first_elements, second_elements):
            if first > second:
                return False
        return True

    def is_increasing(self) -> bool:
        """Returns true if the Cayley permutation is strictly increasing.

        Example:
        >>> CayleyPermutation([0, 1, 2, 2]).is_increasing()
        False
        >>> CayleyPermutation([0, 1, 2]).is_increasing()
        True
        """
        first_elements = self.cperm[:-1]
        second_elements = self.cperm[1:]
        for first, second in zip(first_elements, second_elements):
            if first >= second:
                return False
        return True

    def is_decreasing(self) -> bool:
        """Returns true if the Cayley permutation is strictly decreasing.

        Example:
        >>> CayleyPermutation([2, 1, 0, 0]).is_decreasing()
        False
        >>> CayleyPermutation([2, 1, 0]).is_decreasing()
        True
        """
        first_elements = self.cperm[:-1]
        second_elements = self.cperm[1:]
        for first, second in zip(first_elements, second_elements):
            if first <= second:
                return False
        return True

    def is_constant(self) -> bool:
        """Returns true if the Cayley permutation is constant.

        Example:
        >>> CayleyPermutation([0, 0, 1, 0]).is_constant()
        False
        """
        first_elements = self.cperm[:-1]
        second_elements = self.cperm[1:]
        for first, second in zip(first_elements, second_elements):
            if first != second:
                return False
        return True

    def check_is_strict(self) -> bool:
        """Returns true if the Cayley permutation is strictly increasing, strictly decreasing, or constant."""
        if self.is_increasing():
            return True
        if self.is_decreasing():
            return True
        if self.is_constant():
            return True
        return False

    def is_canonical(self) -> bool:
        """Returns true if the Cayley permutation is canonical.
        To be in canonical form, any number in the Cayley permutation must have
        every number smaller than it at smaller indices.

        Example:
        >>> CayleyPermutation([0, 1, 2, 1, 0]).is_canonical()
        True
        >>> CayleyPermutation([1, 2, 1, 0]).is_canonical()
        False
        """
        if len(self) == 0:
            return True
        max_val = max(self.cperm)
        for i in range(max_val + 1):
            for val in self.cperm:
                if val > i:
                    return False
                if val == i:
                    break
        return True

    def as_canonical(self) -> Iterator["CayleyPermutation"]:
        """Converts a Cayley permutation into a list
        of Cayley permutations that are in canonical form.
        A state is a 4-tuple of a Cayley perm and indices

        Examples:
        >>> list(CayleyPermutation([2, 1, 0]).as_canonical())
        [CayleyPermutation([0, 1, 2, 1, 0])]
        >>> list(CayleyPermutation([0, 1, 0, 3, 2]).as_canonical())
        [CayleyPermutation([0, 1, 0, 2, 3, 2]), CayleyPermutation([0, 1, 2, 0, 3, 2])]
        """
        idx_current_max, val_current_max, working_index = -1, -1, 0
        states = [(self.cperm, idx_current_max, val_current_max, working_index)]
        while states:
            new_states: List[tuple[List[int], int, int, int]] = []
            for state in states:
                if len(state[0]) == state[3]:
                    yield CayleyPermutation(state[0])
                else:
                    new_states.extend(self._fix_first_max(state))
            states = new_states

    def _fix_first_max(
        self, state: Tuple[List[int], int, int, int]
    ) -> Iterator[Tuple[List[int], int, int, int]]:
        """Checks values in a Cayley permutation to see if they are in canonical form
        and if not then inserts the numbers needed in every possible way"""
        cperm, idx_current_max, val_current_max, working_index = state
        if cperm[working_index] <= val_current_max:
            working_index += 1
            new_state = (cperm, idx_current_max, val_current_max, working_index)
            yield new_state
        elif cperm[working_index] == val_current_max + 1:
            val_current_max += 1
            idx_current_max = working_index
            working_index += 1
            new_state = (cperm, idx_current_max, val_current_max, working_index)
            yield new_state
        elif cperm[working_index] > val_current_max + 1:
            list_a = cperm[idx_current_max + 1 : working_index]
            list_b = list(range(val_current_max + 1, cperm[working_index]))
            new_val_current_max = cperm[working_index]
            new_idx_current_max = working_index + len(list_b)
            new_working_index = working_index + 1 + len(list_b)
            for shuff in self.shuffle(list_a, list_b):
                new_cperm = (
                    cperm[: idx_current_max + 1] + tuple(shuff) + cperm[working_index:]
                )
                new_state = (
                    new_cperm,
                    new_idx_current_max,
                    new_val_current_max,
                    new_working_index,
                )
                yield new_state

    @staticmethod
    def shuffle(list_a: List[int], list_b: List[int]) -> Iterator[List[int]]:
        """Returns all possible shuffles of two lists list_a and list_b.

        Example:
        >>> for shuff in CayleyPermutation.shuffle([1, 2], [3, 4]):
        ...     print(shuff)
        [1, 2, 3, 4]
        [1, 3, 2, 4]
        [1, 3, 4, 2]
        [3, 1, 2, 4]
        [3, 1, 4, 2]
        [3, 4, 1, 2]
        """
        length_a = len(list_a)
        length_b = len(list_b)
        for a_indices in combinations(range(length_a + length_b), length_a):
            b_indices = [i for i in range(length_a + length_b) if i not in a_indices]
            shuff = list(range(length_a + length_b))
            for idx_a, idx_shuff in enumerate(a_indices):
                shuff[idx_shuff] = list_a[idx_a]
            for idx_b, idx_shuff in enumerate(b_indices):
                shuff[idx_shuff] = list_b[idx_b]
            yield list(shuff)

    def ascii_plot(self) -> str:
        """Returns an ascii plot of the Cayley permutation.
        Example:
        >>> print(CayleyPermutation([0, 1, 2, 1, 0]).ascii_plot())
           |   |   |   |   |
        ---+---+---●---+---+---
           |   |   |   |   |
        ---+---●---+---●---+---
           |   |   |   |   |
        ---●---+---+---+---●---
           |   |   |   |   |
        """
        if len(self) == 0:
            return "+---+\n|   |\n+---+\n"
        n = len(self.cperm)
        m = max(self.cperm)
        empty_cell = "   "
        point = "\u25cf"
        normal_row = "---"
        crossing_lines = "+"
        normal_column = "|"
        point_rows = []
        for i in range(m + 1):
            new_row = normal_row
            for j in self.cperm:
                if j == i:
                    new_row += point + normal_row
                else:
                    new_row += crossing_lines + normal_row
            new_row += "\n"
            point_rows.append(new_row)
        empty_row = normal_column.join(empty_cell for _ in range(n + 1)) + "\n"
        grid = empty_row + empty_row.join(reversed(point_rows)) + empty_row
        return grid

    def to_jsonable(self) -> dict:
        """Returns a dictionary of the Cayley permutation."""
        return {"cperm": self.cperm}

    @classmethod
    def from_dict(cls, d: dict) -> "CayleyPermutation":
        """Returns a Cayley permutation from a dictionary."""
        return cls(d["cperm"])

    def __len__(self):
        return len(self.cperm)

    def __iter__(self):
        return iter(self.cperm)

    def __hash__(self):
        return hash(tuple(self.cperm))

    def __str__(self):
        return "".join(str(x) if x < 10 else f"({str(x)})" for x in self.cperm)

    def __repr__(self):
        return f"CayleyPermutation({self.cperm})"

    def __lt__(self, other: "CayleyPermutation") -> bool:
        return (len(self.cperm), self.cperm) < (len(other.cperm), other.cperm)

    def __le__(self, other: "CayleyPermutation") -> bool:
        return (len(self.cperm), self.cperm) <= (len(other.cperm), other.cperm)

    def __getitem__(self, key: int) -> int:
        return self.cperm[key]
