"""Removes all the empty rows and columns from a tiling."""

from typing import Dict, Iterator, Optional, Tuple
from comb_spec_searcher import DisjointUnionStrategy
from comb_spec_searcher.exception import StrategyDoesNotApply

from gridded_cayley_permutations import Tiling
from gridded_cayley_permutations import GriddedCayleyPerm

Cell = Tuple[int, int]


class RemoveEmptyRowsAndColumnsStrategy(
    DisjointUnionStrategy[Tiling, GriddedCayleyPerm]
):
    def __init__(
        self,
        ignore_parent: bool = True,
        possibly_empty: bool = False,
    ):
        super().__init__(ignore_parent=ignore_parent, possibly_empty=possibly_empty)

    def decomposition_function(self, comb_class: Tiling) -> Tuple[Tiling, ...]:
        rows_and_cols = comb_class.find_empty_rows_and_columns()
        if len(rows_and_cols[0]) == 0 and len(rows_and_cols[1]) == 0:
            raise StrategyDoesNotApply("No empty rows or columns")
        return (comb_class.remove_empty_rows_and_columns(),)

    def extra_parameters(
        self, comb_class: Tiling, children: Optional[Tuple[Tiling, ...]] = None
    ) -> Tuple[Dict[str, str], ...]:
        return tuple({} for _ in self.decomposition_function(comb_class))

    def formal_step(self):
        return "Removed empty rows and columns"

    def backward_map(
        self,
        comb_class: Tiling,
        objs: Tuple[Optional[GriddedCayleyPerm], ...],
        children: Optional[Tuple[Tiling, ...]] = None,
    ) -> Iterator[GriddedCayleyPerm]:
        if children is None:
            children = self.decomposition_function(comb_class)
        raise NotImplementedError

    def forward_map(
        self,
        comb_class: Tiling,
        obj: GriddedCayleyPerm,
        children: Optional[Tuple[Tiling, ...]] = None,
    ) -> Tuple[Optional[GriddedCayleyPerm], ...]:
        if children is None:
            children = self.decomposition_function(comb_class)
        raise NotImplementedError

    def __str__(self) -> str:
        return self.formal_step()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"ignore_parent={self.ignore_parent}, "
            f"possibly_empty={self.possibly_empty})"
        )

    def to_jsonable(self) -> dict:
        """Return a dictionary form of the strategy."""
        d: dict = super().to_jsonable()
        d.pop("workable")
        d.pop("inferrable")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RemoveEmptyRowsAndColumnsStrategy":
        return cls(**d)
