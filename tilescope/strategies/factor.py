"""Factors the tiling into sections that are independent of each other."""

from typing import Dict, Iterator, Optional, Tuple
from comb_spec_searcher import CartesianProductStrategy, Strategy
from comb_spec_searcher.exception import StrategyDoesNotApply
from comb_spec_searcher.strategies.constructor import Constructor
from gridded_cayley_permutations import Tiling, GriddedCayleyPerm
from gridded_cayley_permutations.factors import Factors, ShuffleFactors

# from .dummy_constructor import DummyConstructor


class AbstractFactorStrategy:
    def __init__(
        self,
        ignore_parent: bool = True,
        workable: bool = True,
    ):
        # TODO: input should include partition: Iterable[Iterable[Cell]] to
        #       allow for interleaving factors.
        super().__init__(
            ignore_parent=ignore_parent, workable=workable, inferrable=True
        )

    def decomposition_function(self, comb_class: Tiling) -> Tuple[Tiling, ...]:
        # TODO: raise StrategyDoesNotApply if no factors.
        return Factors(comb_class).find_factors()

    def extra_parameters(
        self, comb_class: Tiling, children: Optional[Tuple[Tiling, ...]] = None
    ) -> Tuple[Dict[str, str], ...]:
        if children is None:
            children = self.decomposition_function(comb_class)
            if children is None:
                raise StrategyDoesNotApply("Strategy does not apply")
        return tuple({} for _ in children)

    def formal_step(self) -> str:
        """
        Return a string that describe the operation performed on the tiling.
        """
        return "Factor the tiling into factors."

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
    ) -> Tuple[GriddedCayleyPerm, ...]:
        if children is None:
            children = self.decomposition_function(comb_class)
        raise NotImplementedError

    def __str__(self) -> str:
        return self.formal_step()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"ignore_parent={self.ignore_parent}, "
            f"workable={self.workable})"
        )

    # JSON methods

    def to_jsonable(self) -> dict:
        """Return a dictionary form of the strategy."""
        d: dict = super().to_jsonable()
        d.pop("inferrable")
        d.pop("possibly_empty")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AbstractFactorStrategy":
        return cls(**d)


class FactorStrategy(
    AbstractFactorStrategy, CartesianProductStrategy[Tiling, GriddedCayleyPerm]
):
    pass


class ShuffleFactorStrategy(
    AbstractFactorStrategy, Strategy[Tiling, GriddedCayleyPerm]
):
    def decomposition_function(self, comb_class: Tiling) -> Tuple[Tiling, ...]:
        if 1 not in comb_class.dimensions:
            raise StrategyDoesNotApply(
                "Tiling is not a row or column shuffle of factors."
            )
        # TODO: raise StrategyDoesNotApply if no factors.
        return ShuffleFactors(comb_class).find_factors()

    def constructor(
        self, comb_class: Tiling, children: Tuple[Tiling, ...] | None = None
    ) -> Constructor:
        return DummyConstructor()

    def can_be_equivalent(self) -> bool:
        return True

    def is_reversible(self, comb_class: Tiling) -> bool:
        return False

    def is_two_way(self, comb_class: Tiling) -> bool:
        return False

    def reverse_constructor(
        self, idx: int, comb_class: Tiling, children: Tuple[Tiling, ...] | None = None
    ) -> Constructor:
        raise NotImplementedError

    def shifts(
        self, comb_class: Tiling, children: Tuple[Tiling, ...] | None
    ) -> Tuple[int, ...]:
        min_sizes = tuple(child.minimum_size_of_object() for child in children)
        point_sum = sum(min_sizes)
        return tuple(point_sum - min_size for min_size in min_sizes)
