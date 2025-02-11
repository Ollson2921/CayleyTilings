"""Places a point requirement in a mappling in extreme directions.
0 = rightmost 
1 = topmost, taking the rightmost if there are multiple
2 = topmost, taking the leftmost if there are multiple
3 = leftmost
4 = bottommost, taking the leftmost if there are multiple
5 = bottommost, taking the rightmost if there are multiple"""

from typing import Dict, Iterable, Iterator, Optional, Tuple
from comb_spec_searcher import DisjointUnionStrategy, StrategyFactory
from gridded_cayley_permutations.point_placements import (
    Directions,
    Right_bot,
    Left,
    Right,
    Left_bot,
    Left_top,
    Right_top,
)
from gridded_cayley_permutations import GriddedCayleyPerm
from cayley_permutations import CayleyPermutation
from mapped_tiling import MappedTiling
from MT_point_placement import MTRequirementPlacement, MTPartialPointPlacements

Cell = Tuple[int, int]


class MTRequirementPlacementStrategy(
    DisjointUnionStrategy[MappedTiling, GriddedCayleyPerm]
):
    """Places the requirements in gcps at the indices in the direction given in the mappling."""

    DIRECTIONS = Directions

    def __init__(
        self,
        gcps: Iterable[GriddedCayleyPerm],
        indices: Iterable[int],
        direction: int,
        ignore_parent: bool = False,
    ):
        self.gcps = tuple(gcps)
        self.indices = tuple(indices)
        self.direction = direction
        assert direction in self.DIRECTIONS
        super().__init__(ignore_parent=ignore_parent)

    def algorithm(self, mappling: MappedTiling) -> MTRequirementPlacement:
        return MTRequirementPlacement(mappling)

    def decomposition_function(
        self, comb_class: MappedTiling
    ) -> Tuple[MappedTiling, ...]:
        """Either the cells doesn't contain gcp so add it as obstruction
        or it contains an occurrence of it furthest in the direction given
        so add it as a requirement in every possible way."""
        return (comb_class.add_obstructions(self.gcps),) + self.algorithm(
            comb_class
        ).point_placement(self.gcps, self.indices, self.direction)

    def extra_parameters(
        self,
        comb_class: MappedTiling,
        children: Optional[Tuple[MappedTiling, ...]] = None,
    ) -> Tuple[Dict[str, str], ...]:
        return tuple({} for _ in self.decomposition_function(comb_class))

    def formal_step(self):
        return f"Placed the point of the requirement {self.gcps} at indices {self.indices} in direction {self.direction}"

    def backward_map(
        self,
        comb_class: MappedTiling,
        objs: Tuple[Optional[GriddedCayleyPerm], ...],
        children: Optional[Tuple[MappedTiling, ...]] = None,
    ) -> Iterator[GriddedCayleyPerm]:
        if children is None:
            children = self.decomposition_function(comb_class)
        raise NotImplementedError

    def forward_map(
        self,
        comb_class: MappedTiling,
        obj: GriddedCayleyPerm,
        children: Optional[Tuple[MappedTiling, ...]] = None,
    ) -> Tuple[Optional[GriddedCayleyPerm], ...]:
        if children is None:
            children = self.decomposition_function(comb_class)
        raise NotImplementedError

    def __str__(self) -> str:
        return self.formal_step()

    def __repr__(self) -> str:
        return (
            f"MTRequirementPlacementStrategy(gcps={self.gcps}, "
            f"indices={self.indices}, direction={self.direction}, "
            f"ignore_parent={self.ignore_parent})"
        )

    def to_jsonable(self) -> dict:
        """Return a dictionary form of the strategy."""
        d: dict = super().to_jsonable()
        d.pop("workable")
        d.pop("inferrable")
        d.pop("possibly_empty")
        d["gcps"] = tuple(gp.to_jsonable() for gp in self.gcps)
        d["indices"] = self.indices
        d["direction"] = self.direction
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MTRequirementPlacementStrategy":
        gcps = tuple(GriddedCayleyPerm.from_dict(gcp) for gcp in d.pop("gcps"))
        return cls(gcps=gcps, **d)


class MTPointPlacementFactory(StrategyFactory[MappedTiling]):
    def __call__(
        self, comb_class: MappedTiling
    ) -> Iterator[MTRequirementPlacementStrategy]:
        """Factory to place a point requirement in a mappling in extreme directions
        in each positive cell of the base tiling."""
        for cell in comb_class.tiling.positive_cells():
            for direction in Directions:
                gcps = (GriddedCayleyPerm(CayleyPermutation([0]), [cell]),)
                indices = (0,)
                yield MTRequirementPlacementStrategy(gcps, indices, direction)
                if direction in MTPartialRequirementPlacementStrategy.DIRECTIONS:
                    yield MTPartialRequirementPlacementStrategy(
                        gcps, indices, direction
                    )

    @classmethod
    def from_dict(cls, d: dict) -> "MTPointPlacementFactory":
        return cls(**d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return "Point placement"


class MTPartialRequirementPlacementStrategy(MTRequirementPlacementStrategy):
    DIRECTIONS = [Left, Right]

    def algorithm(self, mappling: MappedTiling) -> MTPartialPointPlacements:
        return MTPartialPointPlacements(mappling)  # TODO: Implement this

    def formal_step(self):
        return f"Partially placed the point of the requirement {self.gcps} at indices {self.indices} in direction {self.direction}"


class RowInsertionFactory(StrategyFactory[MappedTiling]):
    def __call__(
        self, comb_class: MappedTiling
    ) -> Iterator[MTRequirementPlacementStrategy]:
        not_point_rows = (
            set(range(comb_class.tiling.dimensions[1])) - comb_class.tiling.point_rows()
        )
        for row in not_point_rows:
            all_gcps = []
            for col in range(comb_class.tiling.dimensions[0]):
                cell = (col, row)
                gcps = GriddedCayleyPerm(CayleyPermutation([0]), [cell])
                all_gcps.append(gcps)
            indices = tuple(0 for _ in all_gcps)
            for direction in [Left_bot, Right_bot, Left_top, Right_top]:
                yield MTRequirementPlacementStrategy(all_gcps, indices, direction)

    @classmethod
    def from_dict(cls, d: dict) -> "RowInsertionFactory":
        return cls(**d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return "Row insertion"


class ColInsertionFactory(StrategyFactory[MappedTiling]):
    def __call__(
        self, comb_class: MappedTiling
    ) -> Iterator[MTRequirementPlacementStrategy]:
        not_point_cols = set(range(comb_class.tiling.dimensions[0])) - set(
            [cell[0] for cell in comb_class.tiling.point_cells()]
        )
        for col in not_point_cols:
            all_gcps = []
            for row in range(comb_class.tiling.dimensions[1]):
                cell = (col, row)
                gcps = GriddedCayleyPerm(CayleyPermutation([0]), [cell])
                all_gcps.append(gcps)
            indices = tuple(0 for _ in all_gcps)
            for direction in [Left, Right]:
                yield MTRequirementPlacementStrategy(all_gcps, indices, direction)

    @classmethod
    def from_dict(cls, d: dict) -> "RowInsertionFactory":
        return cls(**d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return "Column insertion"
