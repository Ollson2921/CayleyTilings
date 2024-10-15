"""Places a point requirement in a tiling in extreme directions.
0 = rightmost 
1 = topmost, taking the rightmost if there are multiple
2 = topmost, taking the leftmost if there are multiple
3 = leftmost
4 = bottommost, taking the leftmost if there are multiple
5 = bottommost, taking the rightmost if there are multiple"""

from typing import Dict, Iterable, Iterator, Optional, Tuple
from comb_spec_searcher import DisjointUnionStrategy, StrategyFactory

from gridded_cayley_permutations import Tiling
from gridded_cayley_permutations.point_placements import (
    PointPlacement,
    PartialPointPlacements,
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


Cell = Tuple[int, int]


class RequirementPlacementStrategy(DisjointUnionStrategy[Tiling, GriddedCayleyPerm]):
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

    def algorithm(self, tiling: Tiling) -> PointPlacement:
        return PointPlacement(tiling)

    def decomposition_function(self, comb_class: Tiling) -> Tuple[Tiling, ...]:
        return (comb_class.add_obstructions(self.gcps),) + self.algorithm(
            comb_class
        ).point_placement(self.gcps, self.indices, self.direction)

    def extra_parameters(
        self, comb_class: Tiling, children: Optional[Tuple[Tiling, ...]] = None
    ) -> Tuple[Dict[str, str], ...]:
        return tuple({} for _ in self.decomposition_function(comb_class))

    def formal_step(self):
        return f"Placed the point of the requirement {self.gcps} at indices {self.indices} in direction {self.direction}"

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
            f"RequirementPlacementStrategy(gcps={self.gcps}, "
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
    def from_dict(cls, d: dict) -> "RequirementPlacementStrategy":
        gcps = tuple(GriddedCayleyPerm.from_dict(gcp) for gcp in d.pop("gcps"))
        return cls(gcps=gcps, **d)


class InsertionEncodingPlacementFactory(StrategyFactory[Tiling]):
    def __call__(self, comb_class: Tiling) -> Iterator[RequirementPlacementStrategy]:
        cells = comb_class.active_cells() - comb_class.point_cells()
        gcps = tuple(
            GriddedCayleyPerm(CayleyPermutation([0]), [cell]) for cell in cells
        )
        indices = tuple(0 for _ in gcps)
        direction = Right_bot
        yield RequirementPlacementStrategy(gcps, indices, direction)

    @classmethod
    def from_dict(cls, d: dict) -> "InsertionEncodingPlacementFactory":
        return cls(**d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return "Place next point of insertion encoding"


class PointPlacementFactory(StrategyFactory[Tiling]):
    def __call__(self, comb_class: Tiling) -> Iterator[RequirementPlacementStrategy]:
        for cell in comb_class.positive_cells():
            for direction in Directions:
                gcps = (GriddedCayleyPerm(CayleyPermutation([0]), [cell]),)
                indices = (0,)
                yield RequirementPlacementStrategy(gcps, indices, direction)
                if direction in PartialRequirementPlacementStrategy.DIRECTIONS:
                    yield PartialRequirementPlacementStrategy(gcps, indices, direction)

    @classmethod
    def from_dict(cls, d: dict) -> "PointPlacementFactory":
        return cls(**d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return "Point placement"


class PartialRequirementPlacementStrategy(RequirementPlacementStrategy):
    DIRECTIONS = [Left, Right]

    def algorithm(self, tiling: Tiling) -> PointPlacement:
        return PartialPointPlacements(tiling)

    def formal_step(self):
        return f"Partially placed the point of the requirement {self.gcps} at indices {self.indices} in direction {self.direction}"


class RowInsertionFactory(StrategyFactory[Tiling]):
    def __call__(self, comb_class: Tiling) -> Iterator[RequirementPlacementStrategy]:
        not_point_rows = set(range(comb_class.dimensions[1])) - comb_class.point_rows()
        for row in not_point_rows:
            all_gcps = []
            for col in range(comb_class.dimensions[0]):
                cell = (col, row)
                gcps = GriddedCayleyPerm(CayleyPermutation([0]), [cell])
                all_gcps.append(gcps)
            indices = tuple(0 for _ in all_gcps)
            for direction in [Left_bot, Right_bot, Left_top, Right_top]:
                yield RequirementPlacementStrategy(all_gcps, indices, direction)

    @classmethod
    def from_dict(cls, d: dict) -> "RowInsertionFactory":
        return cls(**d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return "Row insertion"


class ColInsertionFactory(StrategyFactory[Tiling]):
    def __call__(self, comb_class: Tiling) -> Iterator[RequirementPlacementStrategy]:
        not_point_cols = set(range(comb_class.dimensions[0])) - set(
            [cell[0] for cell in comb_class.point_cells]
        )
        for col in not_point_cols:
            all_gcps = []
            for row in range(comb_class.dimensions[1]):
                cell = (col, row)
                gcps = (GriddedCayleyPerm(CayleyPermutation([0]), [cell]),)
                all_gcps.append(gcps)
            indices = tuple(0 for _ in all_gcps)
            for direction in [Left, Right]:
                yield RequirementPlacementStrategy(all_gcps, indices, direction)

    @classmethod
    def from_dict(cls, d: dict) -> "RowInsertionFactory":
        return cls(**d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return "Row insertion"