from comb_spec_searcher import StrategyFactory

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


class FusionFactory(StrategyFactory[Tiling]):
    def __call__(self, comb_class: Tiling):
        pass

    @classmethod
    def from_dict(cls, d: dict) -> "FusionFactory":
        return cls(**d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return "Fusion factory"
