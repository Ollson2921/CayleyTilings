from comb_spec_searcher import StrategyPack, AtomStrategy
from .strategies import (
    RemoveEmptyRowsAndColumnsStrategy,
    FactorStrategy,
    ShuffleFactorStrategy,
    InsertionEncodingRequirementInsertionFactory,
    InsertionEncodingPlacementFactory,
    CellInsertionFactory,
    PointPlacementFactory,
    LessThanRowColSeparationStrategy,
    LessThanOrEqualRowColSeparationStrategy,
)


class TileScopePack(StrategyPack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def insertion_encoding(cls):
        return TileScopePack(
            initial_strats=[
                FactorStrategy(),
                InsertionEncodingRequirementInsertionFactory(),
            ],  # Iterable[Strategy]
            inferral_strats=[RemoveEmptyRowsAndColumnsStrategy()],  # Iterable[Strategy]
            expansion_strats=[
                [InsertionEncodingPlacementFactory()]
            ],  # Iterable[Iterable[Strategy]]
            ver_strats=[AtomStrategy()],  # Iterable[Strategy]
            name="Insertion Encoding",
            symmetries=[],
            iterative=False,
        )

    @classmethod
    def point_placement(cls):
        return TileScopePack(
            initial_strats=[
                FactorStrategy(),
                LessThanOrEqualRowColSeparationStrategy(),
            ],  # Iterable[Strategy]
            inferral_strats=[
                RemoveEmptyRowsAndColumnsStrategy(),
                LessThanRowColSeparationStrategy(),
            ],  # Iterable[Strategy]
            expansion_strats=[
                [CellInsertionFactory(), PointPlacementFactory()]
            ],  # Iterable[Iterable[Strategy]]
            ver_strats=[AtomStrategy()],  # Iterable[Strategy]
            name="Point Placement",
            symmetries=[],
            iterative=False,
        )

    @classmethod
    def point_placements_shuffle(cls):
        return TileScopePack(
            initial_strats=[
                ShuffleFactorStrategy(),
                LessThanOrEqualRowColSeparationStrategy(),
            ],  # Iterable[Strategy]
            inferral_strats=[
                RemoveEmptyRowsAndColumnsStrategy(),
                LessThanRowColSeparationStrategy(),
            ],  # Iterable[Strategy]
            expansion_strats=[
                [CellInsertionFactory(), PointPlacementFactory()]
            ],  # Iterable[Iterable[Strategy]]
            ver_strats=[AtomStrategy()],  # Iterable[Strategy]
            name="Point Placements Shuffle",
            symmetries=[],
            iterative=False,
        )
