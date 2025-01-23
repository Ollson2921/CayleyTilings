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
from comb_spec_searcher import Strategy
from typing import Tuple, Optional, Dict, Set, Iterator
from comb_spec_searcher.exception import StrategyDoesNotApply
from comb_spec_searcher.strategies.constructor import Constructor


class FusionStrategy(Strategy[Tiling, GriddedCayleyPerm]):
    def __init__(self, direction: int, index: int, tracked: bool = False):
        self.direction = direction
        self.index = index
        self.tracked = tracked
        if direction not in [1, 0]:
            raise ValueError("Direction must be 1 or 0")
        if index < 0:
            raise ValueError("Index must be non-negative")
        super().__init__(
            ignore_parent=False, inferrable=True, possibly_empty=False, workable=True
        )

    # def __call__(
    #     self,
    #     comb_class: Tiling,
    #     children: Optional[Tuple[Tiling, ...]] = None,
    # ):
    #     if children is None:
    #         children = self.decomposition_function(comb_class)
    #         # if children is None:
    #         #     raise StrategyDoesNotApply("Strategy does not apply")
    #     # return FusionRule(self, comb_class, children=children)
    #     return children

    def decomposition_function(self, tiling: Tiling) -> Tiling:
        # if tiling.is_fuseable(self.direction, self.index):
        return (tiling.fuse(self.direction, self.index),)
        # raise AttributeError("Trying to fuse a tiling that does not fuse")

    def can_be_equivalent(self) -> bool:
        return False

    def is_two_way(self, comb_class: Tiling):
        return False

    def is_reversible(self, comb_class: Tiling) -> bool:
        # algo = self.fusion_algorithm(comb_class)
        # new_ass = algo.new_assumption()
        # fused_assumptions = (
        #     ass.__class__(gps)
        #     for ass, gps in zip(comb_class.assumptions, algo.assumptions_fuse_counters)
        # )
        # return new_ass in fused_assumptions
        """TODO: We told this to return true to make it work but for tracked tilings and counting will need to change"""
        return True
        raise NotImplementedError

    def shifts(
        self, comb_class: Tiling, children: Optional[Tuple[Tiling, ...]] = None
    ) -> Tuple[int, ...]:
        return (0,)
        raise NotImplementedError

    def constructor(
        self, comb_class: Tiling, children: Optional[Tuple[Tiling, ...]] = None
    ):
        if not self.tracked:
            # constructor only enumerates when tracked.
            raise NotImplementedError("The fusion strategy was not tracked.")
        # Need to recompute some info to count, so ignoring passed in children
        if not comb_class.is_fuseable(self.direction, self.index):
            raise StrategyDoesNotApply("Strategy does not apply")
        child = comb_class.fuse(self.direction, self.index)
        assert children is None or children == (child,)
        # min_left, min_right = algo.min_left_right_points()
        # return FusionConstructor(
        #     comb_class,
        #     child,
        #     self._fuse_parameter(comb_class),
        #     self.extra_parameters(comb_class, children)[0],
        #     *self.left_right_both_sided_parameters(comb_class),
        #     min_left,
        #     min_right,
        # )
        raise NotImplementedError

    def reverse_constructor(  # pylint: disable=no-self-use
        self,
        idx: int,
        comb_class: Tiling,
        children: Optional[Tuple[Tiling, ...]] = None,
    ) -> Constructor:
        # if not self.tracked:
        #     # constructor only enumerates when tracked.
        #     raise NotImplementedError("The fusion strategy was not tracked.")
        # if children is None:
        #     children = self.decomposition_function(comb_class)
        # # Need to recompute some info to count, so ignoring passed in children
        # algo = self.fusion_algorithm(comb_class)
        # if not algo.fusable():
        #     raise StrategyDoesNotApply("Strategy does not apply")
        # child = algo.fused_tiling()
        # assert children is None or children == (child,)
        # (
        #     left_sided_params,
        #     right_sided_params,
        #     _,
        # ) = self.left_right_both_sided_parameters(comb_class)
        # if not left_sided_params and not right_sided_params:
        #     if algo.min_left_right_points() != (0, 0):
        #         raise NotImplementedError(
        #             "Reverse positive fusion counting not implemented"
        #         )
        #     fused_assumption = algo.new_assumption()
        #     unfused_assumption = fused_assumption.__class__(
        #         chain.from_iterable(
        #             algo.unfuse_gridded_perm(gp) for gp in fused_assumption.gps
        #         )
        #     )
        #     assert unfused_assumption in comb_class.assumptions
        #     return DivideByK(
        #         comb_class,
        #         children,
        #         1,
        #         comb_class.get_assumption_parameter(unfused_assumption),
        #         self.extra_parameters(comb_class, children),
        #     )
        # left_points, right_points = algo.min_left_right_points()
        # return ReverseFusionConstructor(
        #     comb_class,
        #     child,
        #     self._fuse_parameter(comb_class),
        #     self.extra_parameters(comb_class, children)[0],
        #     tuple(left_sided_params),
        #     tuple(right_sided_params),
        #     left_points,
        #     right_points,
        # )
        raise NotImplementedError

    def extra_parameters(
        self, comb_class: Tiling, children: Optional[Tuple[Tiling, ...]] = None
    ) -> Tuple[Dict[str, str]]:
        # if children is None:
        #     children = self.decomposition_function(comb_class)
        #     if children is None:
        #         raise StrategyDoesNotApply("Strategy does not apply")
        # algo = self.fusion_algorithm(comb_class)
        # child = children[0]
        # mapped_assumptions = [
        #     child.forward_map.map_assumption(ass.__class__(gps))
        #     for ass, gps in zip(comb_class.assumptions, algo.assumptions_fuse_counters)
        # ]
        # return (
        #     {
        #         k: child.get_assumption_parameter(ass)
        #         for k, ass in zip(comb_class.extra_parameters, mapped_assumptions)
        #         if ass.gps
        #     },
        # )
        raise NotImplementedError

    def left_right_both_sided_parameters(
        self, comb_class: Tiling
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        # left_sided_params: Set[str] = set()
        # right_sided_params: Set[str] = set()
        # both_sided_params: Set[str] = set()
        # algo = self.fusion_algorithm(comb_class)
        # for assumption in comb_class.assumptions:
        #     parent_var = comb_class.get_assumption_parameter(assumption)
        #     left_sided = algo.is_left_sided_assumption(assumption)
        #     right_sided = algo.is_right_sided_assumption(assumption)
        #     if left_sided and not right_sided:
        #         left_sided_params.add(parent_var)
        #     elif right_sided and not left_sided:
        #         right_sided_params.add(parent_var)
        #     elif not left_sided and not right_sided:
        #         both_sided_params.add(parent_var)
        # return (
        #     left_sided_params,
        #     right_sided_params,
        #     both_sided_params,
        # )
        raise NotImplementedError

    def _fuse_parameter(self, comb_class: Tiling) -> str:
        # algo = self.fusion_algorithm(comb_class)
        # child = algo.fused_tiling()
        # ass = algo.new_assumption()
        # fuse_assumption = ass.__class__(child.forward_map.map_gp(gp) for gp in ass.gps)
        # return child.get_assumption_parameter(fuse_assumption)
        raise NotImplementedError

    def formal_step(self) -> str:
        fusing = "rows" if self.direction is 1 else "columns"
        idx = self.index
        return f"Fuse {fusing} {idx} and {idx+1}"

    # pylint: disable=arguments-differ
    def backward_map(
        self,
        comb_class: Tiling,
        objs: Tuple[Optional[GriddedCayleyPerm], ...],
        children: Optional[Tuple[Tiling, ...]] = None,
        left_points: Optional[int] = None,
    ) -> Iterator[GriddedCayleyPerm]:
        """
        The backward direction of the underlying bijection used for object
        generation and sampling.
        """
        # if children is None:
        #     children = self.decomposition_function(comb_class)
        # gp = objs[0]
        # assert gp is not None
        # gp = children[0].backward_map.map_gp(gp)
        # yield from self.fusion_algorithm(comb_class).unfuse_gridded_perm(
        #     gp, left_points
        # )
        raise NotImplementedError

    def forward_map(
        self,
        comb_class: Tiling,
        obj: GriddedCayleyPerm,
        children: Optional[Tuple[Tiling, ...]] = None,
    ) -> Tuple[Optional[GriddedCayleyPerm], ...]:
        """
        The forward direction of the underlying bijection used for object
        generation and sampling.
        """
        # if children is None:
        #     children = self.decomposition_function(comb_class)
        # fused_gp = self.fusion_algorithm(comb_class).fuse_gridded_perm(obj)
        # return (children[0].forward_map.map_gp(fused_gp),)
        raise NotImplementedError

    def to_jsonable(self) -> dict:
        d = super().to_jsonable()
        d.pop("ignore_parent")
        d.pop("inferrable")
        d.pop("possibly_empty")
        d.pop("workable")
        d["direction"] = self.direction
        d["index"] = self.index
        d["tracked"] = self.tracked
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FusionStrategy":
        return cls(**d)

    @staticmethod
    def get_op_symbol() -> str:
        return "⚮"

    @staticmethod
    def get_eq_symbol() -> str:
        return "↣"

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(direction={self.direction}, index={self.index}, "
            f"tracked={self.tracked})"
        )


class FusionFactory(StrategyFactory[Tiling]):
    def __call__(self, comb_class: Tiling):
        # print("Trying fusion")
        for direction in [1, 0]:
            for index in range(comb_class.dimensions[direction] - 1):
                if comb_class.is_fuseable(direction, index):
                    yield FusionStrategy(direction, index)

    @classmethod
    def from_dict(cls, d: dict) -> "FusionFactory":
        return cls(**d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return "Fusion factory"
