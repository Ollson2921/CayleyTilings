from typing import Iterable, Iterator, Tuple
from collections import defaultdict
from copy import copy
from itertools import product
from math import factorial
from comb_spec_searcher import CombinatorialClass

from cayley_permutations import CayleyPermutation
from .row_col_map import RowColMap
from .gridded_cayley_perms import GriddedCayleyPerm
from .simplify_obstructions_and_requirements import SimplifyObstructionsAndRequirements
from .minimal_gridded_cperms import MinimalGriddedCayleyPerm
from .tilings import Tiling
from .point_placements import PointPlacement


class Parameter:
    def __init__(self, tiling, row_col_map):
        """we may need to keep track of which direction the row_col_map goes"""
        self.param = tiling
        self.map = row_col_map

    def new_row_col_map(self, cell):
        """returns a new row_col_map for point placement"""
        new_row_map = dict()
        for item in self.map.row_map.items():
            if item[0] < cell[0]:
                new_row_map[item[0]] = item[1]
            elif item[0] == cell[0]:
                new_row_map[item[0]] = cell[0]
                new_row_map[item[0] + 1] = cell[0] + 1
                new_row_map[item[0] + 2] = cell[0] + 2
            else:
                new_row_map[item[0] + 2] = item[1] + 2
        new_col_map = dict()
        for item in self.map.col_map.items():
            if item[0] < cell[0]:
                new_col_map[item[0]] = item[1]
            elif item[0] == cell[0]:
                new_col_map[item[0]] = cell[0]
                new_col_map[item[0] + 1] = cell[0] + 1
                new_col_map[item[0] + 2] = cell[0] + 2
            else:
                new_col_map[item[0] + 2] = item[1] + 2
        return RowColMap(new_col_map, new_row_map)

    def __repr__(self):
        return str((repr(self.param), str(self.map)))

    def __str__(self) -> str:
        return str(self.param) + "\n" + str(self.map)


class MappedTiling:

    def __init__(self, tiling: Tiling, parameters: Iterable[Parameter]):
        self.tiling = tiling
        self.parameters = parameters

    def pop_parameter(self, parameter_index=0):
        """removes the parameter at an index and creates a new mapped tiling"""
        param = self.parameters.pop(parameter_index)
        return MappedTiling(self.tiling, [param])

    def pop_all_parameters(self):
        """yields all mapped tilings with a single parameter"""
        while len(self.parameters) > 0:
            yield self.pop_parameter()

    def add_parameter(self, parameter):
        self.parameters.append(parameter)

    def add_obstructions(self, obstructions):
        """adds obstructions to the tiling (and corrects the parameters)"""
        """Here we assume the direction of row_col_map"""
        new_parameters = []
        for parameter in self.parameters:
            new_parameter = parameter.param.add_obstructions(
                parameter.map.preimage_of_obstructions(obstructions)
            )
            new_parameters.append(Parameter(new_parameter, parameter.map))
        return MappedTiling(self.tiling.add_obstructions(obstructions), new_parameters)

    def add_requirements(self, requirements):
        """adds requirements to the tiling (and corrects the parameters)"""
        """Here we assume the direction of row_col_map"""
        new_parameters = []
        for parameter in self.parameters:
            new_parameter = parameter.param.add_requirements(
                parameter.map.preimage_of_requirements(requirements)
            )
            new_parameters.append(Parameter(new_parameter, parameter.map))
        return MappedTiling(self.tiling.add_requirements(requirements), new_parameters)

    def point_placement(self, cell, direction) -> "MappedTiling":
        """returns the point placement of a cell in a direction"""
        """Here we assume the direction of row_col_map"""
        point = [GriddedCayleyPerm(CayleyPermutation([0]), [cell])]
        indices = (0,)
        new_tiling = PointPlacement(self.tiling).point_placement(
            point, indices, direction
        )[0]
        new_parameters = []
        for parameter in self.parameters:
            for preimage_cell in parameter.map.preimage_of_cell(cell):
                point = [GriddedCayleyPerm(CayleyPermutation([0]), [preimage_cell])]
                new_param = PointPlacement(parameter.param).point_placement(
                    point, indices, direction
                )[0]
                new_col_map = parameter.new_row_col_map(cell)
                new_parameters.append(Parameter(new_param, new_col_map))
        return MappedTiling(new_tiling, new_parameters)

    def __repr__(self):
        return str((repr(self.tiling), [repr(p) for p in self.parameters]))

    def __str__(self) -> str:
        return (
            str(self.tiling)
            + "\n"
            + "Parameters:"
            + "\n"
            + "\n".join([str(p) for p in self.parameters])
        )
