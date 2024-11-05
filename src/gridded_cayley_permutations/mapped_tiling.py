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


class Parameter:
    def __init__(self, tiling, row_col_map):
        '''we may need to keep track of which direction the row_col_map goes'''
        self.param = tiling
        self.map = row_col_map

    def __repr__(self):
        return str((repr(self.param), str(self.map)))


class MappedTiling:

    def __init__(self, tiling : Tiling, parameters : Iterable[Parameter]):
        self.tiling = tiling
        self.parameters = parameters

    def pop_parameter(self,parameter_index = 0): 
        '''removes the parameter at an index and creates a new mapped tiling'''
        param = self.parameters.pop(parameter_index)
        return MappedTiling(self.tiling, [param])
    
    def pop_all_parameters(self):
        '''yields all mapped tilings with a single parameter'''
        while len(self.parameters) > 0:
            yield self.pop_parameter()

    def add_parameter(self,parameter):
        self.parameters.append(parameter)

    def add_obstructions(self,obstructions):
        '''adds obstructions to the tiling (and corrects the parameters)'''
        '''Here we assume the direction of row_col_map'''
        new_parameters = []
        for parameter in self.parameters:
            new_parameter = parameter.param.add_obstructions(parameter.map.preimage_of_obstructions(obstructions))
            new_parameters.append(Parameter(new_parameter,parameter.map))
        return MappedTiling(self.tiling.add_obstructions(obstructions),new_parameters)
    
    def add_requirements(self,requirements):
        '''adds requirements to the tiling (and corrects the parameters)'''
        '''Here we assume the direction of row_col_map'''
        new_parameters = []
        for parameter in self.parameters:
            new_parameter = parameter.param.add_requirements(parameter.map.preimage_of_requirements(requirements))
            new_parameters.append(Parameter(new_parameter,parameter.map))
        return MappedTiling(self.tiling.add_requirements(requirements),new_parameters)
    



    
