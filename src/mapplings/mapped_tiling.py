from typing import Iterable, Iterator, Tuple, List, Dict, DefaultDict
from itertools import chain
from collections import defaultdict
from comb_spec_searcher import CombinatorialClass


from cayley_permutations import CayleyPermutation
from gridded_cayley_permutations import (
    RowColMap,
    GriddedCayleyPerm,
    Tiling,
)

Objects = DefaultDict[Tuple[int, ...], List[GriddedCayleyPerm]]


class Parameter:
    def __init__(self, tiling : Tiling, row_col_map : RowColMap ):
        """we may need to keep track of which direction the row_col_map goes"""
        self.ghost = tiling
        self.map = row_col_map

    def is_contradictory(self, tiling: Tiling) -> bool:
        """Returns True if the parameter is contradictory.
        Is contradictory if any of the requirements in the ghost map to a gcp
        containing an obstruction in the tiling
        """
        for req_list in self.ghost.requirements:
            if all(
                self.map.map_gridded_cperm(gcp).contains(tiling.obstructions)
                for gcp in req_list
            ):
                return True
        return False

    def preimage_of_gcp(self, gcp: GriddedCayleyPerm) -> Iterator[GriddedCayleyPerm]:
        """Returns the preimage of a gridded cayley permutation"""
        for gcp in self.map.preimage_of_gridded_cperm(gcp):
            if self.ghost.gcp_in_tiling(gcp):
                yield gcp

    def expand_row_col_map_at_index(
        self, number_of_cols, number_of_rows, col_index, row_index
    ):
        """Adds number_of_cols new columns to the at col_index and
        Adds number_of_rows new rows to the map at row_index
            Assumes we've modified the parameter and the tiling in the same way"""
        new_col_map, new_row_map = dict(), dict()
        """This bit moves the existing mappings"""
        for item in self.map.col_map.items():
            adjust = int(item[0] >= col_index) * number_of_cols
            new_col_map[item[0] + adjust] = item[1] + adjust
        for item in self.map.row_map.items():
            adjust = int(item[0] >= row_index) * number_of_rows
            new_row_map[item[0] + adjust] = item[1] + adjust
        """This bit adds the new dictionary items"""
        original_col, original_row = (
            self.map.col_map[col_index],
            self.map.row_map[row_index],
        )
        for i in range(number_of_cols):
            new_col_map[col_index + i] = original_col + i
        for i in range(number_of_rows):
            new_row_map[row_index + i] = original_row + i
        return RowColMap(new_col_map, new_row_map)

    def reduce_row_col_map(self, col_preimages, row_preimages):
        """This function removes rows and collumns from the map and standardizes the output"""
        new_col_map, new_row_map = self.map.col_map.copy(), self.map.row_map.copy()
        for index in col_preimages:
            del new_col_map[index]
        for index in row_preimages:
            del new_row_map[index]
        return RowColMap(new_col_map, new_row_map).standardise_map()
    
    def back_map_obs_and_reqs(self, tiling : Tiling):
        '''Places all obs and reqs of tiling into the parameter according to the row/col map. 
        Returns a new parameter, but maybe we should just add obs and reqs to existing parameters, IDK
        Doing this for req lists is weird...
        '''
        new_obs, new_reqs = list(self.ghost.obstructions), list(self.ghost.requirements)
        for obstruction in tiling.obstructions:
            new_obs+= list(self.map.preimage_of_gridded_cperm(obstruction))
        for reqs in tiling.requirements:
            new_req_list=[]
            for req in reqs:
                new_req_list+= list(self.map.preimage_of_gridded_cperm(req))
            new_reqs.append(new_req_list)
        return Parameter(Tiling(new_obs,new_reqs,self.ghost.dimensions), self.map)

    def sub_parameter(self, factor):
        preimage_of_cells = self.map.preimage_of_cells(factor)
        return Parameter(self.ghost.sub_tiling(preimage_of_cells), self.map)

    def __repr__(self):
        return str((repr(self.ghost), str(self.map)))

    def __str__(self) -> str:
        return str(self.ghost) + "\n" + str(self.map)


class MappedTiling(CombinatorialClass):

    def __init__(
        self,
        tiling: Tiling,
        avoiding_parameters: Iterable[Parameter],
        containing_parameters: Iterable[Iterable[Parameter]],
        enumeration_parameters: Iterable[Iterable[Parameter]],
    ):
        self.tiling = tiling
        self.avoiding_parameters = avoiding_parameters
        self.containing_parameters = containing_parameters
        self.enumeration_parameters = enumeration_parameters

    def objects_of_size(self, n, **parameters):  # Good
        for val in self.get_objects(n).values():
            for gcp in val:
                yield gcp

    def get_objects(self, n: int) -> Objects:  # Good
        objects = defaultdict(list)
        for gcp in self.tiling.objects_of_size(n):
            if self.gcp_in_tiling(gcp):
                param = self.get_parameters(gcp)
                objects[param].append(gcp)
        return objects

    def get_parameters(self, gcp: GriddedCayleyPerm) -> Tuple[int, ...]:  # Good
        """Parameters are not what you think!!! This is specific to combinatorical class parameters"""
        all_lists = []
        for param_list in self.enumeration_parameters:
            all_lists.append(
                sum(1 for _ in param.preimage_of_gcp(gcp)) for param in param_list
            )
        return tuple(all_lists)

    def gcp_in_tiling(self, gcp: GriddedCayleyPerm) -> bool:  # Good
        """Returns True if the gridded cayley permutation is in the tiling"""
        return self.gcp_satisfies_containing_params(
            gcp
        ) and self.gcp_satisfies_avoiding_params(gcp)

    def gcp_satisfies_avoiding_params(self, gcp: GriddedCayleyPerm) -> bool:  # Good
        """Returns True if the gridded cayley permutation satisfies the avoiding parameters"""
        return not any(
            any(True for _ in param.preimage_of_gcp(gcp))
            for param in self.avoiding_parameters
        )

    def gcp_satisfies_containing_params(self, gcp: GriddedCayleyPerm) -> bool:  # Good
        """Returns True if the gridded cayley permutation satisfies the containing parameters"""
        return all(
            any(any(True for _ in param.preimage_of_gcp(gcp)) for param in params)
            for params in self.containing_parameters
        )

    def reap_contradictory_ghosts(self):  # BAD
        """Removes parameters which are contradictory"""
        for n in range(len(self.parameters)):
            ghost = self.parameters[n]
            if ghost.is_contradictory(self.tiling):
                self.kill_ghost(n)

    def kill_ghost(self, ghost_number: int):  # BAD
        """removes a ghost from the mapped tiling"""
        new_ghost = self.parameters.pop(ghost_number)
        for i in range(new_ghost.ghost.dimensions[0]):
            for j in range(new_ghost.ghost.dimensions[1]):
                new_ghost.ghost = new_ghost.ghost.add_obstruction(
                    GriddedCayleyPerm(CayleyPermutation([0]), [(i, j)])
                )
        self.parameters.append(new_ghost)

    def is_trivial(self, confidence=8):  # TODO: Make this better and based on theory
        return set(self.objects_of_size(confidence)) == set(
            self.tiling.objects_of_size(confidence)
        )

    def avoiders_are_trivial(self):
        for param in self.avoiding_parameters:
            if param.ghost != self.tiling:
                return False
        return True
                

    def is_contradictory(
        self, confidence=8
    ):  # TODO: Make this better and based on theory and correct
        return len(set(self.objects_of_size(confidence))) == 0

    def pop_parameter(self, parameter_index=0):  # BAD
        """removes the parameter at an index and creates a new mapped tiling"""
        param = self.parameters.pop(parameter_index)
        return MappedTiling(self.tiling, [param])

    def pop_all_parameters(self):  # BAD
        """yields all mapped tilings with a single parameter"""
        while len(self.parameters) > 0:
            yield self.pop_parameter()

    def add_parameter(self, parameter: Parameter):  # BAD
        self.parameters.append(parameter)

    def add_obs_to_param_list(
        self, parameters: List[Parameter], obs: List[GriddedCayleyPerm]
    ):  # Good
        """Adds obstructions to a list of parameters and returns the new list"""
        new_parameters = []
        for parameter in parameters:
            new_parameter = parameter.ghost.add_obstructions(
                parameter.map.preimage_of_obstructions(obs)
            )
            new_parameters.append(Parameter(new_parameter, parameter.map))
        return new_parameters

    def add_obstructions(self, obstructions: List[GriddedCayleyPerm]):  # Good
        """Adds obstructions to the tiling (and corrects the parameters)"""
        new_containing_parameters = []
        for parameter_list in self.containing_parameters:
            new_containing_parameters.append(
                self.add_obs_to_param_list(parameter_list, obstructions)
            )
        new_enumeration_parameters = []
        for parameter_list in self.enumeration_parameters:
            new_enumeration_parameters.append(
                self.add_obs_to_param_list(parameter_list, obstructions)
            )
        return MappedTiling(
            self.tiling.add_obstructions(obstructions),
            self.add_obs_to_param_list(self.avoiding_parameters, obstructions),
            new_containing_parameters,
            new_enumeration_parameters,
        )

    def add_reqs_to_param_list(
        self, parameters: List[Parameter], reqs: List[List[GriddedCayleyPerm]]
    ):  # Good
        """Adds requirements to a list of parameters and returns the new list"""
        new_parameters = []
        for parameter in parameters:
            new_parameter = parameter.ghost.add_requirements(
                parameter.map.preimage_of_requirements(reqs)
            )
            new_parameters.append(Parameter(new_parameter, parameter.map))
        return new_parameters

    def add_requirements(self, requirements: List[List[GriddedCayleyPerm]]):  # Good
        """Adds requirements to the mappling by adding them to each of the
        parameters in all possible ways."""
        new_containing_parameters = []
        for parameter_list in self.containing_parameters:
            new_containing_parameters.append(
                self.add_reqs_to_param_list(parameter_list, requirements)
            )
        new_enumeration_parameters = []
        for parameter_list in self.enumeration_parameters:
            new_enumeration_parameters.append(
                self.add_reqs_to_param_list(parameter_list, requirements)
            )

        return MappedTiling(
            self.tiling.add_requirements(requirements),
            self.add_reqs_to_param_list(self.avoiding_parameters, requirements),
            new_containing_parameters,
            new_enumeration_parameters,
        )

    def remove_empty_rows_and_columns(self):  # Good
        """Finds and removes empty rows and cols in the base tiling then removes the
        corresponding rows and columns in the parameters"""
        empty_cols, empty_rows = self.tiling.find_empty_rows_and_columns()
        new_tiling = self.tiling.delete_rows_and_columns(empty_cols, empty_rows)
        new_avoiding_parameters = self.remove_empty_rows_and_cols_from_param_list(
            self.avoiding_parameters, empty_cols, empty_rows
        )
        new_containing_parameters = []
        for parameter_list in self.containing_parameters:
            new_containing_parameters.append(
                self.remove_empty_rows_and_cols_from_param_list(
                    parameter_list, empty_cols, empty_rows
                )
            )
        new_enumeration_parameters = []
        for parameter_list in self.enumeration_parameters:
            new_enumeration_parameters.append(
                self.remove_empty_rows_and_cols_from_param_list(
                    parameter_list, empty_cols, empty_rows
                )
            )
        return MappedTiling(
            new_tiling,
            new_avoiding_parameters,
            new_containing_parameters,
            new_enumeration_parameters,
        )
        
    
    def remove_empty_rows_and_cols_from_param_list(
        self, parameters, empty_cols, empty_rows
    ):  # Good
        """Removes the rows and cols from each ghost in the parameter list then
        returns new parameter list."""
        new_parameters = []
        for P in parameters:
            col_preimages, row_preimages = P.map.preimages_of_cols(
                empty_cols
            ), P.map.preimages_of_rows(empty_rows)
            new_parameter = P.ghost.delete_rows_and_columns(
                col_preimages, row_preimages
            )
            new_map = P.reduce_row_col_map(col_preimages, row_preimages)
            new_parameters.append(Parameter(new_parameter, new_map))
        return new_parameters

    def all_parameters(self):  # Good
        """Returns a list of all parameters."""
        return (
            self.avoiding_parameters
            + list(chain.from_iterable(self.containing_parameters))
            + list(chain.from_iterable(self.enumeration_parameters))
        )

    def __eq__(self, other) -> bool:
        return (
            self.tiling == other.tiling
            and self.avoiding_parameters == other.avoiding_parameters
            and self.containing_parameters == other.containing_parameters
            and self.enumeration_parameters == other.enumeration_parameters
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.tiling,
                tuple(self.avoiding_parameters),
                tuple(self.containing_parameters),
                tuple(self.enumeration_parameters),
            )
        )

    def from_dict(self, d):
        return MappedTiling(
            Tiling.from_dict(d["tiling"]),
            [Parameter.from_dict(p) for p in d["avoiding_parameters"]],
            [[Parameter.from_dict(p) for p in ps] for ps in d["containing_parameters"]],
            [
                [Parameter.from_dict(p) for p in ps]
                for ps in d["enumeration_parameters"]
            ],
        )

    def is_empty(self) -> bool:
        return self.tiling.is_empty()

    def __repr__(self):
        return str(
            (
                repr(self.tiling),
                [repr(p) for p in self.avoiding_parameters],
                [[repr(p) for p in ps] for ps in self.containing_parameters],
                [[repr(p) for p in ps] for ps in self.enumeration_parameters],
            )
        )

    def __str__(self) -> str:
        return (
            "Base tiling: \n"
            + str(self.tiling)
            + "\nAvoiding parameters:\n"
            + "\n".join([str(p) for p in self.avoiding_parameters])
            + "\nContaining parameters:\n"
            + "\nNew containing parameters list \n".join(
                ["\n".join([str(p) for p in ps]) for ps in self.containing_parameters]
            )
            + "\nEnumeration parameters:\n"
            + "\nNew enumeration parameters list\n".join(
                ["\n".join([str(p) for p in ps]) for ps in self.enumeration_parameters]
            )
        )
