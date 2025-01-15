from typing import Iterable, Iterator, Tuple, List, Dict, DefaultDict
from itertools import chain
from collections import defaultdict
from comb_spec_searcher import CombinatorialClass
from gridded_cayley_permutations import Factors


from cayley_permutations import CayleyPermutation
from gridded_cayley_permutations import (
    RowColMap,
    GriddedCayleyPerm,
    Tiling,
    PointPlacement,
)

Objects = DefaultDict[Tuple[int, ...], List[GriddedCayleyPerm]]


class Parameter:
    def __init__(self, tiling, row_col_map):
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

    def objects_of_size(self, n, **parameters):
        for val in self.get_objects(n).values():
            for gcp in val:
                yield gcp

    def get_objects(self, n: int) -> Objects:
        objects = defaultdict(list)
        for gcp in self.tiling.objects_of_size(n):
            if self.gcp_in_tiling(gcp):
                param = self.get_parameters(gcp)
                objects[param].append(gcp)
        return objects

    def get_parameters(self, gcp: GriddedCayleyPerm) -> Tuple[int, ...]:
        """Parameters are not what you think!!! This is specific to combinatorical class parameters"""
        all_lists = []
        for param_list in self.enumeration_parameters:
            all_lists.append(
                sum(1 for _ in param.preimage_of_gcp(gcp)) for param in param_list
            )
        return tuple(all_lists)

    def gcp_in_tiling(self, gcp: GriddedCayleyPerm) -> bool:
        """Returns True if the gridded cayley permutation is in the tiling"""
        return self.gcp_satisfies_containing_params(
            gcp
        ) and self.gcp_satisfies_avoiding_params(gcp)

    def gcp_satisfies_avoiding_params(self, gcp: GriddedCayleyPerm) -> bool:
        """Returns True if the gridded cayley permutation satisfies the avoiding parameters"""
        return not any(
            any(True for _ in param.preimage_of_gcp(gcp))
            for param in self.avoiding_parameters
        )

    def gcp_satisfies_containing_params(self, gcp: GriddedCayleyPerm) -> bool:
        """Returns True if the gridded cayley permutation satisfies the containing parameters"""
        return all(
            any(any(True for _ in param.preimage_of_gcp(gcp)) for param in params)
            for params in self.containing_parameters
        )

    def reap_contradictory_ghosts(self):
        """Removes parameters which are contradictory"""
        for n in range(len(self.parameters)):
            ghost = self.parameters[n]
            if ghost.is_contradictory(self.tiling):
                self.kill_ghost(n)

    def kill_ghost(self, ghost_number: int):
        """removes a ghost from the mapped tiling"""
        new_ghost = self.parameters.pop(ghost_number)
        for i in range(new_ghost.ghost.dimensions[0]):
            for j in range(new_ghost.ghost.dimensions[1]):
                new_ghost.ghost = new_ghost.ghost.add_obstruction(
                    GriddedCayleyPerm(CayleyPermutation([0]), [(i, j)])
                )
        self.parameters.append(new_ghost)

    def pop_parameter(self, parameter_index=0):
        """removes the parameter at an index and creates a new mapped tiling"""
        param = self.parameters.pop(parameter_index)
        return MappedTiling(self.tiling, [param])

    def pop_all_parameters(self):
        """yields all mapped tilings with a single parameter"""
        while len(self.parameters) > 0:
            yield self.pop_parameter()

    def add_parameter(self, parameter: Parameter):
        self.parameters.append(parameter)

    def add_obs_to_param_list(
        self, parameters: List[Parameter], obs: List[GriddedCayleyPerm]
    ):
        """Adds obstructions to a list of parameters and returns the new list"""
        new_parameters = []
        for parameter in parameters:
            new_parameter = parameter.ghost.add_obstructions(
                parameter.map.preimage_of_obstructions(obs)
            )
            new_parameters.append(Parameter(new_parameter, parameter.map))
        return new_parameters

    def add_obstructions(self, obstructions: List[GriddedCayleyPerm]):
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
    ):
        """Adds requirements to a list of parameters and returns the new list"""
        new_parameters = []
        for parameter in parameters:
            new_parameter = parameter.ghost.add_requirements(
                parameter.map.preimage_of_requirements(reqs)
            )
            new_parameters.append(Parameter(new_parameter, parameter.map))
        return new_parameters

    def add_requirements(self, requirements: List[List[GriddedCayleyPerm]]):
        """Adds requirements to the tiling (and corrects the parameters)."""
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

    def point_placement(self, cell: Tuple[int, int], direction: int) -> "MappedTiling":
        """Returns the point placement of a cell in a direction"""
        point = [GriddedCayleyPerm(CayleyPermutation([0]), [cell])]
        indices = (0,)
        new_tiling = PointPlacement(self.tiling).point_placement(
            point, indices, direction
        )[0]
        new_continaing_parameters = []
        for parameter_list in self.containing_parameters:
            new_continaing_parameters.append(
                self.add_point_to_param_list(
                    parameter_list, point, indices, direction, cell
                )
            )
        new_enumeration_parameters = []
        for parameter_list in self.enumeration_parameters:
            new_enumeration_parameters.append(
                self.add_point_to_param_list(parameter_list, indices, direction, cell)
            )
        return MappedTiling(
            new_tiling,
            self.add_point_to_param_list(
                self.avoiding_parameters, indices, direction, cell
            ),
            new_continaing_parameters,
            new_enumeration_parameters,
        )

    def add_point_to_param_list(
        self, parameters: List[Parameter], indices, direction, cell
    ):
        """Adds a point to a list of parameters in the cell and direction
        it is being added in the base tiling and returns the new list"""
        new_parameters = []
        for parameter in parameters:
            for preimage_cell in parameter.map.preimage_of_cell(cell):
                point = [GriddedCayleyPerm(CayleyPermutation([0]), [preimage_cell])]
                new_param = PointPlacement(parameter.ghost).point_placement(
                    point, indices, direction
                )[0]
                new_map = parameter.expand_row_col_map_at_index(
                    2, 2, preimage_cell[0], preimage_cell[1]
                )
                new_parameters.append(Parameter(new_param, new_map))
        return new_parameters

    def remove_empty_rows_and_columns(self):
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
    ):
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

    def all_parameters(self):
        """Returns a list of all parameters."""
        return (
            self.avoiding_parameters
            + list(chain.from_iterable(self.containing_parameters))
            + list(chain.from_iterable(self.enumeration_parameters))
        )

    def find_factor_cells(self):
        """Returns a partition of the cells so that the mapped tiling is factored."""
        parameters = self.all_parameters()
        all_factors = Factors(self.tiling).find_factors_tracked()
        for parameter in parameters:
            t_factors = all_factors
            print(all_factors)
            p_factors = Factors(parameter.ghost).find_factors_tracked()
            all_factors = []
            queue = t_factors
            while queue:
                t_factor = queue.pop()
                final_t_factor = t_factor
                final_p_factors = []
                new_t_factors = [t_factor]
                while True:
                    new_p_factors = []
                    for t_factor in new_t_factors:
                        p_factors_so_far = self.map_t_factor_to_p_factor(
                            t_factor, parameter, p_factors
                        )
                        p_factors = [p for p in p_factors if p not in p_factors_so_far]
                        for P in p_factors_so_far:
                            final_p_factors += P
                        new_p_factors += p_factors_so_far
                    new_t_factors = []
                    for p_factor in new_p_factors:
                        temp = self.map_p_factor_to_t_factor(p_factor, parameter, queue)
                        new_t_factors += temp
                        queue = [t for t in queue if t not in temp]
                    if not new_t_factors:
                        break
                    for T in new_t_factors:
                        final_t_factor += T
                print(final_t_factor)
                all_factors.append(final_t_factor)
        return all_factors

    def is_factorable(self) -> bool:
        """Returns True if the mapped tiling is factorable."""
        factors = self.find_factor_cells()
        for factor in factors:
            pass

    def find_factors(self):
        for factor in self.find_factor_cells():
            factor_tiling = self.tiling.sub_tiling(factor)
            factor_avoiding_parameters = []
            for avoiding_param in self.avoiding_parameters:
                factor_avoiding_parameters.append(avoiding_param.sub_parameter(factor))
            factor_containing_parameters = [
                [contain_param.sub_parameter(factor) for contain_param in param_list]
                for param_list in self.containing_parameters
            ]
            factor_enumeration_parameters = [
                [enum_param.sub_parameter(factor) for enum_param in param_list]
                for param_list in self.enumeration_parameters
            ]
            yield MappedTiling(
                factor_tiling,
                factor_avoiding_parameters,
                factor_containing_parameters,
                factor_enumeration_parameters,
            ).remove_empty_rows_and_columns()

    @staticmethod
    def map_p_factor_to_t_factor(p_factor, parameter, t_factors):
        """maps a factor of the parameter to a list of factors of the tiling"""
        image_cells = set(
            (parameter.map.col_map[cell[0]], parameter.map.row_map[cell[1]])
            for cell in p_factor
        )
        return [
            factor
            for factor in t_factors
            if any(cell in image_cells for cell in factor)
        ]

    @staticmethod
    def map_t_factor_to_p_factor(t_factor, parameter, p_factors):
        """maps a factor of the tiling to a list of parameters"""
        preimage_cells = set()
        for cell in t_factor:
            preimage_cells = preimage_cells.union(parameter.map.preimage_of_cell(cell))
        return [
            factor
            for factor in p_factors
            if any(cell in preimage_cells for cell in factor)
        ]

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
            str(self.tiling)
            + "\nAvoiding parameters:\n"
            + "\n".join([str(p) for p in self.avoiding_parameters])
            + "\nContaining parameters:\n"
            + "\n".join([str(p) for p in self.containing_parameters])
            + "\nEnumeration parameters:\n"
            + "\n".join([str(p) for p in self.enumeration_parameters])
        )
