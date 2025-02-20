from .mapped_tiling import MappedTiling, Parameter
from tilescope_folder.strategies.factor import Factors
from gridded_cayley_permutations import Tiling


class MTFactor:
    def __init__(self, MappedTiling):
        self.mappling = MappedTiling

    def find_factor_cells(self):
        """Returns a partition of the cells so that the mapped tiling is factored."""
        parameters = self.mappling.all_parameters()
        all_factors = Factors(self.mappling.tiling).find_factors_tracked()
        for parameter in parameters:
            t_factors = all_factors
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
                    # print(new_t_factors)
                    for T in new_t_factors:
                        final_t_factor += T
                all_factors.append(final_t_factor)
        return all_factors

    # def is_factorable(self, confidence = 8) -> bool:
    #     """Returns True if no more than 1 factor is nontrivial in regards.
    #     TODO: This is a temporary method and not optimal"""
    #     factor_cells = self.mappling.find_factor_cells()
    #     factors = MappedTiling(self.mappling.tiling, self.mappling.avoiding_parameters,[],[]).make_factors(factor_cells)
    #     non_trivial_factors = 0
    #     for factor in factors:
    #         non_trivial_factors += int(not factor.is_trivial(confidence))
    #         if non_trivial_factors > 1:
    #             return False
    #     return True

    def is_factorable(self, factor_cells):
        factors = MTFactor(
            MappedTiling(
                self.mappling.tiling, self.mappling.avoiding_parameters, [], []
            )
        ).make_factors(factor_cells)
        non_trivial_factors = 0
        for factor in factors:
            non_trivial_factors += int(not factor.avoiders_are_trivial())
            if non_trivial_factors > 1:
                return False
        return True

    def make_factors(self, factor_cells):
        for factor in factor_cells:
            factor_tiling = self.mappling.tiling.sub_tiling(factor)
            factor_avoiding_parameters = []
            for avoiding_param in self.mappling.avoiding_parameters:
                factor_avoiding_parameters.append(avoiding_param.sub_parameter(factor))
            factor_containing_parameters = [
                [contain_param.sub_parameter(factor) for contain_param in param_list]
                for param_list in self.mappling.containing_parameters
            ]
            factor_enumeration_parameters = [
                [enum_param.sub_parameter(factor) for enum_param in param_list]
                for param_list in self.mappling.enumeration_parameters
            ]
            yield MappedTiling(
                factor_tiling,
                factor_avoiding_parameters,
                factor_containing_parameters,
                factor_enumeration_parameters,
            ).remove_empty_rows_and_columns()

    def find_factors(self):
        return self.make_factors(self.find_factor_cells())

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
