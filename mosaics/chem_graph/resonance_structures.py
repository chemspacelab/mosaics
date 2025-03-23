"""
Procedures for generating resonance structures inside ChemGraph objects.
"""

import itertools
from copy import deepcopy

import numpy as np

from ..misc_procedures import InvalidAdjMat, all_equal, sorted_by_membership, sorted_tuple
from ..periodic import max_charge_feasibility
from .base_chem_graph import BaseChemGraph
from .heavy_atom import HeavyAtom


# Procedures for generating which valence and charge states to iterate over.
# TODO Need a smarter way to iterate that does not involve creating a long list and sorting it.
def divide_by_character(input_list, character_function, **character_function_kwargs):
    character = None
    divisors = []
    for i, element in enumerate(input_list):
        new_character = character_function(element, **character_function_kwargs)
        if (character is not None) and (new_character == character):
            continue
        character = new_character
        divisors.append(i)
    if character == np.inf:
        if len(divisors) == 1:
            raise StopIteration
    else:
        divisors.append(len(input_list))
    return divisors


def ValenceConfigurationCharacter(valences: list):
    """
    Measure of how realistic valences for a resonance structure are.
    """
    return sum(valences)


def ChargeConfigurationCharacter(charges_valences: list, chemgraph_charge=None):
    """
    Measure of how realistic charges are.
    """
    tot_charge = 0
    charge_deviation = 0
    for charge, _ in charges_valences:
        tot_charge += charge
        charge_deviation += np.abs(charge)

    if chemgraph_charge is not None:
        if tot_charge == chemgraph_charge:
            return charge_deviation
        else:
            return np.inf
    return np.abs(tot_charge), charge_deviation


def iter_prod_list(iters: list):
    return list(itertools.product(*iters))


def valences_iterator(all_avail_valences: list, extra_val_ids: list, hatom_list: list):
    HeavyAtomValenceIterators = []
    IteratedValenceIds = []
    for hatom_id, avail_valences in zip(extra_val_ids, all_avail_valences):
        if len(avail_valences) == 1:
            hatom_list[hatom_id].valence = avail_valences[0]
            continue
        HeavyAtomValenceIterators.append(avail_valences)
        IteratedValenceIds.append(hatom_id)
    HeavyAtomValences = iter_prod_list(HeavyAtomValenceIterators)

    return HeavyAtomValences, IteratedValenceIds


def feasibility_checked_charges_valences(
    avail_charges: list, avail_valences: list, hatom: HeavyAtom, charge_feasibility: int
):
    if hatom.can_be_charged(charge_feasibility):
        return avail_charges, avail_valences
    else:
        assert avail_charges[0] == 0, InvalidAdjMat
        return avail_charges[:1], avail_valences[:1]


class ExtraValenceSubgraph:
    def __init__(
        self,
        chemgraph: BaseChemGraph,
        extra_val_ids,
        coord_nums,
        extra_val_subgraph,
        subgraph_avail_charges,
        subgraph_avail_valences,
        charge_feasibility=0,
    ):
        """
        Auxiliary class storing information about a resonance structure region and iterating over all possible charge and valence states.
        """
        self.chemgraph = chemgraph
        self.extra_val_ids = extra_val_ids
        self.coord_nums = coord_nums
        self.extra_val_subgraph = extra_val_subgraph

        self.subgraph_avail_charges = subgraph_avail_charges
        self.subgraph_avail_valences = subgraph_avail_valences

        self.charge_feasibility = charge_feasibility

    def empty_id_lists(self):
        return [[] for _ in self.extra_val_ids]

    def init_valences_charges_iterator(self):
        """
        Combine lists of available charges and valences into an iterator where charge and valence configurations are sorted by relevance. If charge or valence cannot be iterated for a     heavy atom initialize the valence there as well.
        """
        HeavyAtomChargeIterators = []
        for hatom_id, avail_charges, avail_valences in zip(
            self.extra_val_ids,
            self.subgraph_avail_charges,
            self.subgraph_avail_valences,
        ):
            checked_avail_charges, checked_avail_valences = feasibility_checked_charges_valences(
                avail_charges,
                avail_valences,
                self.chemgraph.hatoms[hatom_id],
                self.charge_feasibility,
            )
            HeavyAtomChargeIterators.append(zip(checked_avail_charges, checked_avail_valences))
        self.HeavyAtomCharges = iter_prod_list(HeavyAtomChargeIterators)
        if self.chemgraph.resonance_structures_merged:
            unaccounted_charge = self.chemgraph.charge
            for ha in self.chemgraph.hatoms:
                if ha.charge is not None:
                    unaccounted_charge -= ha.charge
            charge_configuration_character_kwargs = {"chemgraph_charge": unaccounted_charge}
        else:
            charge_configuration_character_kwargs = {}
        self.HeavyAtomCharges.sort(
            key=lambda x: ChargeConfigurationCharacter(x, **charge_configuration_character_kwargs)
        )
        self.HeavyAtomCharges_divisors = divide_by_character(
            self.HeavyAtomCharges,
            ChargeConfigurationCharacter,
            **charge_configuration_character_kwargs
        )

    def init_piecewise_valences_charges_iterator(self):
        if self.considered_charge_character_interval == len(self.HeavyAtomCharges_divisors) - 1:
            raise StopIteration
        div1, div2 = self.HeavyAtomCharges_divisors[
            self.considered_charge_character_interval : self.considered_charge_character_interval
            + 2
        ]
        self.partialHeavyAtomCharges = []
        for charges_valences_iterators in self.HeavyAtomCharges[div1:div2]:
            charges = [t[0] for t in charges_valences_iterators]
            avail_valences = iter_prod_list([t[1] for t in charges_valences_iterators])
            self.partialHeavyAtomCharges += list(
                zip(itertools.repeat(charges, len(avail_valences)), avail_valences)
            )
        self.partialHeavyAtomCharges.sort(key=lambda x: ValenceConfigurationCharacter(x[1]))
        self.partialHeavyAtomCharges_iter = iter(self.partialHeavyAtomCharges)

    def __iter__(self):
        # where valid charge and valenc values will be stored
        self.possible_charges = self.empty_id_lists()
        self.possible_valences = self.empty_id_lists()
        # which charges and valences are currently considered
        self.current_charges = None
        self.current_valences = None

        self.valid_valence_configuration_character = None

        self.init_valences_charges_iterator()
        self.considered_charge_character_interval = 0
        self.init_piecewise_valences_charges_iterator()

        return self

    def __next__(self):
        try:
            output = self.partialHeavyAtomCharges_iter.__next__()
        except StopIteration:
            if self.valid_valence_configuration_character is not None:
                raise StopIteration
            self.considered_charge_character_interval += 1
            self.init_piecewise_valences_charges_iterator()
            output = self.partialHeavyAtomCharges_iter.__next__()
        if (self.valid_valence_configuration_character is not None) and (
            ValenceConfigurationCharacter(output[1]) != self.valid_valence_configuration_character
        ):
            raise StopIteration
        self.current_charges, self.current_valences = output
        return output

    def append_current_valence_state(self):
        for i, (charge, valence) in enumerate(zip(self.current_charges, self.current_valences)):
            self.possible_charges[i].append(charge)
            self.possible_valences[i].append(valence)

    def collapse_repetition(self, possibility_list):
        val = possibility_list[0]
        if all_equal(possibility_list):
            return val, None
        else:
            return val, possibility_list

    def dump_appended_valence_states(self):
        for true_id, possible_charges, possible_valences in zip(
            self.extra_val_ids, self.possible_charges, self.possible_valences
        ):
            ha = self.chemgraph.hatoms[true_id]
            ha.charge, ha.possible_charges = self.collapse_repetition(possible_charges)
            ha.valence, ha.possible_valences = self.collapse_repetition(possible_valences)

    def get_extra_valences(self, valences):
        extra_valences = np.zeros(len(self.extra_val_ids), dtype=int)
        for i, (valence, coord_num) in enumerate(zip(valences, self.coord_nums)):
            extra_valences[i] = valence - coord_num
        return extra_valences

    def get_connection_opportunities(self, extra_valences):
        """
        Check how many neighboring atoms a given atom can be connected to with the nonsigma bonds.
        """

        connection_opportunities = np.zeros(len(self.extra_val_ids), dtype=int)
        # Check how many neighboring atoms a given atom can be connected to with the nonsigma bonds.
        for cur_id, extra_valence in enumerate(extra_valences):
            if extra_valence != 0:
                neighs = self.extra_val_subgraph.neighbors(cur_id)
                for neigh in neighs:
                    if extra_valences[neigh] != 0:
                        connection_opportunities[cur_id] += 1
                if connection_opportunities[cur_id] == 0:
                    return None
        return connection_opportunities


# Function saying how large can a bond order be between two atoms.
def max_bo(hatom1, hatom2):
    return 3


# Subroutines for enumerating all ways extra valences can be "paired up" into non-sigma bonds.
def possible_closed_pairs(closed_atom, extra_valences, extra_val_subgraph):
    output = []
    for i in extra_val_subgraph.neighbors(closed_atom):
        if extra_valences[i] != 0:
            output.append(i)
    return output


def added_edges_list_to_dict(added_edges, hatoms):
    """
    Transform a list of added edges into a bond order dictionnary.
    """
    add_bond_orders = {}
    for e in added_edges:
        se = sorted_tuple(*e)
        if se in add_bond_orders:
            add_bond_orders[se] += 1
            if add_bond_orders[se] == max_bo(hatoms[se[0]], hatoms[se[1]]):
                return None
        else:
            add_bond_orders[se] = 1
    return add_bond_orders


# TODO break down into more manageable pieces.
def complete_valences_attempt(
    extra_valence_subgraph: ExtraValenceSubgraph,
    valences: list,
    all_possibilities=False,
):
    """
    For a subgraph of the chemical graph extra_val_subgraph which spawns over hatoms with ids extra_val_ids with
    coordination numbers coord_nums, generate assignment of all non-sigma valence electrons into nonsigma bonds.
    If all_possibilities is False returns one such resonance structure, otherwise enumerate and return all resonance structures.
    """
    extra_val_ids = extra_valence_subgraph.extra_val_ids
    chemgraph = extra_valence_subgraph.chemgraph
    extra_valences = extra_valence_subgraph.get_extra_valences(valences)
    # TODO is it needed?
    if np.all(extra_valences == 0):
        return {}

    output = None
    added_edges = []

    connection_opportunities = extra_valence_subgraph.get_connection_opportunities(extra_valences)
    if connection_opportunities is None:
        return output

    saved_extra_valences = {}
    saved_connection_opportunities = {}
    saved_closed_atom = {}
    saved_potential_other_atoms = {}
    path_taken = {}
    added_edges_stops = {}
    cur_decision_fork = 0
    while True:
        min_connectivity = 0
        if np.any(connection_opportunities != 0):
            # Check extra electrons from which atom can be connected to the least number of neighboring atoms.
            min_connectivity = np.min(
                connection_opportunities[np.nonzero(connection_opportunities)]
            )
            closed_atom = np.where(connection_opportunities == min_connectivity)[0][0]
            potential_other_atoms = possible_closed_pairs(
                closed_atom, extra_valences, extra_valence_subgraph.extra_val_subgraph
            )
        else:
            closed_atom = None
        if min_connectivity == 1:
            # We add an extra nonsigma bond between closed_atom and the only atom for which the other such connection is possible.
            choice = 0
        else:
            cur_decision_fork += 1
            if closed_atom is None:
                if np.any(extra_valences != 0) or all_possibilities:
                    # Fall back to an earlier save point
                    while True:
                        cur_decision_fork -= 1
                        if cur_decision_fork == 0:
                            return output
                        path_taken[cur_decision_fork] += 1
                        if path_taken[cur_decision_fork] != len(
                            saved_potential_other_atoms[cur_decision_fork]
                        ):
                            break
                extra_valences[:] = saved_extra_valences[cur_decision_fork][:]
                connection_opportunities[:] = saved_connection_opportunities[cur_decision_fork][:]
                potential_other_atoms = deepcopy(saved_potential_other_atoms[cur_decision_fork])
                closed_atom = saved_closed_atom[cur_decision_fork]
                del added_edges[added_edges_stops[cur_decision_fork] :]
            else:
                path_taken[cur_decision_fork] = 0
                saved_extra_valences[cur_decision_fork] = np.copy(extra_valences)
                saved_connection_opportunities[cur_decision_fork] = np.copy(
                    connection_opportunities
                )
                saved_potential_other_atoms[cur_decision_fork] = deepcopy(potential_other_atoms)
                saved_closed_atom[cur_decision_fork] = closed_atom
                added_edges_stops[cur_decision_fork] = len(added_edges)
            choice = path_taken[cur_decision_fork]
        other_closed_atom = potential_other_atoms[choice]

        # Add the extra nonsigma bond.
        added_edges.append((extra_val_ids[closed_atom], extra_val_ids[other_closed_atom]))
        # Delete the now tied valence atoms.
        for cur_id in [closed_atom, other_closed_atom]:
            extra_valences[cur_id] -= 1
            if extra_valences[cur_id] == 0:
                connection_opportunities[cur_id] = 0
                # Neighbors of an atom with "spent" extra valence electrons can no longer connect to it.
                for neigh_id in extra_valence_subgraph.extra_val_subgraph.neighbors(cur_id):
                    if connection_opportunities[neigh_id] != 0:
                        connection_opportunities[neigh_id] -= 1
        if np.all(extra_valences == 0):
            added_bonds_dict = added_edges_list_to_dict(added_edges, chemgraph.hatoms)
            if added_bonds_dict is None:
                if all_possibilities:
                    continue
                else:
                    raise InvalidAdjMat
            if all_possibilities:
                if output is None:
                    output = [added_bonds_dict]
                else:
                    if added_bonds_dict not in output:
                        output.append(added_bonds_dict)
            else:
                return added_bonds_dict


def gen_avail_charges_valences(chemgraph: BaseChemGraph, coordination_numbers: list):
    """
    Find atoms that possibly contain extra valences, as well as the corresponding
    """
    all_avail_charges = []
    all_avail_valences = []
    extra_valence_indices = []
    extra_valence_coord_numbers = []
    for hatom_id, hatom in enumerate(chemgraph.hatoms):
        coord_num = coordination_numbers[hatom_id]
        avail_charges, avail_valences, has_extra_valence = hatom.get_available_valences_charges(
            coord_num, charge_feasibility=chemgraph.overall_charge_feasibility
        )
        if not has_extra_valence:
            continue
        all_avail_charges.append(avail_charges)
        all_avail_valences.append(avail_valences)
        extra_valence_indices.append(hatom_id)
        extra_valence_coord_numbers.append(coord_num)
    return (
        all_avail_charges,
        all_avail_valences,
        extra_valence_indices,
        extra_valence_coord_numbers,
    )


def get_extra_valence_subgraphs(
    chemgraph: BaseChemGraph,
    extra_valence_indices: list,
    extra_valence_coord_numbers: list,
    all_avail_charges: list,
    all_avail_valences: list,
):
    """
    Divide extra valences into isolated subgraphs.
    """
    total_subgraph = chemgraph.graph.induced_subgraph(extra_valence_indices)
    if chemgraph.resonance_structures_merged:
        return [
            ExtraValenceSubgraph(
                chemgraph,
                extra_valence_indices,
                extra_valence_coord_numbers,
                total_subgraph,
                all_avail_charges,
                all_avail_valences,
            )
        ]
    ts_components = total_subgraph.components()
    members = ts_components.membership
    extra_val_subgraph_list = ts_components.subgraphs()
    extra_val_ids_lists = sorted_by_membership(members, extra_valence_indices)
    coord_nums_lists = sorted_by_membership(members, extra_valence_coord_numbers)
    avail_charges_lists = sorted_by_membership(members, all_avail_charges)
    avail_valences_lists = sorted_by_membership(members, all_avail_valences)

    output = []
    for (
        subgraph_extra_val_ids,
        subgraph_coord_nums,
        subgraph,
        subgraph_avail_charges,
        subgraph_avail_valences,
    ) in zip(
        extra_val_ids_lists,
        coord_nums_lists,
        extra_val_subgraph_list,
        avail_charges_lists,
        avail_valences_lists,
    ):
        output.append(
            ExtraValenceSubgraph(
                chemgraph,
                subgraph_extra_val_ids,
                subgraph_coord_nums,
                subgraph,
                subgraph_avail_charges,
                subgraph_avail_valences,
            )
        )
    return output


def assign_neighboring_pairs_to_resonance_structure_map(
    vertices: list, chemgraph: BaseChemGraph, resonance_region_id: int
):
    for i, vert1 in enumerate(vertices):
        neighs = chemgraph.neighbors(vert1)
        for vert2 in vertices[:i]:
            if vert2 in neighs:
                chemgraph.resonance_structure_map[(vert2, vert1)] = resonance_region_id


def create_local_resonance_structure_single_charge_feasibility(
    chemgraph: BaseChemGraph,
    extra_valence_subgraph: ExtraValenceSubgraph,
    charge_feasibility: int,
):
    extra_valence_subgraph.charge_feasibility = charge_feasibility
    valid_valence_configuration_character = None
    tot_subgraph_res_struct = []
    valence_option_ids = []
    current_valence_option = 0
    try:
        extra_valence_iterator = iter(extra_valence_subgraph)
    except StopIteration:
        return False, None
    while True:
        try:
            _, valences = extra_valence_iterator.__next__()
        except StopIteration:
            break
        subgraph_res_struct_list = complete_valences_attempt(
            extra_valence_subgraph, valences, all_possibilities=True
        )
        if subgraph_res_struct_list is None:
            continue
        if valid_valence_configuration_character is None:
            valid_valence_configuration_character = ValenceConfigurationCharacter(valences)
            extra_valence_subgraph.valid_valence_configuration_character = (
                valid_valence_configuration_character
            )
        extra_valence_subgraph.append_current_valence_state()
        tot_subgraph_res_struct += subgraph_res_struct_list
        valence_option_ids += list(
            itertools.repeat(current_valence_option, len(subgraph_res_struct_list))
        )
        current_valence_option += 1

    if valid_valence_configuration_character is None:
        return False, None

    extra_valence_subgraph.dump_appended_valence_states()

    if len(tot_subgraph_res_struct) <= 1:
        if len(tot_subgraph_res_struct) == 1:
            chemgraph.assign_extra_edge_orders(
                extra_valence_subgraph.extra_val_ids, tot_subgraph_res_struct[0]
            )
        return True, True

    if all_equal(valence_option_ids):
        # No need to adjust valences depending on the resonance structure
        valence_option_ids = [None for _ in valence_option_ids]
    chemgraph.resonance_structure_valence_vals.append(valence_option_ids)
    chemgraph.resonance_structure_orders.append(tot_subgraph_res_struct)
    chemgraph.resonance_structure_inverse_map.append(extra_valence_subgraph.extra_val_ids)
    return True, False


def create_resonance_structures_single_charge_feasibility(
    chemgraph: BaseChemGraph, coordination_numbers: list
):
    chemgraph.resonance_structure_orders = []
    chemgraph.resonance_structure_valence_vals = []
    chemgraph.resonance_structure_map = {}
    chemgraph.resonance_structure_inverse_map = []
    chemgraph.resonance_structure_charge_feasibilities = []

    for ha in chemgraph.hatoms:
        ha.clear_possibilities()
    chemgraph.bond_orders = {}

    (
        all_avail_charges,
        all_avail_valences,
        extra_valence_indices,
        extra_valence_coord_numbers,
    ) = gen_avail_charges_valences(chemgraph, coordination_numbers)
    if len(extra_valence_indices) == 0:
        return
    extra_valence_subgraph_list = get_extra_valence_subgraphs(
        chemgraph,
        extra_valence_indices,
        extra_valence_coord_numbers,
        all_avail_charges,
        all_avail_valences,
    )

    cur_resonance_region_id = 0
    for extra_valence_subgraph in extra_valence_subgraph_list:
        for charge_feasibility in range(chemgraph.overall_charge_feasibility + 1):
            (
                completed,
                res_struct_redundant,
            ) = create_local_resonance_structure_single_charge_feasibility(
                chemgraph, extra_valence_subgraph, charge_feasibility
            )
            if completed:
                break
        if not completed:
            raise InvalidAdjMat
        if res_struct_redundant:
            continue
        chemgraph.resonance_structure_charge_feasibilities.append(charge_feasibility)
        assign_neighboring_pairs_to_resonance_structure_map(
            extra_valence_subgraph.extra_val_ids, chemgraph, cur_resonance_region_id
        )
        cur_resonance_region_id += 1


def create_resonance_structures(chemgraph: BaseChemGraph):
    # first try to complete the bonds without assigning any charges
    chemgraph.overall_charge_feasibility = 0
    chemgraph.resonance_structures_merged = False
    coordination_numbers = [
        chemgraph.coordination_number(hatom_id) for hatom_id in range(chemgraph.nhatoms())
    ]
    while True:
        try:
            create_resonance_structures_single_charge_feasibility(chemgraph, coordination_numbers)
            # check that the charge are conserved
            tot_charge = sum(ha.charge for ha in chemgraph.hatoms)
            if tot_charge == chemgraph.charge:
                break
            if chemgraph.resonance_structures_merged:
                raise InvalidAdjMat
            else:
                chemgraph.resonance_structures_merged = True
                chemgraph.overall_charge_feasibility = 1
        except InvalidAdjMat:
            # try assigning more charges
            chemgraph.overall_charge_feasibility += 1
            if chemgraph.overall_charge_feasibility > max_charge_feasibility:
                raise InvalidAdjMat
