"""
Procedures for generating resonance structures inside ChemGraph objects.

TODO: bit of a mess, should be revised
"""

import itertools
from copy import deepcopy

import numpy as np

from ..misc_procedures import InvalidAdjMat, all_equal, sorted_by_membership, sorted_tuple
from ..periodic import get_max_charge_feasibility
from .base_chem_graph import BaseChemGraph
from .heavy_atom import HeavyAtom

max_resonance_structure_number = None


def set_max_resonance_structure_number(new_max_resonance_structure_number):
    global max_resonance_structure_number
    max_resonance_structure_number = new_max_resonance_structure_number


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
        if avail_charges[0] != 0:
            raise InvalidAdjMat
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

    def assign_charge_feasibility(self, new_charge_feasibility):
        self.charge_feasibility = new_charge_feasibility

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
            **charge_configuration_character_kwargs,
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


# Subroutines for enumerating all ways extra valences can be "paired up" into non-sigma bonds.
class ExtraValenceAddedEdgesIterator:
    def __init__(self, extra_valence_subgraph: ExtraValenceSubgraph, valences):
        self.extra_valence_subgraph = extra_valence_subgraph
        self.extra_val_ids = extra_valence_subgraph.extra_val_ids
        self.valences = valences
        # store some values that will be reused a lot
        self.saved_neighbors = {}
        self.max_added_bonds = {}
        max_bond_orders = {}
        for internal_index, true_index in enumerate(self.extra_val_ids):
            self.saved_neighbors[internal_index] = list(
                self.extra_valence_subgraph.extra_val_subgraph.neighbors(internal_index)
            )
            max_bo = self.extra_valence_subgraph.chemgraph.hatoms[true_index].max_bo()
            max_bond_orders[internal_index] = max_bo
        for internal_index1 in range(len(self.valences)):
            for internal_index2 in self.saved_neighbors[internal_index1]:
                if internal_index2 > internal_index1:
                    continue
                max_added_bond_order = (
                    min(max_bond_orders[internal_index1], max_bond_orders[internal_index2]) - 1
                )
                if max_added_bond_order == 0:
                    # remove the edge from consideration;
                    self.saved_neighbors[internal_index1].remove(internal_index2)
                    self.saved_neighbors[internal_index2].remove(internal_index1)
                else:
                    self.max_added_bonds[(internal_index2, internal_index1)] = max_added_bond_order

    def internal_neighbors(self, atom_id):
        atom_id = int(atom_id)
        return self.saved_neighbors[atom_id]

    def internal_neighbors_wvalences(self, atom_id):
        return (
            neigh_id
            for neigh_id in self.internal_neighbors(atom_id)
            if self.extra_valences[neigh_id] != 0
        )

    def get_cur_decision_lbound(self, atom_id):
        return self.decision_lbounds[atom_id][-1]

    def neighbor_above_lbound(self, atom_id, neighbor_id):
        return (
            atom_id not in self.decision_lbounds
            or self.get_cur_decision_lbound(atom_id) <= neighbor_id
        )

    def nonmax_added_bonds(self, *atom_ids):
        st = sorted_tuple(*atom_ids)
        return not (
            st in self.cur_bond_dict and self.cur_bond_dict[st] >= self.max_added_bonds[st]
        )

    def get_atom_connection_opportunities(self, atom_id):
        return (
            neigh_id
            for neigh_id in self.internal_neighbors_wvalences(atom_id)
            if self.neighbor_above_lbound(atom_id, neigh_id)
            and self.nonmax_added_bonds(atom_id, neigh_id)
        )

    def count_atom_connection_opportunities(self, atom_id):
        return sum(1 for _ in self.get_atom_connection_opportunities(atom_id))

    def update_atom_connection_opportunities(self, *atom_ids):
        has_isolated_valence = False
        for atom_id in atom_ids:
            self.connection_opportunities[atom_id] = self.count_atom_connection_opportunities(
                atom_id
            )
            if self.connection_opportunities[atom_id] == 0:
                has_isolated_valence = True
        return has_isolated_valence

    def finalize_bond_dict(self, internal_added_bond_dict):
        """
        Transform a list of added edges into a bond order dictionnary.
        """
        add_bond_orders = {}
        for bond_tuple, bond_order in internal_added_bond_dict.items():
            se = sorted_tuple(*[self.extra_val_ids[i] for i in bond_tuple])
            add_bond_orders[se] = bond_order
        return add_bond_orders

    def update_affected_neighbors(self, atom_ids, need_update_vals):
        has_isolated_valence = False
        for atom_id, need_update_val in zip(atom_ids, need_update_vals):
            if not need_update_val:
                continue
            has_isolated_valence = (
                self.update_atom_connection_opportunities(
                    *self.internal_neighbors_wvalences(atom_id)
                )
                or has_isolated_valence
            )
        return has_isolated_valence

    def add_valence_electron(self, atom_id):
        previously_no_valence = self.extra_valences[atom_id] == 0
        self.extra_valences[atom_id] += 1
        return previously_no_valence

    def reopen_pair(self, *atom_ids):
        bt = sorted_tuple(*atom_ids)
        if self.cur_bond_dict[bt] == 1:
            del self.cur_bond_dict[bt]
        else:
            self.cur_bond_dict[bt] -= 1
        created_valence_vals = [self.add_valence_electron(atom_id) for atom_id in atom_ids]
        self.update_atom_connection_opportunities(*atom_ids)
        self.update_affected_neighbors(atom_ids, created_valence_vals)

    def remove_valence_electron(self, atom_id):
        self.extra_valences[atom_id] -= 1
        currently_no_valence = self.extra_valences[atom_id] == 0
        if self.extra_valences[atom_id] == 0:
            self.connection_opportunities[atom_id] = 0
        return currently_no_valence

    def close_pair(self, *atom_ids):
        bt = sorted_tuple(*atom_ids)
        if bt in self.cur_bond_dict:
            self.cur_bond_dict[bt] += 1
        else:
            self.cur_bond_dict[bt] = 1
        no_valence_vals = [self.remove_valence_electron(atom_id) for atom_id in atom_ids]
        has_isolated_valence = False
        for atom_id, no_valence in zip(atom_ids, no_valence_vals):
            if not no_valence:
                has_isolated_valence = (
                    self.update_atom_connection_opportunities(atom_id) or has_isolated_valence
                )
        self.cur_closed_atom_pairs.append(atom_ids)
        has_isolated_valence = (
            self.update_affected_neighbors(atom_ids, no_valence_vals) or has_isolated_valence
        )
        self.cur_decision_level += 1
        return has_isolated_valence

    def get_nonzero_extra_valences(self):
        return np.nonzero(self.extra_valences)[0]

    def get_starting_bond_atom(self):
        # we choose an atom from which outstretch a bond
        possible_starting_bond_atoms = np.nonzero(self.connection_opportunities)[0]
        if len(possible_starting_bond_atoms) == 1:
            return int(possible_starting_bond_atoms[0])
        min_connectivity = np.min(self.connection_opportunities[possible_starting_bond_atoms])
        return int(np.where(self.connection_opportunities == min_connectivity)[0][0])

    def complete_current_closed_atoms(self):
        while np.any(self.extra_valences != 0):
            if np.all(self.connection_opportunities == 0):
                return
            starting_bond_atom = self.get_starting_bond_atom()
            atom_connection_opportunities = list(
                self.get_atom_connection_opportunities(starting_bond_atom)
            )
            final_opportunity = atom_connection_opportunities[0]

            if len(atom_connection_opportunities) != 1:
                assert len(atom_connection_opportunities) != 0
                self.decision_possibilities[
                    self.cur_decision_level
                ] = atom_connection_opportunities
                self.decisions[self.cur_decision_level] = 0
                if starting_bond_atom in self.decision_lbounds:
                    self.decision_lbounds[starting_bond_atom].append(final_opportunity)
                else:
                    self.decision_lbounds[starting_bond_atom] = [final_opportunity]
            has_isolated_valence = self.close_pair(starting_bond_atom, final_opportunity)
            if has_isolated_valence:
                return

    def next_closed_atoms(self):
        found = False
        while self.cur_decision_level != 0:
            self.cur_decision_level -= 1
            prev_starting_atom, prev_other_atom = self.cur_closed_atom_pairs.pop()
            if self.cur_decision_level in self.decisions:
                self.decisions[self.cur_decision_level] += 1
                decision_possibilities = self.decision_possibilities[self.cur_decision_level]
                new_decision = self.decisions[self.cur_decision_level]
                if new_decision == len(decision_possibilities):
                    del self.decisions[self.cur_decision_level]
                    del self.decision_possibilities[self.cur_decision_level]
                    self.decision_lbounds[prev_starting_atom].pop()
                    if len(self.decision_lbounds[prev_starting_atom]) == 0:
                        del self.decision_lbounds[prev_starting_atom]
                else:
                    new_atom_id = self.decision_possibilities[self.cur_decision_level][
                        new_decision
                    ]
                    self.decision_lbounds[prev_starting_atom][-1] = new_atom_id
                    found = True
            self.reopen_pair(prev_starting_atom, prev_other_atom)
            if found:
                self.close_pair(prev_starting_atom, new_atom_id)
                return
        raise StopIteration

    def __iter__(self):
        self.cur_decision_level = 0
        self.cur_closed_atom_pairs = []
        self.decision_possibilities = {}
        self.decisions = {}
        self.decision_lbounds = {}
        self.stopped = False
        self.cur_bond_dict = {}

        self.connection_opportunities = np.zeros(len(self.extra_val_ids), dtype=int)
        # Check how many neighboring atoms a given atom can be connected to with the nonsigma bonds.
        self.extra_valences = self.extra_valence_subgraph.get_extra_valences(self.valences)
        if np.all(self.extra_valences == 0):
            return iter(({},))
        for cur_id in self.get_nonzero_extra_valences():
            self.update_atom_connection_opportunities(cur_id)
        return self

    def __next__(self):
        final_dict = None
        while True:
            if self.stopped:
                raise StopIteration
            self.complete_current_closed_atoms()
            if np.all(self.extra_valences == 0):
                final_dict = deepcopy(self.cur_bond_dict)
            try:
                self.next_closed_atoms()
            except StopIteration:
                self.stopped = True
            if final_dict is not None:
                return final_dict


def complete_valences_attempt(extra_valence_subgraph, valences, all_possibilities=False):
    """
    For a subgraph of the chemical graph extra_val_subgraph which spawns over hatoms with ids extra_val_ids with
    coordination numbers coord_nums, generate assignment of all non-sigma valence electrons into nonsigma bonds.
    If all_possibilities is False returns one such resonance structure, otherwise enumerate and return all resonance structures.
    """
    if all_possibilities:
        full_list = []
        added_bond_dict_canon_keys = set()
    iterator = ExtraValenceAddedEdgesIterator(extra_valence_subgraph, valences)
    for added_bonds_dict in iterator:
        if all_possibilities:
            added_bonds_dict_canon_key = frozenset(added_bonds_dict.items())
            if added_bonds_dict_canon_key not in added_bond_dict_canon_keys:
                full_list.append(iterator.finalize_bond_dict(added_bonds_dict))
                if (
                    max_resonance_structure_number is not None
                    and max_resonance_structure_number == len(full_list)
                ):
                    break
                added_bond_dict_canon_keys.add(added_bonds_dict_canon_key)
        else:
            return added_bonds_dict
    if all_possibilities and len(full_list) != 0:
        return full_list
    return None


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
    extra_valence_subgraph.assign_charge_feasibility(charge_feasibility)
    valid_valence_configuration_character = None
    tot_subgraph_res_struct = []
    valence_option_ids = []
    current_valence_option = 0
    try:
        extra_valence_iterator = iter(extra_valence_subgraph)
    except (StopIteration, InvalidAdjMat):
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
        assert res_struct_redundant is not None
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
            if chemgraph.overall_charge_feasibility > get_max_charge_feasibility():
                raise InvalidAdjMat
