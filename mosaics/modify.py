import random
from copy import deepcopy
from typing import Union

import numpy as np
from sortedcontainers import SortedList

from .chem_graph import ChemGraph, InvalidChange, canonically_permuted_ChemGraph, str2ChemGraph
from .chem_graph.heavy_atom import next_valence
from .elementary_mutations import (
    add_heavy_atom_chain,
    atom_removal_possibilities,
    atom_replacement_possibilities,
    bond_change_possibilities,
    bond_order_change_possible_resonance_structures,
    breaking_bond_obeys_num_fragments,
    chain_addition_possibilities,
    change_bond_order,
    change_bond_order_valence,
    change_valence,
    change_valence_add_atoms,
    change_valence_remove_atoms,
    polyvalent_hatom_indices,
    remove_heavy_atom,
    replace_heavy_atom,
    valence_bond_change_possibilities,
    valence_change_add_atoms_possibilities,
    valence_change_possibilities,
    valence_change_remove_atoms_possibilities,
)
from .ext_graph_compound import (
    ExtGraphCompound,
    connection_forbidden,
    log_atom_multiplicity_in_list,
)
from .misc_procedures import (
    llenlog,
    lookup_or_none,
    random_choice_from_dict,
    random_choice_from_nested_dict,
    str_atom_corr,
)
from .periodic import max_bo_ncharges

global_step_traj_storage_label = "global"
nonglobal_step_traj_storage_label = "local"


def tp_or_chemgraph(tp):
    if isinstance(tp, ChemGraph):
        return tp
    if isinstance(tp, TrajectoryPoint):
        return tp.chemgraph()
    raise Exception()


class TrajectoryPoint:
    def __init__(
        self,
        egc: Union[ExtGraphCompound, None] = None,
        cg: Union[ChemGraph, None] = None,
        num_visits: Union[int, None] = None,
    ):
        """
        This class stores an ExtGraphCompound object along with all the information needed to preserve detailed balance of the random walk.
        egc : ExtGraphCompound object to be stored.
        cg : ChemGraph object used to define egc if the latter is None
        num_visits : initial numbers of visits to the trajectory
        """
        if egc is None:
            if cg is not None:
                egc = ExtGraphCompound(chemgraph=cg)
        self.egc = egc

        if num_visits is not None:
            num_visits = deepcopy(num_visits)
        self.num_visits = num_visits

        self.visit_step_ids = {}
        self.visit_step_num_ids = {}

        self.first_MC_step_encounter = None
        self.first_global_MC_step_encounter = None

        self.first_MC_step_acceptance = None
        self.first_global_MC_step_acceptance = None

        self.first_encounter_replica = None
        self.first_acceptance_replica = None

        # The last time minimized function was looked up for the trajectory point.
        self.last_tot_pot_call_global_MC_step = None

        # Information for keeping detailed balance.
        self.possibility_dict = None

        self.modified_possibility_dict = None

        self.calculated_data = {}

    # TO-DO better way to write this?
    def init_possibility_info(self, **kwargs):
        # self.bond_order_change_possibilities is None - to check whether the init_* procedure has been called before.
        # self.egc.chemgraph.canonical_permutation - to check whether egc.chemgraph.changed() has been called.
        if self.possibility_dict is not None:
            return
        self.egc.chemgraph.init_resonance_structures()

        change_prob_dict = lookup_or_none(kwargs, "change_prob_dict")
        if change_prob_dict is None:
            return

        self.possibility_dict = {}
        for change_procedure in change_prob_dict:
            cur_subdict = {}
            pos_label = change_possibility_label[change_procedure]
            cur_pos_generator = possibility_generator_func[change_procedure]
            if pos_label is None:
                cur_possibilities = cur_pos_generator(self.egc, **kwargs)
                if len(cur_possibilities) != 0:
                    self.possibility_dict[change_procedure] = cur_possibilities
            else:
                pos_label_vals = lookup_or_none(kwargs, pos_label)
                if pos_label_vals is None:
                    raise Exception(
                        "Randomized change parameter "
                        + pos_label
                        + " undefined, leading to problems with "
                        + str(change_procedure)
                        + ". Check code input!"
                    )
                for pos_label_val in pos_label_vals:
                    cur_possibilities = cur_pos_generator(self.egc, pos_label_val, **kwargs)
                    if len(cur_possibilities) != 0:
                        cur_subdict[pos_label_val] = cur_possibilities
                if len(cur_subdict) != 0:
                    self.possibility_dict[change_procedure] = cur_subdict

    def possibilities(self, **kwargs):
        self.init_possibility_info(**kwargs)
        return self.possibility_dict

    def clear_possibility_info(self):
        self.modified_possibility_dict = None
        self.possibility_dict = None

    def calc_or_lookup(self, func_dict, args_dict=None, kwargs_dict=None):
        output = {}
        for quant_name in func_dict.keys():
            if quant_name not in self.calculated_data:
                if args_dict is None:
                    args = ()
                else:
                    args = args_dict[quant_name]
                if kwargs_dict is None:
                    kwargs = {}
                else:
                    kwargs = kwargs_dict[quant_name]
                func = func_dict[quant_name]
                calc_val = func(self, *args, **kwargs)
                self.calculated_data[quant_name] = calc_val
            output[quant_name] = self.calculated_data[quant_name]
        return output

    def visit_num(self, replica_id):
        if self.num_visits is None:
            return 0
        else:
            return self.num_visits[replica_id]

    def mod_poss_dict_subdict(self, full_modification_path):
        cur_subdict = self.modified_possibility_dict
        for choice in full_modification_path:
            cur_subdict = cur_subdict[choice]
        return cur_subdict

    def delete_mod_poss_dict(self, full_modification_path):
        subdict = self.mod_poss_dict_subdict(full_modification_path[:-1])
        if isinstance(subdict, list):
            subdict.remove(full_modification_path[-1])
        if isinstance(subdict, dict):
            del subdict[full_modification_path[-1]]

    def delete_mod_path(self, full_modification_path):
        fmp_len = len(full_modification_path)
        while len(self.modified_possibility_dict) != 0:
            self.delete_mod_poss_dict(full_modification_path[:fmp_len])
            fmp_len -= 1
            if fmp_len == 0:
                break
            if len(self.mod_poss_dict_subdict(full_modification_path[:fmp_len])) != 0:
                break

    def copy_extra_data_to(self, other_tp, linear_storage=False, omit_data=None):
        """
        Copy all calculated data from self to other_tp.
        """
        for quant_name in self.calculated_data:
            if quant_name not in other_tp.calculated_data:
                if omit_data is not None:
                    if quant_name in omit_data:
                        continue
                other_tp.calculated_data[quant_name] = self.calculated_data[quant_name]
        # Dealing with making sure the order is preserved is too complicated.
        # if self.bond_order_change_possibilities is not None:
        #    if other_tp.bond_order_change_possibilities is None:
        #        other_tp.bond_order_change_possibilities = deepcopy(
        #            self.bond_order_change_possibilities
        #        )
        #        other_tp.chain_addition_possibilities = deepcopy(
        #            self.chain_addition_possibilities
        #        )
        #        other_tp.nuclear_charge_change_possibilities = deepcopy(
        #            self.nuclear_charge_change_possibilities
        #        )
        #        other_tp.atom_removal_possibilities = deepcopy(
        #            self.atom_removal_possibilities
        #        )
        #        other_tp.valence_change_possibilities = deepcopy(
        #            self.valence_change_possibilities
        #        )
        self.egc.chemgraph.copy_extra_data_to(
            other_tp.egc.chemgraph, linear_storage=linear_storage
        )

    def add_visit_step_id(self, step_id, beta_id, step_type=global_step_traj_storage_label):
        if step_type not in self.visit_step_ids:
            self.visit_step_ids[step_type] = {}
            self.visit_step_num_ids[step_type] = {}
        if beta_id not in self.visit_step_num_ids[step_type]:
            self.visit_step_num_ids[step_type][beta_id] = 0
            self.visit_step_ids[step_type][beta_id] = np.array([-1])
        if (
            self.visit_step_num_ids[step_type][beta_id]
            == self.visit_step_ids[step_type][beta_id].shape[0]
        ):
            self.visit_step_ids[step_type][beta_id] = np.append(
                self.visit_step_ids[step_type][beta_id],
                np.repeat(-1, self.visit_step_num_ids[step_type][beta_id]),
            )
        self.visit_step_ids[step_type][beta_id][
            self.visit_step_num_ids[step_type][beta_id]
        ] = step_id
        self.visit_step_num_ids[step_type][beta_id] += 1

    def merge_visit_data(self, other_tp):
        """
        Merge visit data with data from TrajectoryPoint in another histogram.
        """
        if other_tp.num_visits is not None:
            if self.num_visits is None:
                self.num_visits = deepcopy(other_tp.num_visits)
            else:
                self.num_visits += other_tp.num_visits

        for step_type, other_visit_step_all_ids in other_tp.visit_step_ids.items():
            for beta_id, other_visit_step_ids in other_visit_step_all_ids.items():
                other_visit_step_num_ids = other_tp.visit_step_num_ids[step_type][beta_id]
                if other_visit_step_num_ids == 0:
                    continue
                if step_type not in self.visit_step_ids:
                    self.visit_step_ids[step_type] = {}
                    self.visit_step_num_ids[step_type] = {}
                if beta_id in self.visit_step_ids[step_type]:
                    new_visit_step_ids = SortedList(
                        self.visit_step_ids[step_type][beta_id][
                            : self.visit_step_num_ids[step_type][beta_id]
                        ]
                    )
                    for visit_step_id in other_visit_step_ids[:other_visit_step_num_ids]:
                        new_visit_step_ids.add(visit_step_id)
                    self.visit_step_ids[step_type][beta_id] = np.array(new_visit_step_ids)
                    self.visit_step_num_ids[step_type][beta_id] = len(new_visit_step_ids)
                else:
                    self.visit_step_ids[step_type][beta_id] = deepcopy(other_visit_step_ids)
                    self.visit_step_num_ids[step_type][beta_id] = other_tp.visit_step_num_ids[
                        step_type
                    ][beta_id]

    def canonize_chemgraph(self):
        """
        Order heavy atoms inside the ChemGraph object according to canonical ordering. Used to make some tests consistent.
        """
        self.egc = ExtGraphCompound(chemgraph=canonically_permuted_ChemGraph(self.chemgraph()))
        self.clear_possibility_info()

    def chemgraph(self):
        return self.egc.chemgraph

    def __hash__(self):
        return hash(self.chemgraph())

    # TODO: Is comparison to ChemGraph objects worth the trouble?
    def __lt__(self, tp2):
        return self.chemgraph() < tp_or_chemgraph(tp2)

    def __gt__(self, tp2):
        return self.chemgraph() > tp_or_chemgraph(tp2)

    def __eq__(self, tp2):
        return self.chemgraph() == tp_or_chemgraph(tp2)

    def __str__(self):
        return str(self.egc)

    def __repr__(self):
        return str(self)


def str2TrajectoryPoint(str_in):
    return TrajectoryPoint(cg=str2ChemGraph(str_in))


# Minimal set of procedures that allow to claim that our MC chains are Markovian.
# replace_heavy_atom is only necessary for this claim to be valid if we are constrained to molecules with only one heavy atom.
minimized_change_list = [
    add_heavy_atom_chain,
    remove_heavy_atom,
    replace_heavy_atom,
    change_bond_order,
    change_valence,
]

# Full list of procedures for "simple MC moves" available for simulation.
full_change_list = [
    add_heavy_atom_chain,
    remove_heavy_atom,
    replace_heavy_atom,
    change_bond_order,
    change_valence,
    change_valence_add_atoms,
    change_valence_remove_atoms,
    change_bond_order_valence,
]

# A list of operations mostly (?) sufficient for exploring chemical space where polyvalent heavy atoms are not protonated.
valence_ha_change_list = [
    add_heavy_atom_chain,
    remove_heavy_atom,
    replace_heavy_atom,
    change_bond_order,
    change_valence_add_atoms,
    change_valence_remove_atoms,
    change_bond_order_valence,
]


def is_bond_change(func_in):
    return (func_in is change_bond_order) or (func_in is change_bond_order_valence)


# For randomly applying elementary mutations and maintaining detailed balance.

inverse_procedure = {
    add_heavy_atom_chain: remove_heavy_atom,
    remove_heavy_atom: add_heavy_atom_chain,
    replace_heavy_atom: replace_heavy_atom,
    change_bond_order: change_bond_order,
    change_valence: change_valence,
    change_valence_add_atoms: change_valence_remove_atoms,
    change_valence_remove_atoms: change_valence_add_atoms,
    change_bond_order_valence: change_bond_order_valence,
}

change_possibility_label = {
    add_heavy_atom_chain: "possible_elements",
    remove_heavy_atom: "possible_elements",
    replace_heavy_atom: "possible_elements",
    change_bond_order: "bond_order_changes",
    change_valence_add_atoms: "possible_elements",
    change_valence_remove_atoms: "possible_elements",
    change_valence: None,
    change_bond_order_valence: "bond_order_valence_changes",
}

possibility_generator_func = {
    add_heavy_atom_chain: chain_addition_possibilities,
    remove_heavy_atom: atom_removal_possibilities,
    replace_heavy_atom: atom_replacement_possibilities,
    change_bond_order: bond_change_possibilities,
    change_valence: valence_change_possibilities,
    change_valence_add_atoms: valence_change_add_atoms_possibilities,
    change_valence_remove_atoms: valence_change_remove_atoms_possibilities,
    change_bond_order_valence: valence_bond_change_possibilities,
}


def egc_change_func(
    egc_in: ExtGraphCompound,
    modification_path,
    change_function,
    chain_addition_tuple_possibilities=False,
    **other_kwargs,
) -> ExtGraphCompound:
    """
    Apply a modification defined through modification_path and change_function to ExtGraphCompound instance.
    """
    if (change_function is change_bond_order) or (change_function is change_bond_order_valence):
        atom_id_tuple = modification_path[1][:2]
        resonance_structure_id = modification_path[1][-1]
        bo_change = modification_path[0]

        return change_function(
            egc_in,
            *atom_id_tuple,
            bo_change,
            resonance_structure_id=resonance_structure_id,
        )
    if change_function is remove_heavy_atom:
        removed_atom_id = modification_path[1][0]
        resonance_structure_id = modification_path[1][1]
        return change_function(
            egc_in,
            removed_atom_id,
            resonance_structure_id=resonance_structure_id,
        )
    if change_function is change_valence:
        modified_atom_id = modification_path[0]
        new_valence = modification_path[1][0]
        resonance_structure_id = modification_path[1][1]
        return change_function(
            egc_in,
            modified_atom_id,
            new_valence,
            resonance_structure_id=resonance_structure_id,
        )
    if change_function is add_heavy_atom_chain:
        added_element = modification_path[0]
        if chain_addition_tuple_possibilities:
            modified_atom_id = modification_path[1][0]
            added_bond_order = modification_path[1][1]
        else:
            modified_atom_id = modification_path[1]
            added_bond_order = modification_path[2]
        return change_function(
            egc_in,
            modified_atom_id,
            [added_element],
            [added_bond_order],
        )
    if change_function is replace_heavy_atom:
        inserted_atom_type = modification_path[0]
        replaced_atom_id = modification_path[1][0]
        resonance_structure_id = modification_path[1][1]
        return change_function(
            egc_in,
            replaced_atom_id,
            inserted_atom_type,
            resonance_structure_id=resonance_structure_id,
        )
    if change_function is change_valence_add_atoms:
        added_element = modification_path[0]
        modified_atom_id = modification_path[1]
        new_bond_order = modification_path[2]
        return change_function(egc_in, modified_atom_id, added_element, new_bond_order)
    if change_function is change_valence_remove_atoms:
        modified_atom_id = modification_path[1]
        removed_neighbors = modification_path[2][0]
        resonance_structure_id = modification_path[2][1]
        return change_function(
            egc_in,
            modified_atom_id,
            removed_neighbors,
            resonance_structure_id=resonance_structure_id,
        )
    raise Exception()


def inverse_mod_path(
    new_egc: ExtGraphCompound,
    old_egc: ExtGraphCompound,
    change_procedure,
    forward_path: list,
    linear_scaling_elementary_mutations=False,
    **other_kwargs,
):
    """
    Find modification path inverse to the forward_path.
    """
    if (change_procedure is change_bond_order) or (change_procedure is change_bond_order_valence):
        if linear_scaling_elementary_mutations:
            return [-forward_path[0], forward_path[1]]
        else:
            return [-forward_path[0]]
    if change_procedure is remove_heavy_atom:
        removed_atom = forward_path[-1][0]
        removed_elname = str_atom_corr(old_egc.chemgraph.hatoms[removed_atom].ncharge)

        neigh_id = old_egc.chemgraph.neighbors(removed_atom)[0]
        if removed_atom < neigh_id:
            neigh_id -= 1
        if not linear_scaling_elementary_mutations:
            neigh_id = new_egc.chemgraph.min_id_equivalent_atom_unchecked(neigh_id)
        return [removed_elname, neigh_id]
    if change_procedure is replace_heavy_atom:
        changed_atom_id = forward_path[-1][0]
        inserted_elname = str_atom_corr(old_egc.chemgraph.hatoms[changed_atom_id].ncharge)
        return [inserted_elname, changed_atom_id]
    if change_procedure is add_heavy_atom_chain:
        added_element = forward_path[0]
        added_atom_id = new_egc.chemgraph.nhatoms() - 1
        return [added_element, added_atom_id, None]
    if change_procedure is change_valence:
        changed_id = forward_path[0]
        if not linear_scaling_elementary_mutations:
            changed_id = new_egc.chemgraph.min_id_equivalent_atom_unchecked(changed_id)
        return [changed_id, None]
    if change_procedure is change_valence_add_atoms:
        added_element = forward_path[0]
        changed_id = forward_path[1]
        if not linear_scaling_elementary_mutations:
            changed_id = new_egc.chemgraph.min_id_equivalent_atom_unchecked(changed_id)
        return [
            added_element,
            changed_id,
            list(range(old_egc.num_heavy_atoms(), new_egc.num_heavy_atoms())),
        ]
    if change_procedure is change_valence_remove_atoms:
        modified_id = forward_path[1]
        new_modified_id = modified_id
        removed_ids = forward_path[2][0]
        for removed_id in removed_ids:
            if removed_id < modified_id:
                new_modified_id -= 1
        if not linear_scaling_elementary_mutations:
            new_modified_id = new_egc.chemgraph.min_id_equivalent_atom_unchecked(new_modified_id)
        bo = old_egc.chemgraph.bond_order(modified_id, removed_ids[0])
        return [forward_path[0], new_modified_id, bo]
    raise Exception()


# Special change functions required for changing bond orders while ignoring equivalence.
def get_second_changed_atom_res_struct_list(
    egc: ExtGraphCompound,
    first_changed_atom,
    possible_atom_choices,
    bond_order_change,
    max_fragment_num=None,
    forbidden_bonds=None,
    **other_kwargs,
):
    """
    Which atoms inside an ExtGraphCompound can have their bond with first_changed_atom altered.
    """
    # Note accounting for not_protonated is not needed.
    output = []
    for pos_atom_choice in possible_atom_choices:
        if pos_atom_choice == first_changed_atom:
            continue
        res_structs = bond_order_change_possible_resonance_structures(
            egc,
            first_changed_atom,
            pos_atom_choice,
            bond_order_change,
            max_fragment_num=max_fragment_num,
            forbidden_bonds=forbidden_bonds,
        )
        if res_structs is None:
            continue
        for res_struct in res_structs:
            output.append((pos_atom_choice, res_struct))
    return output


def choose_bond_change_parameters_linear_scaling(egc, possibilities, choices=None, **other_kwargs):
    # possibilities is structured as dictionnary of "bond order change" : list of potential atoms.
    # First choose the bond order change:
    (
        bond_order_change,
        possible_atom_choices,
        log_choice_prob,
    ) = random_choice_from_dict(possibilities, choices=choices)
    first_changed_atom = random.choice(possible_atom_choices)
    log_choice_prob -= llenlog(possible_atom_choices)
    log_choice_prob += log_atom_multiplicity_in_list(
        egc, first_changed_atom, possible_atom_choices, **other_kwargs
    )
    possible_second_changed_atom_res_struct_list = get_second_changed_atom_res_struct_list(
        egc,
        first_changed_atom,
        possible_atom_choices,
        bond_order_change,
        **other_kwargs,
    )
    second_atom_res_struct = random.choice(possible_second_changed_atom_res_struct_list)
    second_atom = second_atom_res_struct[0]
    res_struct = second_atom_res_struct[1]
    log_choice_prob -= llenlog(possible_second_changed_atom_res_struct_list)
    log_choice_prob += log_atom_multiplicity_in_list(
        egc,
        second_atom,
        possible_second_changed_atom_res_struct_list,
        special_atom_id=first_changed_atom,
        **other_kwargs,
    )
    mod_path = [bond_order_change, (first_changed_atom, second_atom, res_struct)]
    return mod_path, log_choice_prob


def get_valence_changed_atom_res_struct_list(
    egc: ExtGraphCompound,
    pres_val_atom_id,
    bond_order_change,
    max_fragment_num=None,
    forbidden_bonds=None,
    **other_kwargs,
):
    origin_point = None
    if bond_order_change < 0:
        origin_point = pres_val_atom_id

    found_without_sigma_bond_alteration = False

    possible_changed_val_atoms = polyvalent_hatom_indices(
        egc, bond_order_change, origin_point=origin_point
    )

    cg = egc.chemgraph
    hatoms = cg.hatoms
    pres_val_ha = hatoms[pres_val_atom_id]
    pres_val_ha_nc = pres_val_ha.ncharge

    output = []
    for changed_val_atom_id in possible_changed_val_atoms:
        if changed_val_atom_id == pres_val_atom_id:
            continue
        changed_val_atom = hatoms[changed_val_atom_id]
        changed_val_atom_nc = changed_val_atom.ncharge
        if bond_order_change > 0:
            if connection_forbidden(
                pres_val_ha_nc,
                changed_val_atom_nc,
                forbidden_bonds,
            ):
                continue
            are_neighbors = cg.are_neighbors(pres_val_atom_id, changed_val_atom_id)
            if found_without_sigma_bond_alteration and are_neighbors:
                continue

        resonance_struct_ids = cg.possible_res_struct_ids(changed_val_atom_id)
        if bond_order_change < 0:
            full_disconnections = []

        for resonance_struct_id in resonance_struct_ids:
            cur_mod_valence, valence_option_id = cg.valence_woption(
                changed_val_atom_id, resonance_structure_id=resonance_struct_id
            )
            if (
                next_valence(
                    changed_val_atom,
                    np.sign(bond_order_change),
                    valence_option_id=valence_option_id,
                )
                != cur_mod_valence + bond_order_change
            ):
                continue
            cur_bo = cg.bond_order(
                changed_val_atom_id,
                pres_val_atom_id,
                resonance_structure_id=resonance_struct_id,
            )
            if bond_order_change < 0:
                if cur_bo < -bond_order_change:
                    continue
                if cur_bo == -bond_order_change:
                    if breaking_bond_obeys_num_fragments(
                        egc,
                        changed_val_atom_id,
                        pres_val_atom_id,
                        max_fragment_num=max_fragment_num,
                    ):
                        cur_full_disconnection = True
                    else:
                        continue
                else:
                    cur_full_disconnection = False
                if cur_full_disconnection in full_disconnections:
                    continue
                else:
                    if not cur_full_disconnection:
                        if found_without_sigma_bond_alteration:
                            continue
                        found_without_sigma_bond_alteration = True
                    full_disconnections.append(cur_full_disconnection)
            else:
                if cur_bo + bond_order_change > max_bo_ncharges(
                    changed_val_atom_nc, pres_val_ha_nc
                ):
                    continue
                if not found_without_sigma_bond_alteration:
                    found_without_sigma_bond_alteration = are_neighbors
            output.append((changed_val_atom_id, resonance_struct_id))
            if bond_order_change > 0:  # we only need to find one way to create a connection
                break
    return output


def bond_valence_change_params_preserve_sigma_bonds(
    egc: ExtGraphCompound,
    bond_order_change: int,
    pres_val_atom: int,
    change_val_atom: int,
    change_val_res_struct: int,
):
    cg = egc.chemgraph
    if not cg.are_neighbors(change_val_atom, pres_val_atom):
        return False
    if bond_order_change > 0:
        return True
    changed_bo = cg.bond_order(
        pres_val_atom, change_val_atom, resonance_structure_id=change_val_res_struct
    )
    return changed_bo != -bond_order_change


def choose_bond_valence_change_parameters_linear_scaling(
    egc: ExtGraphCompound, possibilities, choices=None, **other_kwargs
):
    (
        bond_order_change,
        pres_val_change_atom_choices,
        log_choice_prob,
    ) = random_choice_from_dict(possibilities, choices=choices)
    pres_val_atom = random.choice(pres_val_change_atom_choices)
    log_choice_prob -= llenlog(pres_val_change_atom_choices)
    log_choice_prob += log_atom_multiplicity_in_list(
        egc, pres_val_atom, pres_val_change_atom_choices
    )
    change_val_atom_res_struct_list = get_valence_changed_atom_res_struct_list(
        egc,
        pres_val_atom,
        bond_order_change,
        **other_kwargs,
    )
    change_val_atom_res_struct = random.choice(change_val_atom_res_struct_list)
    change_val_atom = change_val_atom_res_struct[0]
    change_val_res_struct = change_val_atom_res_struct[1]

    log_choice_prob -= llenlog(change_val_atom_res_struct_list)
    if not bond_valence_change_params_preserve_sigma_bonds(
        egc, bond_order_change, pres_val_atom, change_val_atom, change_val_res_struct
    ):
        log_choice_prob += log_atom_multiplicity_in_list(
            egc,
            change_val_atom,
            change_val_atom_res_struct_list,
            special_atom_id=pres_val_atom,
            **other_kwargs,
        )

    mod_path = [
        bond_order_change,
        (change_val_atom, pres_val_atom, change_val_res_struct),
    ]
    return mod_path, log_choice_prob


special_bond_change_functions = {
    change_bond_order: choose_bond_change_parameters_linear_scaling,
    change_bond_order_valence: choose_bond_valence_change_parameters_linear_scaling,
}


def inv_prob_bond_change_parameters_linear_scaling(
    new_egc: ExtGraphCompound,
    inv_poss_dict: dict or list,
    inv_mod_path: list,
    **other_kwargs,
):
    inv_bo_change = inv_mod_path[0]
    first_changed_atom = inv_mod_path[1][0]
    second_changed_atom = inv_mod_path[1][1]
    possible_atom_choices, log_choice_prob = random_choice_from_dict(
        inv_poss_dict, get_probability_of=inv_bo_change
    )
    log_choice_prob -= llenlog(possible_atom_choices)
    log_choice_prob += log_atom_multiplicity_in_list(
        new_egc, first_changed_atom, possible_atom_choices, **other_kwargs
    )
    second_atom_res_struct_choices = get_second_changed_atom_res_struct_list(
        new_egc,
        first_changed_atom,
        possible_atom_choices,
        inv_bo_change,
        **other_kwargs,
    )
    log_choice_prob -= llenlog(second_atom_res_struct_choices)
    log_choice_prob += log_atom_multiplicity_in_list(
        new_egc,
        second_changed_atom,
        second_atom_res_struct_choices,
        special_atom_id=first_changed_atom,
        **other_kwargs,
    )

    return log_choice_prob


def inv_prob_bond_valence_change_parameters_linear_scaling(
    new_egc: ExtGraphCompound,
    inv_poss_dict: list or dict,
    inv_mod_path: list,
    **other_kwargs,
):
    inv_bo_change = inv_mod_path[0]
    pres_val_changed_atom = inv_mod_path[1][1]
    other_val_changed_atom = inv_mod_path[1][0]
    all_pres_val_choices, log_choice_prob = random_choice_from_dict(
        inv_poss_dict, get_probability_of=inv_bo_change
    )
    log_choice_prob -= llenlog(all_pres_val_choices)
    log_choice_prob += log_atom_multiplicity_in_list(
        new_egc, pres_val_changed_atom, all_pres_val_choices, **other_kwargs
    )
    changed_val_atoms = get_valence_changed_atom_res_struct_list(
        new_egc,
        pres_val_changed_atom,
        inv_bo_change,
        **other_kwargs,
    )
    log_choice_prob -= llenlog(changed_val_atoms)
    log_choice_prob += log_atom_multiplicity_in_list(
        new_egc,
        other_val_changed_atom,
        changed_val_atoms,
        special_atom_id=pres_val_changed_atom,
        **other_kwargs,
    )
    return log_choice_prob


special_inv_prob_calculators = {
    change_bond_order: inv_prob_bond_change_parameters_linear_scaling,
    change_bond_order_valence: inv_prob_bond_valence_change_parameters_linear_scaling,
}


def needed_special_bond_change_func(
    cur_change_procedure, linear_scaling_elementary_mutations=False
):
    return linear_scaling_elementary_mutations and is_bond_change(cur_change_procedure)


changed_atom_mod_path_level = {
    change_valence_add_atoms: 1,
    change_valence: 0,
    add_heavy_atom_chain: 1,
    remove_heavy_atom: 1,
    replace_heavy_atom: 1,
    change_valence_remove_atoms: 1,
}


def prob_atom_invariance_factor(
    egc: ExtGraphCompound,
    cur_procedure,
    possibilities: dict,
    mod_path: list,
    **other_kwargs,
):
    atom_id_mod_path_level = changed_atom_mod_path_level[cur_procedure]
    changed_atom_id = mod_path[atom_id_mod_path_level]
    if isinstance(changed_atom_id, tuple):
        changed_atom_id = changed_atom_id[0]
    cur_level_options = possibilities
    for mod_option in mod_path[:atom_id_mod_path_level]:
        cur_level_options = cur_level_options[mod_option]
    if isinstance(cur_level_options, dict):
        atom_list = list(cur_level_options.keys())
    else:
        atom_list = cur_level_options
    return log_atom_multiplicity_in_list(egc, changed_atom_id, atom_list, **other_kwargs)


def random_modification_path_choice(
    egc: ExtGraphCompound,
    possibilities: dict,
    cur_change_procedure,
    choices=None,
    get_probability_of=None,
    linear_scaling_elementary_mutations=False,
    **other_kwargs,
):
    special_bond_change_func = needed_special_bond_change_func(
        cur_change_procedure,
        linear_scaling_elementary_mutations=linear_scaling_elementary_mutations,
    )
    if get_probability_of is None:
        if special_bond_change_func:
            mod_path, log_prob_mod_path = special_bond_change_functions[cur_change_procedure](
                egc,
                possibilities,
                choices=choices,
                **other_kwargs,
            )
        else:
            mod_path, log_prob_mod_path = random_choice_from_nested_dict(
                possibilities, choices=choices
            )
            if linear_scaling_elementary_mutations:
                log_prob_mod_path += prob_atom_invariance_factor(
                    egc, cur_change_procedure, possibilities, mod_path, **other_kwargs
                )
        return mod_path, log_prob_mod_path
    else:
        if special_bond_change_func:
            log_prob_mod_path = special_inv_prob_calculators[cur_change_procedure](
                egc,
                possibilities,
                get_probability_of,
                **other_kwargs,
            )
        else:
            log_prob_mod_path = random_choice_from_nested_dict(
                possibilities, choices=choices, get_probability_of=get_probability_of
            )
            if linear_scaling_elementary_mutations:
                log_prob_mod_path += prob_atom_invariance_factor(
                    egc,
                    cur_change_procedure,
                    possibilities,
                    get_probability_of,
                    **other_kwargs,
                )
        return log_prob_mod_path


def randomized_change(
    tp: TrajectoryPoint,
    change_prob_dict=full_change_list,
    visited_tp_list: list or None = None,
    delete_chosen_mod_path: bool = False,
    linear_scaling_elementary_mutations: bool = False,
    **other_kwargs,
):
    """
    Randomly modify a TrajectoryPoint object.
    visited_tp_list : list of TrajectoryPoint objects for which data is available.
    linear_scaling_elementary_mutations : whether equivalence is accounted for during bond change moves (False is preferable for large systems).
    """

    init_possibilities_kwargs = {
        "change_prob_dict": change_prob_dict,
        "linear_scaling_elementary_mutations": linear_scaling_elementary_mutations,
        "exclude_equivalent": (not linear_scaling_elementary_mutations),
        **other_kwargs,
    }

    if delete_chosen_mod_path:
        if tp.modified_possibility_dict is None:
            tp.modified_possibility_dict = deepcopy(tp.possibilities(**init_possibilities_kwargs))
        full_possibility_dict = tp.modified_possibility_dict
        if len(full_possibility_dict) == 0:
            return None, None
    else:
        full_possibility_dict = tp.possibilities(**init_possibilities_kwargs)

    cur_change_procedure, possibilities, total_forward_prob = random_choice_from_dict(
        full_possibility_dict, change_prob_dict
    )
    special_bond_change_func = needed_special_bond_change_func(
        cur_change_procedure,
        linear_scaling_elementary_mutations=linear_scaling_elementary_mutations,
    )
    possibility_dict_label = change_possibility_label[cur_change_procedure]
    possibility_dict = lookup_or_none(other_kwargs, possibility_dict_label)

    old_egc = tp.egc

    modification_path, forward_prob = random_modification_path_choice(
        old_egc,
        possibilities,
        cur_change_procedure,
        choices=possibility_dict,
        linear_scaling_elementary_mutations=linear_scaling_elementary_mutations,
        **other_kwargs,
    )

    if delete_chosen_mod_path and (not special_bond_change_func):
        tp.delete_mod_path([cur_change_procedure] + modification_path)

    total_forward_prob += forward_prob

    new_egc = egc_change_func(old_egc, modification_path, cur_change_procedure, **other_kwargs)

    if new_egc is None:
        return None, None

    new_tp = TrajectoryPoint(egc=new_egc)
    if visited_tp_list is not None:
        if new_tp in visited_tp_list:
            tp_id = visited_tp_list.index(new_tp)
            visited_tp_list[tp_id].copy_extra_data_to(new_tp)
    new_tp.init_possibility_info(**init_possibilities_kwargs)
    # Calculate the chances of doing the inverse operation
    inv_proc = inverse_procedure[cur_change_procedure]
    inv_pos_label = change_possibility_label[inv_proc]
    inv_poss_dict = lookup_or_none(other_kwargs, inv_pos_label)
    inv_mod_path = inverse_mod_path(
        new_egc,
        old_egc,
        cur_change_procedure,
        modification_path,
        linear_scaling_elementary_mutations=linear_scaling_elementary_mutations,
        **other_kwargs,
    )

    try:
        inv_mod_path = inverse_mod_path(
            new_egc,
            old_egc,
            cur_change_procedure,
            modification_path,
            linear_scaling_elementary_mutations=linear_scaling_elementary_mutations,
            **other_kwargs,
        )
        inverse_possibilities, total_inverse_prob = random_choice_from_dict(
            new_tp.possibilities(),
            change_prob_dict,
            get_probability_of=inv_proc,
        )
        inverse_prob = random_modification_path_choice(
            new_egc,
            inverse_possibilities,
            inv_proc,
            choices=inv_poss_dict,
            get_probability_of=inv_mod_path,
            linear_scaling_elementary_mutations=linear_scaling_elementary_mutations,
            **other_kwargs,
        )
    except KeyError:
        print("NON-INVERTIBLE OPERATION")
        print(old_egc, cur_change_procedure)
        print(new_egc)
        raise InvalidChange

    total_inverse_prob += inverse_prob

    prob_balance = total_forward_prob - total_inverse_prob

    return new_tp, prob_balance
