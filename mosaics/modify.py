# TODO Perhaps add support for atoms with higher valences being added directly?
# TODO check that the currently commenting change_valence function exception is correct

# TODO even more randomness in fragment choice as an option? Is detailed balance viable?

# TODO change_bond_order does not seem to function properly with max_fragment_num None or not 1.

import numpy as np
from .ext_graph_compound import ExtGraphCompound
from .valence_treatment import (
    default_valence,
    avail_val_list,
    connection_forbidden,
    max_bo,
    next_valence,
    sorted_tuple,
)
from copy import deepcopy
from .misc_procedures import int_atom_checked
from .cross_coupling import *


def atom_equivalent_to_list_member(egc, atom_id, atom_id_list):
    if len(atom_id_list) == 0:
        return False
    are_tuples = not isinstance(atom_id_list, dict)
    if are_tuples:
        are_tuples = isinstance(atom_id_list[0], tuple)
    for other_atom_id in atom_id_list:
        if are_tuples:
            true_other_atom_id = other_atom_id[0]
        else:
            true_other_atom_id = other_atom_id
        if egc.chemgraph.atom_pair_equivalent(atom_id, true_other_atom_id):
            return True
    return False


# TODO was this done better in valence_bond_order_change_possibilities?
def atom_pair_equivalent_to_list_member(egc, atom_pair, atom_pair_list):
    for other_atom_pair in atom_pair_list:
        if egc.chemgraph.atom_sets_equivalent(atom_pair[:2], other_atom_pair[:2]):
            return True
    return False


def atom_replacement_possibilities(
    egc,
    inserted_atom,
    inserted_valence=None,
    replaced_atom=None,
    forbidden_bonds=None,
    exclude_equivalent=True,
    not_protonated=None,
    default_valences=None,
    **other_kwargs,
):

    possible_ids = []
    inserted_iac = int_atom_checked(inserted_atom)
    if replaced_atom is not None:
        replaced_iac = int_atom_checked(replaced_atom)
    if inserted_valence is None:
        if default_valences is None:
            inserted_valence = default_valence(inserted_iac)
        else:
            inserted_valence = default_valences[inserted_iac]
    if not_protonated is not None:
        cant_be_protonated = inserted_iac in not_protonated
    cg = egc.chemgraph
    for ha_id, ha in enumerate(cg.hatoms):
        if replaced_atom is not None:
            if ha.ncharge != replaced_iac:
                continue
        if inserted_iac == ha.ncharge:
            continue
        if forbidden_bonds is not None:
            cont = False
            for neigh in cg.neighbors(ha_id):
                if connection_forbidden(
                    egc.nuclear_charges[neigh], inserted_atom, forbidden_bonds
                ):
                    cont = True
                    break
            if cont:
                continue

        ha_default_valence = default_valence(ha.ncharge)

        val_diff = ha_default_valence - inserted_valence
        if val_diff <= ha.nhydrogens:
            if not_protonated is not None:
                if cant_be_protonated and (val_diff != ha.nhydrogens):
                    continue

            if ha.possible_valences is None:
                if ha_default_valence == ha.valence:
                    resonance_structure_id = None
                else:
                    continue
            else:
                if ha_default_valence in ha.possible_valences:
                    resonance_structure_id = cg.atom_valence_resonance_structure_id(
                        hatom_id=ha_id, valence=ha_default_valence
                    )
                else:
                    continue

            if exclude_equivalent:
                if atom_equivalent_to_list_member(egc, ha_id, possible_ids):
                    continue
            if ha.possible_valences is None:
                if ha_default_valence == ha.valence:
                    resonance_structure_id = None
                else:
                    continue
            else:
                if ha_default_valence in ha.possible_valences:
                    resonance_structure_id = cg.atom_valence_resonance_structure_id(
                        hatom_id=ha_id, valence=ha_default_valence
                    )
                else:
                    continue
            possible_ids.append((ha_id, resonance_structure_id))
    return possible_ids


def gen_atom_removal_possible_hnums(added_bond_orders, default_valence):
    possible_hnums = []
    for abo in added_bond_orders:
        hnum = default_valence - abo
        if hnum >= 0:
            possible_hnums.append(hnum)
    return possible_hnums


def atom_removal_possibilities(
    egc,
    deleted_atom="C",
    exclude_equivalent=True,
    nhatoms_range=None,
    not_protonated=None,
    added_bond_orders=[1],
    default_valences=None,
    atom_removal_possible_hnums=None,
    **other_kwargs,
):
    if nhatoms_range is not None:
        if egc.num_heavy_atoms() <= nhatoms_range[0]:
            return []
    possible_ids = []
    deleted_iac = int_atom_checked(deleted_atom)
    if default_valences is not None:
        deleted_default_valence = default_valences[deleted_iac]
    else:
        deleted_default_valence = default_valence(deleted_iac)

    if atom_removal_possible_hnums is None:
        possible_hnums = gen_atom_removal_possible_hnums(
            added_bond_orders, deleted_default_valence
        )
    else:
        possible_hnums = atom_removal_possible_hnums[deleted_iac]

    cg = egc.chemgraph
    hatoms = cg.hatoms

    for ha_id, ha in enumerate(hatoms):
        if (ha.ncharge != deleted_iac) or (ha.nhydrogens not in possible_hnums):
            continue
        neighs = cg.neighbors(ha_id)
        if len(neighs) != 1:
            continue
        if not_protonated is not None:
            if hatoms[neighs[0]].ncharge in not_protonated:
                continue
        if ha.possible_valences is None:
            if ha.valence != deleted_default_valence:
                continue
            resonance_structure_id = None
        else:
            if deleted_default_valence in ha.possible_valences:
                resonance_structure_id = cg.atom_valence_resonance_structure_id(
                    hatom_id=ha_id, valence=deleted_default_valence
                )
            else:
                continue
        if exclude_equivalent:
            if atom_equivalent_to_list_member(egc, ha_id, possible_ids):
                continue
        possible_ids.append((ha_id, resonance_structure_id))
    return possible_ids


def chain_addition_possibilities(
    egc,
    chain_starting_element=None,
    forbidden_bonds=None,
    exclude_equivalent=True,
    nhatoms_range=None,
    not_protonated=None,
    added_bond_orders=[1],
    avail_added_bond_orders=None,
    chain_addition_tuple_possibilities=False,
    **other_kwargs,
):
    if chain_addition_tuple_possibilities:
        possible_ids = []
    else:
        possible_ids = {}

    if nhatoms_range is not None:
        if egc.num_heavy_atoms() >= nhatoms_range[1]:
            return possible_ids
    chain_starting_ncharge = int_atom_checked(chain_starting_element)

    if avail_added_bond_orders is None:
        avail_added_bond_order = available_added_atom_bos(
            chain_starting_ncharge, added_bond_orders, not_protonated=not_protonated
        )
    else:
        avail_added_bond_order = avail_added_bond_orders[chain_starting_ncharge]

    if len(avail_added_bond_order) == 0:
        return possible_ids

    min_avail_added_bond_order = min(avail_added_bond_order)

    for ha_id, ha in enumerate(egc.chemgraph.hatoms):
        if (ha.nhydrogens >= min_avail_added_bond_order) and (
            not connection_forbidden(
                ha.ncharge, chain_starting_ncharge, forbidden_bonds
            )
        ):
            if exclude_equivalent:
                if atom_equivalent_to_list_member(egc, ha_id, possible_ids):
                    continue
            for added_bond_order in avail_added_bond_order:
                if added_bond_order > max_bo(ha, chain_starting_ncharge):
                    continue
                if added_bond_order <= ha.nhydrogens:
                    if chain_addition_tuple_possibilities:
                        possible_ids.append((ha_id, added_bond_order))
                    else:
                        if ha_id in possible_ids:
                            possible_ids[ha_id].append(added_bond_order)
                        else:
                            possible_ids[ha_id] = [added_bond_order]
    return possible_ids


def bond_change_possibilities(
    egc,
    bond_order_change,
    forbidden_bonds=None,
    not_protonated=None,
    fragment_member_vector=None,
    max_fragment_num=None,
    exclude_equivalent=True,
    **other_kwargs,
):
    output = []
    cg = egc.chemgraph
    hatoms = cg.hatoms
    if bond_order_change == 0:
        return output
    for ha_id1, ha1 in enumerate(hatoms):
        nc1 = ha1.ncharge
        if not_protonated is not None:
            if nc1 in not_protonated:
                continue
        if bond_order_change > 0:
            if ha1.nhydrogens < bond_order_change:
                continue
        for ha_id2, ha2 in enumerate(hatoms[:ha_id1]):
            nc2 = ha2.ncharge
            if bond_order_change > 0:
                if ha2.nhydrogens < bond_order_change:
                    continue
            if not_protonated is not None:
                if nc2 in not_protonated:
                    continue
            bond_tuple = (ha_id1, ha_id2)
            if fragment_member_vector is not None:
                if fragment_member_vector[ha_id1] == fragment_member_vector[ha_id2]:
                    continue
            if connection_forbidden(nc1, nc2, forbidden_bonds):
                continue
            possible_bond_orders = cg.aa_all_bond_orders(*bond_tuple)
            max_bond_order = max(possible_bond_orders)
            if bond_order_change > 0:
                min_bond_order = min(possible_bond_orders)
                if min_bond_order + bond_order_change > max_bo(nc1, nc2):
                    continue
                if max_bond_order == min_bond_order:
                    possible_resonance_structures = [None]
                else:
                    possible_resonance_structures = [
                        cg.aa_all_bond_orders(*bond_tuple, unsorted=True).index(
                            min_bond_order
                        )
                    ]
            else:
                if max_bond_order < -bond_order_change:
                    continue
                # The results will be different if we apply the change to a resonance structure where the bond order equals -bond_order_change or not. The algorithm accounts for both options.
                unsorted_bond_orders = None
                possible_resonance_structures = []
                for pbo in possible_bond_orders:
                    if pbo == -bond_order_change:
                        if max_fragment_num is not None:
                            if cg.num_connected() == max_fragment_num:
                                if (
                                    cg.graph.edge_connectivity(
                                        source=ha_id1, target=ha_id2
                                    )
                                    == 1
                                ):
                                    continue
                    if pbo >= -bond_order_change:
                        if unsorted_bond_orders is None:
                            unsorted_bond_orders = cg.aa_all_bond_orders(
                                *bond_tuple, unsorted=True
                            )
                        possible_resonance_structures.append(
                            unsorted_bond_orders.index(pbo)
                        )
                        if pbo != -bond_order_change:
                            break

            if exclude_equivalent:
                if atom_pair_equivalent_to_list_member(egc, bond_tuple, output):
                    continue
                for poss_res_struct in possible_resonance_structures:
                    output.append((*bond_tuple, poss_res_struct))

    return output


def gen_val_change_pos_ncharges(possible_elements, not_protonated=None):
    output = []
    for pos_el in possible_elements:
        pos_ncharge = int_atom_checked(pos_el)

        cur_avail_val_list = avail_val_list(pos_ncharge)

        if isinstance(cur_avail_val_list, int):
            continue

        if not_protonated is not None:
            if pos_ncharge in not_protonated:
                continue
        output.append(pos_ncharge)
    return output


def valence_change_possibilities(
    egc,
    val_change_poss_ncharges=None,
    possible_elements=["C"],
    exclude_equivalent=True,
    not_protonated=None,
    **other_kwargs,
):

    if val_change_poss_ncharges is None:
        val_change_poss_ncharges = gen_val_change_pos_ncharges(
            possible_elements, not_protonated=not_protonated
        )

    cg = egc.chemgraph
    cg.init_resonance_structures()

    output = {}

    for ha_id, ha in enumerate(cg.hatoms):
        if ha.ncharge not in val_change_poss_ncharges:
            continue
        if exclude_equivalent:
            if atom_equivalent_to_list_member(egc, ha_id, output):
                continue
        if ha.possible_valences is None:
            min_init_val = ha.valence
            min_res_struct = None
            max_init_val = ha.valence
            max_res_struct = None
        else:
            min_init_val = min(ha.possible_valences)
            min_res_struct = cg.atom_valence_resonance_structure_id(
                hatom_id=ha_id, valence=min_init_val
            )
            max_init_val = max(ha.possible_valences)
            max_res_struct = cg.atom_valence_resonance_structure_id(
                hatom_id=ha_id, valence=max_init_val
            )

        cur_val_list = ha.avail_val_list()
        available_valences = []
        for val in cur_val_list:
            if val > min_init_val:
                available_valences.append((val, min_res_struct))
            if val < max_init_val:
                if max_init_val - val <= ha.nhydrogens:
                    available_valences.append((val, max_res_struct))
        if len(available_valences) != 0:
            output[ha_id] = available_valences
    return output


def available_added_atom_bos(added_element, added_bond_orders, not_protonated=None):
    max_added_valence = default_valence(added_element)
    if not_protonated is not None:
        if int_atom_checked(added_element) in not_protonated:
            if max_added_valence in added_bond_orders:
                return [max_added_valence]
            else:
                return []
    output = []
    for abo in added_bond_orders:
        if abo <= max_added_valence:
            output.append(abo)
    return output


def gen_val_change_add_atom_pos_ncharges(
    possible_elements, chain_starting_element, forbidden_bonds=None
):
    val_change_pos_changes = gen_val_change_pos_ncharges(
        possible_elements, not_protonated=None
    )
    if forbidden_bonds is None:
        return val_change_pos_changes
    else:
        output = []
        for ncharge in val_change_pos_changes:
            if not connection_forbidden(
                ncharge, chain_starting_element, forbidden_bonds
            ):
                output.append(ncharge)
        return output


def valence_change_add_atoms_possibilities(
    egc,
    chain_starting_element,
    forbidden_bonds=None,
    exclude_equivalent=True,
    nhatoms_range=None,
    added_bond_orders_val_change=[1, 2],
    not_protonated=None,
    avail_added_bond_orders_val_change=None,
    val_change_add_atom_poss_ncharges=None,
    possible_elements=None,
    **other_kwargs,
):
    possibilities = {}
    if val_change_add_atom_poss_ncharges is None:
        val_change_poss_ncharges = gen_val_change_add_atom_pos_ncharges(
            possible_elements, chain_starting_element, forbidden_bonds=forbidden_bonds
        )
    else:
        val_change_poss_ncharges = val_change_add_atom_poss_ncharges[
            chain_starting_element
        ]

    if len(val_change_poss_ncharges) == 0:
        return possibilities

    if avail_added_bond_orders_val_change is None:
        avail_bond_orders = available_added_atom_bos(
            chain_starting_element,
            added_bond_orders_val_change,
            not_protonated=not_protonated,
        )
    else:
        avail_bond_orders = avail_added_bond_orders_val_change[chain_starting_element]

    if nhatoms_range is not None:
        max_added_nhatoms = nhatoms_range[1] - egc.num_heavy_atoms()
        if max_added_nhatoms < 0:
            raise Exception

    for ha_id, ha in enumerate(egc.chemgraph.hatoms):
        if ha.ncharge not in val_change_poss_ncharges:
            continue
        if exclude_equivalent:
            if atom_equivalent_to_list_member(egc, ha_id, possibilities):
                continue
        if ha.possible_valences is None:
            cur_valence = ha.valence
            valence_option = None
        else:
            cur_valence = min(ha.possible_valences)
            valence_option = ha.possible_valences.index(cur_valence)

        new_valence = next_valence(ha, valence_option_id=valence_option)
        if new_valence is None:
            continue
        val_diff = new_valence - cur_valence
        for added_bond_order in avail_bond_orders:
            if val_diff % added_bond_order != 0:
                continue
            added_nhatoms = val_diff // added_bond_order
            if (nhatoms_range is not None) and (added_nhatoms > max_added_nhatoms):
                continue
            if ha_id in possibilities:
                possibilities[ha_id].append(added_bond_order)
            else:
                possibilities[ha_id] = [added_bond_order]
    return possibilities


# ADD RESONANCE STRUCTURE INVARIANCE. SCROLL THROUGH THEM?
def valence_change_remove_atoms_possibilities(
    egc,
    removed_atom_type,
    possible_elements=["C"],
    exclude_equivalent=True,
    nhatoms_range=None,
    added_bond_orders_val_change=[1, 2],
    avail_added_bond_orders_val_change=None,
    val_change_add_atom_poss_ncharges=None,
    forbidden_bonds=None,
    not_protonated=None,
    default_valences=None,
    **other_kwargs,
):

    if nhatoms_range is not None:
        max_removed_nhatoms = egc.num_heavy_atoms() - nhatoms_range[0]
        if max_removed_nhatoms < 0:
            raise Exception()

    possibilities = {}
    if val_change_add_atom_poss_ncharges is None:
        val_change_poss_ncharges = gen_val_change_add_atom_pos_ncharges(
            possible_elements, removed_atom_type, forbidden_bonds=forbidden_bonds
        )
    else:
        val_change_poss_ncharges = val_change_add_atom_poss_ncharges[removed_atom_type]

    if len(val_change_poss_ncharges) == 0:
        return possibilities

    if avail_added_bond_orders_val_change is None:
        avail_bond_orders = available_added_atom_bos(
            removed_atom_type,
            added_bond_orders_val_change,
            not_protonated=not_protonated,
        )
    else:
        avail_bond_orders = avail_added_bond_orders_val_change[removed_atom_type]

    removed_atom_ncharge = int_atom_checked(removed_atom_type)

    if default_valences is None:
        default_removed_valence = default_valence(removed_atom_ncharge)
    else:
        default_removed_valence = default_valences[removed_atom_ncharge]

    cg = egc.chemgraph

    for ha_id, ha in enumerate(egc.chemgraph.hatoms):
        if ha.ncharge not in val_change_poss_ncharges:
            continue
        if exclude_equivalent:
            if atom_equivalent_to_list_member(egc, ha_id, possibilities):
                continue

        res_reg_id = cg.single_atom_resonance_structure(ha_id)
        if res_reg_id is None:
            res_struct_ids = [None]
        else:
            res_struct_ids = list(
                range(len(cg.resonance_structure_valence_vals[res_reg_id]))
            )

        saved_all_bond_orders = {}
        for neigh in cg.neighbors(ha_id):
            saved_all_bond_orders[neigh] = cg.aa_all_bond_orders(
                ha_id, neigh, unsorted=True
            )

        found_options = []

        for res_struct_id in res_struct_ids:
            if res_reg_id is None:
                val_opt = None
            else:
                val_opt = cg.resonance_structure_valence_vals[res_reg_id][res_struct_id]
            if ha.possible_valences is None:
                cur_valence = ha.valence
            else:
                cur_valence = ha.possible_valences[val_opt]
            new_valence = next_valence(ha, int_step=-1, valence_option_id=val_opt)
            if new_valence is None:
                continue

            val_diff = cur_valence - new_valence

            for added_bond_order in avail_bond_orders:
                if added_bond_order in found_options:
                    continue
                if val_diff % added_bond_order != 0:
                    continue
                removed_nhatoms = val_diff // added_bond_order
                if (nhatoms_range is not None) and (
                    removed_nhatoms > max_removed_nhatoms
                ):
                    continue
                removed_hatoms = []
                for neigh in cg.neighbors(ha_id):
                    neigh_ha = cg.hatoms[neigh]
                    if neigh_ha.ncharge != removed_atom_ncharge:
                        continue
                    if cg.num_neighbors(neigh) != 1:
                        continue
                    if neigh_ha.possible_valences is None:
                        neigh_valence = neigh_ha.valence
                    else:
                        neigh_valence = neigh_ha.possible_valences[val_opt]
                    if neigh_valence != default_removed_valence:
                        continue
                    bos = saved_all_bond_orders[neigh]
                    if (len(bos) == 1) or (res_struct_id is None):
                        bo = bos[0]
                    else:
                        bo = bos[res_struct_id]
                    if bo != added_bond_order:
                        continue
                    removed_hatoms.append(neigh)
                    if removed_nhatoms == len(removed_hatoms):
                        break
                if removed_nhatoms != len(removed_hatoms):
                    continue
                found_options.append(added_bond_order)
                poss_tuple = (tuple(removed_hatoms), res_struct_id)
                if ha_id in possibilities:
                    possibilities[ha_id].append(poss_tuple)
                else:
                    possibilities[ha_id] = [poss_tuple]
            if len(avail_bond_orders) == len(found_options):
                break
    return possibilities


# TODO add option for having opportunities as tuples vs dictionary? (PERHAPS NOT RELEVANT WITHOUT 4-order bonds)
def valence_bond_change_possibilities(
    egc,
    bond_order_change,
    forbidden_bonds=None,
    not_protonated=None,
    max_fragment_num=None,
    exclude_equivalent=True,
    **other_kwargs,
):
    # exclude_equivalent used to be a toggleable option here too. Perhaps should be deprecated everywhere.
    cg = egc.chemgraph
    hatoms = cg.hatoms
    output = []
    if bond_order_change == 0:
        return output

    altered_sigma_bond_class_tuples = []
    altered_hydrogen_number_classes = []

    for mod_val_ha_id, mod_val_ha in enumerate(hatoms):
        mod_val_nc = mod_val_ha.ncharge
        if not mod_val_ha.is_polyvalent():
            continue

        resonance_structure_region = cg.single_atom_resonance_structure(mod_val_ha_id)
        if resonance_structure_region is None:
            resonance_struct_ids = [None]
        else:
            res_struct_valence_vals = cg.resonance_structure_valence_vals[
                resonance_structure_region
            ]
            resonance_struct_ids = range(len(res_struct_valence_vals))
            res_struct_added_bos = cg.resonance_structure_orders[
                resonance_structure_region
            ]

        for other_ha_id, other_ha in enumerate(hatoms):
            if other_ha_id == mod_val_ha_id:
                continue

            other_nc = other_ha.ncharge

            if bond_order_change > 0:
                if connection_forbidden(mod_val_nc, other_nc, forbidden_bonds):
                    continue
                if hatoms[other_ha_id].nhydrogens < bond_order_change:
                    continue
            else:
                if not_protonated is not None:
                    if other_nc in not_protonated:
                        continue

            bond_tuple = (mod_val_ha_id, other_ha_id)

            st = sorted_tuple(mod_val_ha_id, other_ha_id)

            for resonance_struct_id in resonance_struct_ids:
                if resonance_struct_id is not None:
                    cur_res_struct_added_bos = res_struct_added_bos[resonance_struct_id]
                if (resonance_struct_id is None) or (
                    mod_val_ha.possible_valences is None
                ):
                    valence_option_id = None
                    cur_mod_valence = mod_val_ha.valence
                else:
                    valence_option_id = res_struct_valence_vals[resonance_struct_id]
                    cur_mod_valence = mod_val_ha.possible_valences[valence_option_id]

                if (
                    next_valence(
                        mod_val_ha,
                        np.sign(bond_order_change),
                        valence_option_id=valence_option_id,
                    )
                    != cur_mod_valence + bond_order_change
                ):
                    continue

                cur_bo = cg.bond_order(mod_val_ha_id, other_ha_id)
                if (resonance_struct_id is not None) and (cur_bo != 0):
                    cur_bo = 1
                    if st in cur_res_struct_added_bos:
                        cur_bo += cur_res_struct_added_bos[st]

                changed_sigma_bond = False
                hydrogenated_atom_class = cg.equivalence_class((other_ha_id,))
                if bond_order_change > 0:
                    if cur_bo + bond_order_change > max_bo(mod_val_nc, other_nc):
                        continue
                    if cur_bo == 0:  # we are creating a new bond.
                        changed_sigma_bond = True
                else:
                    if cur_bo < -bond_order_change:
                        continue
                    if cur_bo == -bond_order_change:
                        if (cg.num_connected() == max_fragment_num) and (
                            cg.graph.edge_connectivity(
                                source=mod_val_ha_id, target=other_ha_id
                            )
                            == 1
                        ):
                            continue
                        changed_sigma_bond = True
                if changed_sigma_bond:
                    change_identifier = (
                        hydrogenated_atom_class,
                        cg.equivalence_class(st),
                    )
                    change_list = altered_sigma_bond_class_tuples
                else:
                    change_identifier = hydrogenated_atom_class
                    change_list = altered_hydrogen_number_classes

                if change_identifier in change_list:
                    continue
                change_list.append(change_identifier)
                output.append((*bond_tuple, resonance_struct_id))

    return output


def val_min_checked_egc(cg):
    if cg.attempt_minimize_valences():
        return ExtGraphCompound(chemgraph=cg)
    else:
        return None


def add_heavy_atom_chain(
    egc, modified_atom_id, new_chain_atoms, chain_bond_orders=None
):
    new_chemgraph = deepcopy(egc.chemgraph)
    new_chemgraph.add_heavy_atom_chain(
        modified_atom_id, new_chain_atoms, chain_bond_orders=chain_bond_orders
    )
    return ExtGraphCompound(chemgraph=new_chemgraph)


def replace_heavy_atom(
    egc,
    replaced_atom_id,
    inserted_atom,
    inserted_valence=None,
    resonance_structure_id=None,
):
    new_chemgraph = deepcopy(egc.chemgraph)
    new_chemgraph.replace_heavy_atom(
        replaced_atom_id,
        inserted_atom,
        inserted_valence=inserted_valence,
        resonance_structure_id=resonance_structure_id,
    )
    return val_min_checked_egc(new_chemgraph)


def remove_heavy_atom(egc, removed_atom_id, resonance_structure_id=None):
    new_chemgraph = deepcopy(egc.chemgraph)
    new_chemgraph.remove_heavy_atom(
        removed_atom_id, resonance_structure_id=resonance_structure_id
    )
    return ExtGraphCompound(chemgraph=new_chemgraph)


def change_bond_order(
    egc, atom_id1, atom_id2, bond_order_change, resonance_structure_id=0
):
    new_chemgraph = deepcopy(egc.chemgraph)
    new_chemgraph.change_bond_order(
        atom_id1,
        atom_id2,
        bond_order_change,
        resonance_structure_id=resonance_structure_id,
    )
    return val_min_checked_egc(new_chemgraph)


def change_valence(egc, modified_atom_id, new_valence, resonance_structure_id=None):
    new_chemgraph = deepcopy(egc.chemgraph)
    if resonance_structure_id is not None:
        resonance_structure_region = new_chemgraph.single_atom_resonance_structure(
            modified_atom_id
        )
        new_chemgraph.adjust_resonance_valences(
            resonance_structure_region, resonance_structure_id
        )
    new_chemgraph.change_valence(modified_atom_id, new_valence)
    return val_min_checked_egc(new_chemgraph)


def change_valence_add_atoms(egc, modified_atom_id, new_atom_element, new_bo):
    new_chemgraph = deepcopy(egc.chemgraph)

    mod_hatom = new_chemgraph.hatoms[modified_atom_id]

    if mod_hatom.possible_valences is not None:
        min_valence = min(mod_hatom.possible_valences)
        min_val_poss = mod_hatom.possible_valences.index(min_valence)
        new_chemgraph.adjust_resonance_valences_atom(
            modified_atom_id, valence_option_id=min_val_poss
        )

    new_atom_charge = int_atom_checked(new_atom_element)

    new_mod_valence_val = next_valence(mod_hatom)
    val_diff = new_mod_valence_val - mod_hatom.valence
    if val_diff % new_bo != 0:
        raise Exception()
    num_added = val_diff // new_bo
    new_chemgraph.change_valence(modified_atom_id, new_mod_valence_val)
    for _ in range(num_added):
        new_chemgraph.add_heavy_atom_chain(
            modified_atom_id, [new_atom_charge], chain_bond_orders=[new_bo]
        )
    return val_min_checked_egc(new_chemgraph)


def change_valence_remove_atoms(
    egc, modified_atom_id, removed_neighbors, resonance_structure_id=None
):
    new_chemgraph = deepcopy(egc.chemgraph)

    new_chemgraph.adjust_resonance_valences_atom(
        modified_atom_id, resonance_structure_id=resonance_structure_id
    )

    mod_hatom = new_chemgraph.hatoms[modified_atom_id]
    new_mod_valence_val = next_valence(mod_hatom, -1)
    val_diff = mod_hatom.valence - new_mod_valence_val

    id_shift = 0
    running_bond = None
    for neigh in removed_neighbors:
        if neigh < modified_atom_id:
            id_shift -= 1
        cur_bond = new_chemgraph.bond_order(neigh, modified_atom_id)
        if running_bond is None:
            running_bond = cur_bond
            if running_bond == 0:
                raise Exception()
        else:
            if cur_bond != running_bond:
                raise Exception()
        val_diff -= cur_bond
        if val_diff == 0:
            break
    if val_diff != 0:
        raise Exception()
    new_chemgraph.remove_heavy_atoms(removed_neighbors)
    new_modified_atom_id = modified_atom_id + id_shift
    new_chemgraph.change_valence(new_modified_atom_id, new_mod_valence_val)
    return val_min_checked_egc(new_chemgraph)


def change_bond_order_valence(
    egc,
    val_changed_atom_id,
    other_atom_id,
    bond_order_change,
    resonance_structure_id=None,
):

    new_chemgraph = deepcopy(egc.chemgraph)

    new_chemgraph.adjust_resonance_valences_atom(
        val_changed_atom_id, resonance_structure_id=resonance_structure_id
    )

    new_valence = new_chemgraph.hatoms[val_changed_atom_id].valence + bond_order_change

    if bond_order_change > 0:
        new_chemgraph.change_valence(val_changed_atom_id, new_valence)

    new_chemgraph.change_bond_order(
        val_changed_atom_id, other_atom_id, bond_order_change
    )

    if bond_order_change < 0:
        new_chemgraph.change_valence(val_changed_atom_id, new_valence)

    if not new_chemgraph.hatoms[val_changed_atom_id].valence_reasonable():
        raise Exception()

    return val_min_checked_egc(new_chemgraph)


def no_forbidden_bonds(egc: ExtGraphCompound, forbidden_bonds: None or list = None):
    """
    Check that an ExtGraphCompound object has no covalent bonds whose nuclear charge tuple is inside forbidden_bonds.
    egc : checked ExtGraphCompound object
    forbidden_bonds : list of sorted nuclear charge tuples.
    """
    if forbidden_bonds is not None:
        cg = egc.chemgraph
        hatoms = cg.hatoms
        for bond_tuple in cg.bond_orders.keys():
            if connection_forbidden(
                hatoms[bond_tuple[0]].ncharge,
                hatoms[bond_tuple[1]].ncharge,
                forbidden_bonds=forbidden_bonds,
            ):
                return False
    return True


def egc_valid_wrt_change_params(
    egc,
    nhatoms_range=None,
    forbidden_bonds=None,
    possible_elements=None,
    not_protonated=None,
    **other_kwargs,
):
    """
    Check that an ExtGraphCompound object is a member of chemical subspace spanned by change params used throughout chemxpl.modify module.
    egc : ExtGraphCompound object
    nhatoms_range : range of possible numbers of heavy atoms
    forbidden_bonds : ordered tuples of nuclear charges corresponding to elements that are forbidden to have bonds.
    """
    if not no_forbidden_bonds(egc, forbidden_bonds=forbidden_bonds):
        return False
    if not_protonated is not None:
        for ha in egc.chemgraph.hatoms:
            if (ha.ncharge in not_protonated) and (ha.nhydrogens != 0):
                return False
    if nhatoms_range is not None:
        nhas = egc.chemgraph.nhatoms()
        if (nhas < nhatoms_range[0]) or (nhas > nhatoms_range[1]):
            return False
    if possible_elements is not None:
        possible_elements_nc = [int_atom_checked(pe) for pe in possible_elements]
        for ha in egc.chemgraph.hatoms:
            if ha.ncharge not in possible_elements_nc:
                return False
    return True
