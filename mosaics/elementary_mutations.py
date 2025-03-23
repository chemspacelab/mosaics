# TODO Perhaps add support for atoms with higher valences being added directly?
# TODO check that the currently commenting change_valence function exception is correct

# TODO even more randomness in fragment choice as an option? Is detailed balance viable?

# TODO change_bond_order does not seem to function properly with max_fragment_num None or not 1.

from copy import deepcopy

import numpy as np

from .chem_graph.heavy_atom import avail_val_list, default_valence, next_valence
from .chem_graph.resonance_structures import max_bo
from .ext_graph_compound import ExtGraphCompound, connection_forbidden
from .misc_procedures import int_atom_checked, sorted_tuple


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
    egc: ExtGraphCompound,
    inserted_atom: str,
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
        if val_diff > ha.nhydrogens:
            continue
        if not_protonated is not None:
            if cant_be_protonated and (val_diff != ha.nhydrogens):
                continue

        if not cg.default_valence_available(ha_id):
            continue

        if exclude_equivalent:
            if atom_equivalent_to_list_member(egc, ha_id, possible_ids):
                continue
        resonance_structure_id = cg.default_valence_resonance_structure_id(ha_id)
        possible_ids.append((ha_id, resonance_structure_id))
    return possible_ids


# TODO are valences properly accounted for here?
def gen_atom_removal_possible_hnums(added_bond_orders, default_valence):
    possible_hnums = []
    for abo in added_bond_orders:
        hnum = default_valence - abo
        if hnum >= 0:
            possible_hnums.append(hnum)
    return possible_hnums


def atom_removal_possibilities(
    egc: ExtGraphCompound,
    deleted_atom: str = "C",
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
        if not cg.default_valence_available(ha_id):
            continue
        if exclude_equivalent:
            if atom_equivalent_to_list_member(egc, ha_id, possible_ids):
                continue
        resonance_structure_id = cg.default_valence_resonance_structure_id(ha_id)
        possible_ids.append((ha_id, resonance_structure_id))
    return possible_ids


def hatoms_with_changeable_nhydrogens(
    egc: ExtGraphCompound,
    bond_order_change: int,
    not_protonated: list or None = None,
    origin_point: int or None = None,
):
    cg = egc.chemgraph
    if origin_point is None:
        ha_ids = range(cg.nhatoms())
    else:
        ha_ids = cg.neighbors(origin_point)
    output = []
    for ha_id in ha_ids:
        ha = cg.hatoms[ha_id]
        if bond_order_change > 0:
            if ha.nhydrogens < bond_order_change:
                continue
        else:
            if not_protonated is not None:
                if ha.ncharge in not_protonated:
                    continue
        output.append(ha_id)
    return output


def chain_addition_possibilities(
    egc: ExtGraphCompound,
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

    potentially_altered_ha_ids = hatoms_with_changeable_nhydrogens(egc, min_avail_added_bond_order)

    for ha_id in potentially_altered_ha_ids:
        ha = egc.chemgraph.hatoms[ha_id]
        if connection_forbidden(ha.ncharge, chain_starting_ncharge, forbidden_bonds):
            continue
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


def breaking_bond_obeys_num_fragments(
    egc: ExtGraphCompound, atom_id1: int, atom_id2: int, max_fragment_num=None
):
    """
    Whether breaking the bond between two heavy atoms results in violation on the number of fragments upper bond.
    """
    if max_fragment_num is None:
        return True
    assert isinstance(max_fragment_num, int)
    if egc.chemgraph.num_connected() < max_fragment_num:
        return True
    return egc.chemgraph.graph.edge_connectivity(source=atom_id1, target=atom_id2) != 1


def bond_order_change_possible_resonance_structures(
    egc: ExtGraphCompound,
    atom_id1: int,
    atom_id2: int,
    bond_order_change: int,
    max_fragment_num=None,
    forbidden_bonds=None,
):
    cg = egc.chemgraph
    cur_min_bo = egc.chemgraph.min_bond_order(atom_id1, atom_id2)
    cur_max_bo = egc.chemgraph.max_bond_order(atom_id1, atom_id2)
    if bond_order_change > 0:
        ncharge1 = cg.hatoms[atom_id1].ncharge
        ncharge2 = cg.hatoms[atom_id2].ncharge
        if connection_forbidden(ncharge1, ncharge2, forbidden_bonds=forbidden_bonds):
            return []
        if cur_min_bo + bond_order_change > max_bo(ncharge1, ncharge2):
            return []
        if cur_min_bo == cur_max_bo:
            return [None]
        else:
            return [cg.aa_all_bond_orders(atom_id1, atom_id2, unsorted=True).index(cur_min_bo)]
    if cur_max_bo < -bond_order_change:
        return []
    # The results will be different if we apply the change to a resonance structure where the bond order equals -bond_order_change or not. The algorithm accounts for both options.
    unsorted_bond_orders = None
    possible_resonance_structures = []
    for pbo in cg.aa_all_bond_orders(atom_id1, atom_id2):
        if pbo == -bond_order_change:
            if not breaking_bond_obeys_num_fragments(
                egc, atom_id1, atom_id2, max_fragment_num=max_fragment_num
            ):
                continue
        if pbo >= -bond_order_change:
            if unsorted_bond_orders is None:
                unsorted_bond_orders = cg.aa_all_bond_orders(atom_id1, atom_id2, unsorted=True)
            possible_resonance_structures.append(unsorted_bond_orders.index(pbo))
            if pbo != -bond_order_change:
                break
    return possible_resonance_structures


def bond_atom_change_possibilities(
    egc: ExtGraphCompound,
    bond_order_change: int,
    forbidden_bonds=None,
    not_protonated=None,
    max_fragment_num=None,
    **other_kwargs,
):
    """
    An alternative to bond_change_possibilities that allows linear scaling w.r.t. molecule size.
    """
    if bond_order_change < 0:
        used_not_protonated = not_protonated
    else:
        used_not_protonated = None

    cg = egc.chemgraph
    # Which atoms can be connected IF there is a suitable other atom.
    potentially_altered_atom_list = []
    for ha_id, ha in enumerate(cg.hatoms):
        if used_not_protonated is not None:
            if ha.ncharge in used_not_protonated:
                continue
        if bond_order_change < 0:
            found_possible_change = False
            for other_ha_id in cg.neighbors(ha_id):
                if used_not_protonated is not None:
                    if cg.hatoms[other_ha_id].ncharge in used_not_protonated:
                        continue
                if (
                    len(
                        bond_order_change_possible_resonance_structures(
                            egc,
                            ha_id,
                            other_ha_id,
                            bond_order_change,
                            max_fragment_num=max_fragment_num,
                        )
                    )
                    == 0
                ):
                    continue
                found_possible_change = True
                break
            if not found_possible_change:
                continue
        else:
            if ha.nhydrogens < bond_order_change:
                continue
        potentially_altered_atom_list.append(ha_id)
    # Check absence of problems due to forbidden_bonds or maximum bond order.
    if bond_order_change > 0:
        checked_atom_list_id = 0
        while checked_atom_list_id != len(potentially_altered_atom_list):
            checked_atom = potentially_altered_atom_list[checked_atom_list_id]
            no_connection_possibilities = True
            for other_checked_atom in potentially_altered_atom_list:
                if other_checked_atom == checked_atom:
                    continue
                if (
                    len(
                        bond_order_change_possible_resonance_structures(
                            egc,
                            checked_atom,
                            other_checked_atom,
                            bond_order_change,
                            max_fragment_num=max_fragment_num,
                            forbidden_bonds=forbidden_bonds,
                        )
                    )
                    == 0
                ):
                    continue
                no_connection_possibilities = False
                break
            if no_connection_possibilities:
                del potentially_altered_atom_list[checked_atom_list_id]
            else:
                checked_atom_list_id += 1
    # If we can only alter one atom then we cannot choose an altered pair.
    if len(potentially_altered_atom_list) == 1:
        return []
    return potentially_altered_atom_list


def bond_change_possibilities(
    egc,
    bond_order_change,
    forbidden_bonds=None,
    not_protonated=None,
    max_fragment_num=None,
    exclude_equivalent=True,
    linear_scaling_elementary_mutations=False,
    **other_kwargs,
):
    if linear_scaling_elementary_mutations:
        return bond_atom_change_possibilities(
            egc,
            bond_order_change,
            forbidden_bonds=forbidden_bonds,
            not_protonated=not_protonated,
            max_fragment_num=max_fragment_num,
        )
    output = []
    if bond_order_change == 0:
        return output
    potentially_altered_first_atoms = hatoms_with_changeable_nhydrogens(
        egc, bond_order_change, not_protonated=not_protonated
    )
    for ha_id1 in potentially_altered_first_atoms:
        if bond_order_change > 0:
            potentially_altered_second_atoms = potentially_altered_first_atoms
        else:
            potentially_altered_second_atoms = hatoms_with_changeable_nhydrogens(
                egc,
                bond_order_change,
                not_protonated=not_protonated,
                origin_point=ha_id1,
            )
        for ha_id2 in potentially_altered_second_atoms:
            if ha_id2 >= ha_id1:
                break
            bond_tuple = (ha_id1, ha_id2)
            possible_resonance_structures = bond_order_change_possible_resonance_structures(
                egc,
                *bond_tuple,
                bond_order_change,
                max_fragment_num=max_fragment_num,
                forbidden_bonds=forbidden_bonds,
            )
            if len(possible_resonance_structures) == 0:
                continue
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
    val_change_pos_changes = gen_val_change_pos_ncharges(possible_elements, not_protonated=None)
    if forbidden_bonds is None:
        return val_change_pos_changes
    else:
        output = []
        for ncharge in val_change_pos_changes:
            if not connection_forbidden(ncharge, chain_starting_element, forbidden_bonds):
                output.append(ncharge)
        return output


def valence_change_add_atoms_possibilities(
    egc: ExtGraphCompound,
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
        val_change_poss_ncharges = val_change_add_atom_poss_ncharges[chain_starting_element]

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

    cg = egc.chemgraph

    for ha_id, ha in enumerate(cg.hatoms):
        if ha.ncharge not in val_change_poss_ncharges:
            continue
        if exclude_equivalent:
            if atom_equivalent_to_list_member(egc, ha_id, possibilities):
                continue
        cur_valence, valence_option = cg.min_valence_woption(ha_id)

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


# TODO ADD RESONANCE STRUCTURE INVARIANCE? SCROLL THROUGH THEM?
def valence_change_remove_atoms_possibilities(
    egc: ExtGraphCompound,
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

    for ha_id, ha in enumerate(cg.hatoms):
        if ha.ncharge not in val_change_poss_ncharges:
            continue
        if exclude_equivalent:
            if atom_equivalent_to_list_member(egc, ha_id, possibilities):
                continue

        res_struct_ids = cg.possible_res_struct_ids(ha_id)

        saved_all_bond_orders = {}
        for neigh in cg.neighbors(ha_id):
            saved_all_bond_orders[neigh] = cg.aa_all_bond_orders(ha_id, neigh, unsorted=True)

        found_options = []

        for res_struct_id in res_struct_ids:
            cur_valence, val_opt = cg.valence_woption(ha_id, resonance_structure_id=res_struct_id)
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
                if (nhatoms_range is not None) and (removed_nhatoms > max_removed_nhatoms):
                    continue
                removed_hatoms = []
                for neigh in cg.neighbors(ha_id):
                    neigh_ha = cg.hatoms[neigh]
                    if neigh_ha.ncharge != removed_atom_ncharge:
                        continue
                    if cg.num_neighbors(neigh) != 1:
                        continue
                    neigh_valence, _ = cg.valence_woption(
                        neigh, resonance_structure_id=res_struct_id
                    )
                    if neigh_valence != default_removed_valence:
                        continue
                    bo = cg.bond_order(ha_id, neigh, resonance_structure_id=res_struct_id)
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
            # TODO why?
            if len(avail_bond_orders) == len(found_options):
                break
    return possibilities


def polyvalent_hatom_indices(
    egc: ExtGraphCompound, bond_order_change: int, origin_point: int or None = None
):
    cg = egc.chemgraph

    if bond_order_change > 0:
        checked_hatom_ids = range(cg.nhatoms())
    else:
        checked_hatom_ids = sorted(cg.neighbors(origin_point))
    output = []
    for ha_id in checked_hatom_ids:
        ha = cg.hatoms[ha_id]
        if not ha.is_polyvalent():
            continue
        if bond_order_change > 0:
            min_valence, _ = cg.min_valence_woption(ha_id)
            if min_valence + bond_order_change > ha.max_valence():
                continue
        else:
            max_valence, _ = cg.max_valence_woption(ha_id)
            if max_valence + bond_order_change < ha.min_valence():
                continue
        output.append(ha_id)
    return output


def valence_bond_change_atom_possibilities(
    egc: ExtGraphCompound,
    bond_order_change: int,
    forbidden_bonds=None,
    not_protonated=None,
    max_fragment_num=None,
    **other_kwargs,
):
    """
    Find ids of atoms whose nhydrogens can be changed to accomodate for a valence-bond chemical graph change.
    """
    # TODO shares a lot with the "badly scaling" version, might be combinable.
    cg = egc.chemgraph
    hatoms = cg.hatoms

    if bond_order_change > 0:
        checked_polyvalent_hatoms = polyvalent_hatom_indices(egc, bond_order_change)

    checked_pres_val_hatoms = hatoms_with_changeable_nhydrogens(
        egc, bond_order_change, not_protonated=not_protonated
    )

    pres_valence_final_hatom_list = []

    for pres_val_ha_id in checked_pres_val_hatoms:
        pres_val_ha_nc = hatoms[pres_val_ha_id].ncharge
        # For each atom check that there is something else
        if bond_order_change < 0:
            checked_polyvalent_hatoms = polyvalent_hatom_indices(
                egc, bond_order_change, origin_point=pres_val_ha_id
            )
        found_valid_mod_val_atom = False
        for mod_val_ha_id in checked_polyvalent_hatoms:
            if mod_val_ha_id == pres_val_ha_id:
                continue
            mod_val_ha = hatoms[mod_val_ha_id]
            mod_val_ha_nc = mod_val_ha.ncharge
            if (bond_order_change > 0) and connection_forbidden(
                pres_val_ha_nc, mod_val_ha.ncharge, forbidden_bonds
            ):
                continue
            resonance_struct_ids = cg.possible_res_struct_ids(mod_val_ha_id)
            found_valid_res_struct = False
            for resonance_struct_id in resonance_struct_ids:
                cur_mod_valence, valence_option_id = cg.valence_woption(
                    mod_val_ha_id, resonance_structure_id=resonance_struct_id
                )
                next_valence_val = next_valence(
                    mod_val_ha,
                    np.sign(bond_order_change),
                    valence_option_id=valence_option_id,
                )
                if next_valence_val is None:
                    continue
                if next_valence_val != cur_mod_valence + bond_order_change:
                    continue

                cur_bo = cg.bond_order(
                    mod_val_ha_id,
                    pres_val_ha_id,
                    resonance_structure_id=resonance_struct_id,
                )
                if bond_order_change > 0:
                    if cur_bo + bond_order_change > max_bo(pres_val_ha_nc, mod_val_ha_nc):
                        continue
                else:
                    if cur_bo < -bond_order_change:
                        continue
                    if (cur_bo == -bond_order_change) and (
                        not breaking_bond_obeys_num_fragments(
                            egc,
                            mod_val_ha_id,
                            pres_val_ha_id,
                            max_fragment_num=max_fragment_num,
                        )
                    ):
                        continue
                found_valid_res_struct = True
                break
            if found_valid_res_struct:
                found_valid_mod_val_atom = True
                break
        if found_valid_mod_val_atom:
            pres_valence_final_hatom_list.append(pres_val_ha_id)
    return pres_valence_final_hatom_list


# TODO add option for having opportunities as tuples vs dictionary? (PERHAPS NOT RELEVANT WITHOUT 4-order bonds)
def valence_bond_change_possibilities(
    egc: ExtGraphCompound,
    bond_order_change,
    forbidden_bonds=None,
    not_protonated=None,
    max_fragment_num=None,
    linear_scaling_elementary_mutations=False,
    **other_kwargs,
):
    # exclude_equivalent used to be a toggleable option here too.
    # The code assumes it to be True, with False being only used for linear_scaling_elementary_mutations anyway.
    if linear_scaling_elementary_mutations:
        return valence_bond_change_atom_possibilities(
            egc,
            bond_order_change,
            forbidden_bonds=forbidden_bonds,
            not_protonated=not_protonated,
            max_fragment_num=max_fragment_num,
        )
    cg = egc.chemgraph
    hatoms = cg.hatoms
    output = []
    if bond_order_change == 0:
        return output

    if bond_order_change > 0:
        possible_second_atoms = hatoms_with_changeable_nhydrogens(egc, bond_order_change)

    altered_sigma_bond_class_tuples = []
    altered_hydrogen_number_classes = []

    for mod_val_ha_id, mod_val_ha in enumerate(hatoms):
        mod_val_nc = mod_val_ha.ncharge
        if not mod_val_ha.is_polyvalent():
            continue
        resonance_struct_ids = cg.possible_res_struct_ids(mod_val_ha_id)
        if bond_order_change < 0:
            possible_second_atoms = hatoms_with_changeable_nhydrogens(
                egc,
                bond_order_change,
                origin_point=mod_val_ha_id,
                not_protonated=not_protonated,
            )
        for other_ha_id in possible_second_atoms:
            if other_ha_id == mod_val_ha_id:
                continue

            other_ha = hatoms[other_ha_id]

            other_nc = other_ha.ncharge

            if bond_order_change > 0:
                if connection_forbidden(mod_val_nc, other_nc, forbidden_bonds):
                    continue

            bond_tuple = (mod_val_ha_id, other_ha_id)

            st = sorted_tuple(mod_val_ha_id, other_ha_id)

            for resonance_struct_id in resonance_struct_ids:
                cur_mod_valence, valence_option_id = cg.valence_woption(
                    mod_val_ha_id, resonance_structure_id=resonance_struct_id
                )
                if (
                    next_valence(
                        mod_val_ha,
                        np.sign(bond_order_change),
                        valence_option_id=valence_option_id,
                    )
                    != cur_mod_valence + bond_order_change
                ):
                    continue

                cur_bo = cg.bond_order(
                    mod_val_ha_id,
                    other_ha_id,
                    resonance_structure_id=resonance_struct_id,
                )

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
                        if not breaking_bond_obeys_num_fragments(
                            egc,
                            mod_val_ha_id,
                            other_ha_id,
                            max_fragment_num=max_fragment_num,
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


def add_heavy_atom_chain(egc, modified_atom_id, new_chain_atoms, chain_bond_orders=None):
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
    new_chemgraph.remove_heavy_atom(removed_atom_id, resonance_structure_id=resonance_structure_id)

    return ExtGraphCompound(chemgraph=new_chemgraph)


def change_bond_order(egc, atom_id1, atom_id2, bond_order_change, resonance_structure_id=0):
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
        new_chemgraph.adjust_resonance_valences(resonance_structure_region, resonance_structure_id)
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

    new_chemgraph.change_bond_order(val_changed_atom_id, other_atom_id, bond_order_change)

    if bond_order_change < 0:
        new_chemgraph.change_valence(val_changed_atom_id, new_valence)

    new_chemgraph.changed()
    assert new_chemgraph.hatoms[val_changed_atom_id].valence_reasonable()

    return val_min_checked_egc(new_chemgraph)
