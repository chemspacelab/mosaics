# TODO Currently, assigning equivalence classes to all nodes scales as O(nhatoms**2). I think that smarter coding can bring that down to O(log(nhatoms)), but for now we have more important methodological problems.

# TODO Should be possible to optimize further by comparing HeavyAtom neighborhood representations instead of colors (should decrease the number of canonical permutation evaluations during simulation).

import copy
import random

import numpy as np
from igraph.operators import disjoint_union

from ..misc_procedures import int_atom_checked, sorted_by_membership, sorted_tuple
from .base_chem_graph import BaseChemGraph
from .heavy_atom import HeavyAtom, default_valence
from .resonance_structures import create_resonance_structures


# TODO perhaps used SortedDict from sortedcontainers more here?
class ChemGraph(BaseChemGraph):
    def __init__(
        self,
        graph=None,
        hatoms=None,
        bond_orders=None,
        bond_list=None,
        adj_mat=None,
        nuclear_charges=None,
        charge=0,
    ):
        """
        Object containing molecular graph, information about its nodes (heavy atoms + hydrogens) and the resonance structures.

        Args:
            graph (igraph.Graph or None): molecular graph.
            hatoms (list or None): list of HeavyAtom objects.
            bond_orders (dict or None): bond orders between heavy atoms (for the initialized resonance structure).
            all_bond_orders (dict or None): bond orders between all atoms (for the initialized resonance structure).
            adj_mat (numpy.array or None): adjacency matrix between all atoms (including hydrogens).
            nuclear_charges (numpy.array or None): all nuclear charges (including hydrogens).
        """

        super().__init__(
            graph=graph,
            hatoms=hatoms,
            bond_orders=bond_orders,
            bond_list=bond_list,
            adj_mat=adj_mat,
            nuclear_charges=nuclear_charges,
            charge=charge,
        )
        if self.info_incomplete():
            self.init_resonance_structures()

    def info_incomplete(self):
        if self.bond_orders is None:
            return True
        for ha in self.hatoms:
            if (ha.valence is None) or (ha.charge is None):
                return True
        return False

    def valences_reasonable(self):
        """
        Check that the currently initialized valences and bond orders make sense. Introduced for debugging.
        """
        for ha_id, ha in enumerate(self.hatoms):
            if not ha.charge_reasonable(charge_feasibility=self.overall_charge_feasibility):
                return False
            if not ha.valence_reasonable():
                return False
            cur_val = ha.nhydrogens
            for neigh in self.neighbors(ha_id):
                btuple = sorted_tuple(ha_id, neigh)
                cur_val += self.bond_orders[btuple]
            if ha.valence != cur_val:
                return False
        return True

    def bond_orders_equivalent(self, atom_id_set1, atom_id_set2):
        return self.aa_all_bond_orders(*atom_id_set1) == self.aa_all_bond_orders(*atom_id_set2)

    # Order of bond between atoms atom_id1 and atom_id2
    def bond_order(self, atom_id1: int, atom_id2: int, resonance_structure_id=None) -> int:
        """
        Bond order between heavy atoms with indices atom_id1 and atom_id2. (0 iff atom_id1 and atom_id2 do not share a bond.)

        Args:
            atom_id1, atom_id2 (int): indices of the heavy atoms (nodes) of interest.
            resonance_structure_id (int or None): if None return bond order for the currently initialized resonance structure. Otherwise defines id of the resonance structure for which the bond order will be returned.

        Returns:
            Bond order (int).
        """
        if resonance_structure_id is not None:
            self.init_resonance_structures()
        stuple = sorted_tuple(atom_id1, atom_id2)
        if stuple in self.bond_orders:
            if (resonance_structure_id is None) or (stuple not in self.resonance_structure_map):
                cur_bo = self.bond_orders[stuple]
            else:
                add_bos = self.resonance_structure_orders[self.resonance_structure_map[stuple]][
                    resonance_structure_id
                ]
                cur_bo = 1
                if stuple in add_bos:
                    cur_bo += add_bos[stuple]
        else:
            cur_bo = 0
        return cur_bo

    def changeable_attribute_woption(
        self,
        atom_id,
        resonance_structure_id=None,
        attr_name="valence",
        possible_vals_attr_name="possible_valences",
    ):
        """
        Returns value of changeable attribute (charge or valence) with the id.
        """
        if resonance_structure_id is not None:
            self.init_resonance_structures()
        ha = self.hatoms[atom_id]
        val = getattr(ha, attr_name)
        possible_vals = getattr(ha, possible_vals_attr_name)
        option_id = None
        if (resonance_structure_id is not None) and (possible_vals is not None):
            resonance_structure_region = self.single_atom_resonance_structure(atom_id)
            option_id = self.resonance_structure_valence_vals[resonance_structure_region][
                resonance_structure_id
            ]
            val = possible_vals[option_id]
        return val, option_id

    # Valence for a given resonance structure.
    def valence_woption(self, atom_id, resonance_structure_id=None):
        self.init_resonance_structures()
        return self.changeable_attribute_woption(
            atom_id, resonance_structure_id=resonance_structure_id
        )

    def charge_woption(self, atom_id, resonance_structure_id=None):
        self.init_resonance_structures()
        return self.changeable_attribute_woption(
            atom_id,
            resonance_structure_id=resonance_structure_id,
            attr_name="charge",
            possible_vals_attr_name="possible_charges",
        )

    def extrema_changeable_attribture_woption(
        self,
        atom_id,
        comparison_operator=max,
        attr_name="valence",
        possible_vals_attr_name="possible_valences",
    ):
        self.init_resonance_structures()
        ha = self.hatoms[atom_id]
        possible_vals = getattr(ha, possible_vals_attr_name)
        if possible_vals is None:
            cur_val = getattr(ha, attr_name)
            val_option = None
        else:
            cur_val = comparison_operator(possible_vals)
            val_option = possible_vals.index(cur_val)
        return cur_val, val_option

    def extrema_valence_woption(self, atom_id, comparison_operator=max):
        return self.extrema_changeable_attribture_woption(
            atom_id, comparison_operator=comparison_operator
        )

    def min_valence_woption(self, atom_id):
        return self.extrema_valence_woption(atom_id, comparison_operator=min)

    def max_valence_woption(self, atom_id):
        return self.extrema_valence_woption(atom_id, comparison_operator=max)

    def extrema_charge_woption(self, atom_id, comparison_operator=max):
        return self.extrema_changeable_attribture_woption(
            atom_id,
            comparison_operator=comparison_operator,
            attr_name="charge",
            possible_vals_attr_name="possible_charges",
        )

    def min_charge_woption(self, atom_id):
        return self.extrema_charge_woption(atom_id, comparison_operator=min)

    def max_charge_woption(self, atom_id):
        return self.extrema_charge_woption(atom_id, comparison_operator=max)

    def default_valence_available(self, atom_id):
        self.init_resonance_structures()
        ha_default_valence = self.hatom_default_valence(atom_id)
        ha = self.hatoms[atom_id]
        if ha.possible_valences is None:
            return ha.valence == ha_default_valence
        else:
            return ha_default_valence in ha.possible_valences

    def default_valence_resonance_structure_id(self, atom_id):
        self.init_resonance_structures()
        ha_default_valence = self.hatom_default_valence(atom_id)
        return self.atom_valence_resonance_structure_id(
            hatom_id=atom_id, valence=ha_default_valence
        )

    def possible_res_struct_ids(self, atom_id):
        self.init_resonance_structures()
        resonance_structure_region = self.single_atom_resonance_structure(atom_id)
        if resonance_structure_region is None:
            return [None]
        else:
            return range(len(self.resonance_structure_orders[resonance_structure_region]))

    def bond_order_float(self, atom_id1, atom_id2):
        """
        Float bond order between heavy atoms with indices atom_id1 and atom_id2 averaged over all resonance structures.
        """
        self.init_resonance_structures()

        stuple = sorted_tuple(atom_id1, atom_id2)
        if stuple in self.bond_orders:
            if stuple in self.resonance_structure_map:
                res_addition = 0.0
                res_struct_ords = self.resonance_structure_orders[
                    self.resonance_structure_map[stuple]
                ]
                for res_struct_ord in res_struct_ords:
                    if stuple in res_struct_ord:
                        res_addition += res_struct_ord[stuple]
                return res_addition / len(res_struct_ords) + 1.0
            else:
                return self.bond_orders[stuple]
        else:
            return 0.0

    def aa_all_bond_orders(
        self, atom_id1, atom_id2, unsorted=False, output_comparison_function=None
    ):
        """
        All bond orders between two heavy atoms.

        Args:
            atom_id1, atom_id2 (int): indices of heavy atoms.
            unsorted (bool): if False (which is default), return orders as a list with the same indexing as the resonance structures giving rise to these orders. If True sort the outptu list.
            output_comparison_function (function or None): if not None, but is `min` or `max`, return the corresponding extremum of bond order list. Default is None.

        Returns:
            List of intergers if output_comparison_function is None, otherwise an integer.
        """
        self.init_resonance_structures()

        stuple = sorted_tuple(atom_id1, atom_id2)
        if stuple in self.bond_orders:
            if stuple in self.resonance_structure_map:
                res_struct_ords = self.resonance_structure_orders[
                    self.resonance_structure_map[stuple]
                ]
                if output_comparison_function is None:
                    output = []
                else:
                    extrema_val = None
                for res_struct_ord in res_struct_ords:
                    if stuple in res_struct_ord:
                        new_bond_order = res_struct_ord[stuple] + 1
                    else:
                        new_bond_order = 1
                    if output_comparison_function is None:
                        if unsorted or (new_bond_order not in output):
                            output.append(new_bond_order)
                    else:
                        if extrema_val is None:
                            extrema_val = new_bond_order
                        extrema_val = output_comparison_function(new_bond_order, extrema_val)
                if output_comparison_function is not None:
                    return extrema_val
                else:
                    if unsorted:
                        return output
                    else:
                        return sorted(output)
            else:
                bond_val = self.bond_orders[stuple]
        else:
            bond_val = 0
        if output_comparison_function is None:
            return [bond_val]
        else:
            return bond_val

    def max_bond_order(self, atom_id1, atom_id2):
        """
        Bond order between atom_id1 and atom_id2 which is maximum over all resonance structures.
        """
        return self.aa_all_bond_orders(atom_id1, atom_id2, output_comparison_function=max)

    def min_bond_order(self, atom_id1, atom_id2):
        """
        Bond order between atom_id1 and atom_id2 which is minimum over all resonance structures.
        """
        return self.aa_all_bond_orders(atom_id1, atom_id2, output_comparison_function=min)

    def valence_config_valid(self, checked_valences):
        """
        Check whether heavy atom valences provided in checked_valences array are valid for some resonance structure.
        """
        self.init_resonance_structures()
        for ha, cv in zip(self.hatoms, checked_valences):
            if ha.possible_valences is None:
                if ha.valence != cv:
                    return False

        for extra_val_ids in self.resonance_structure_inverse_map:
            num_poss = None
            for evi in extra_val_ids:
                ha = self.hatoms[evi]
                poss_vals = ha.possible_valences
                if poss_vals is not None:
                    num_poss = len(poss_vals)
                    break
            if num_poss is not None:
                for val_poss in range(num_poss):
                    match_found = True
                    for evi in extra_val_ids:
                        poss_vals = self.hatoms[evi].possible_valences
                        if poss_vals is not None:
                            if poss_vals[val_poss] != checked_valences[evi]:
                                match_found = False
                                break
                    if match_found:
                        break
                if not match_found:
                    return False

        return True

    # TODO Can this be achieved with fewer operations?
    def attempt_minimize_valences(self):
        if self.polyvalent_atoms_present():
            old_valences = [ha.valence for ha in self.hatoms]
            self.init_resonance_structures()
            return self.valence_config_valid(old_valences)
        else:
            return True

    def valence_change_range(self, ha_ids):
        for ha_id in ha_ids:
            hatom = self.hatoms[ha_id]
            poss_valences = hatom.possible_valences
            if poss_valences is not None:
                return len(poss_valences), poss_valences.index(hatom.valence)
        return None, None

    def change_valence_option(self, hatom_ids, valence_option_id):
        if valence_option_id is not None:
            for hatom_id in hatom_ids:
                if self.hatoms[hatom_id].possible_valences is not None:
                    self.hatoms[hatom_id].valence = self.hatoms[hatom_id].possible_valences[
                        valence_option_id
                    ]

    def single_atom_resonance_structure(self, hatom_id):
        """
        Which resonance structure region contains hatom_id.
        """
        for neigh in self.neighbors(hatom_id):
            st = sorted_tuple(neigh, hatom_id)
            if st in self.resonance_structure_map:
                return self.resonance_structure_map[st]
        return None

    def atom_valence_resonance_structure_id(
        self, hatom_id=None, resonance_region_id=None, valence=None
    ):
        self.init_resonance_structures()
        possible_valences = self.hatoms[hatom_id].possible_valences
        if possible_valences is None:
            return None
        if resonance_region_id is None:
            if hatom_id is None:
                raise Exception()
            resonance_region_id = self.single_atom_resonance_structure(hatom_id)
        resonance_option_id = possible_valences.index(valence)

        return self.resonance_structure_valence_vals[resonance_region_id].index(
            resonance_option_id
        )

    def fill_missing_bond_orders(self):
        for ha_id1 in range(self.nhatoms()):
            for ha_id2 in self.neighbors(ha_id1):
                if ha_id1 > ha_id2:
                    continue
                bond_tuple = (ha_id1, ha_id2)
                if bond_tuple not in self.bond_orders:
                    self.bond_orders[bond_tuple] = 1

    def init_resonance_structures(self):
        # skip if resonance structures have already been initialized
        if (
            (self.resonance_structure_orders is not None)
            and (self.resonance_structure_valence_vals is not None)
            and (self.resonance_structure_map is not None)
            and (self.resonance_structure_inverse_map is not None)
            and (self.resonance_structure_charge_feasibilities is not None)
        ):
            return
        create_resonance_structures(self)
        # By this point, both charges and valences are initialized to correspond to resonance structures with id 0.
        # Here we initialize the bond orders accordingly.
        for resonance_region_id, bond_orders in enumerate(self.resonance_structure_orders):
            if len(bond_orders) == 0:
                # we don't need to adjust bonds
                continue
            self.adjust_resonance_bond_no_valences_no_init(resonance_region_id, 0)
        # create_resonance_structures first sets self.bond_orders to {}, then initializes
        # all the bonds with extra valences. All other ("missing") bond orders should be given order 1.
        self.fill_missing_bond_orders()

    # More sophisticated commands that are to be called in the "modify" module.
    def change_bond_order(self, atom1, atom2, bond_order_change, resonance_structure_id=None):
        if bond_order_change != 0:
            # TODO a way to avoid calling len here?
            if resonance_structure_id is not None:
                if len(self.aa_all_bond_orders(atom1, atom2)) != 1:
                    # Make sure that the correct resonance structure is used as initial one.
                    resonance_structure_region = self.resonance_structure_map[
                        sorted_tuple(atom1, atom2)
                    ]
                    self.adjust_resonance_valences(
                        resonance_structure_region, resonance_structure_id
                    )

            self.change_edge_order(atom1, atom2, bond_order_change)

            for atom_id in [atom1, atom2]:
                self.change_hydrogen_number(atom_id, -bond_order_change)

            self.changed()

    def adjust_resonance_bond_no_valences_no_init(
        self, resonance_structure_region, resonance_structure_id
    ):
        changed_hatom_ids = self.resonance_structure_inverse_map[resonance_structure_region]
        cur_resonance_struct_orders = self.resonance_structure_orders[resonance_structure_region][
            resonance_structure_id
        ]
        self.assign_extra_edge_orders(changed_hatom_ids, cur_resonance_struct_orders)
        return changed_hatom_ids

    def adjust_resonance_valences(self, resonance_structure_region, resonance_structure_id):
        self.init_resonance_structures()
        changed_hatom_ids = self.adjust_resonance_bond_no_valences_no_init(
            resonance_structure_region, resonance_structure_id
        )
        new_valence_option = self.resonance_structure_valence_vals[resonance_structure_region][
            resonance_structure_id
        ]
        self.change_valence_option(changed_hatom_ids, new_valence_option)

    def adjust_resonance_valences_atom(
        self, atom_id, resonance_structure_id=None, valence_option_id=None
    ):
        if (resonance_structure_id is not None) or (valence_option_id is not None):
            self.init_resonance_structures()
            adjusted_resonance_region = self.single_atom_resonance_structure(atom_id)
            if resonance_structure_id is None:
                resonance_structure_id = self.resonance_structure_valence_vals[
                    adjusted_resonance_region
                ].index(valence_option_id)
            self.adjust_resonance_valences(adjusted_resonance_region, resonance_structure_id)

    # TODO Do we need resonance structure invariance here?
    def remove_heavy_atom(self, atom_id, resonance_structure_id=None):
        self.adjust_resonance_valences_atom(atom_id, resonance_structure_id=resonance_structure_id)

        for neigh_id in self.neighbors(atom_id):
            cur_bond_order = self.bond_order(neigh_id, atom_id)
            self.change_bond_order(atom_id, neigh_id, -cur_bond_order)

        self.graph.delete_vertices([atom_id])
        # TO-DO: is it possible to instead tie dict keys to edges of self.graph?
        new_bond_orders = {}
        for bond_tuple, bond_order in self.bond_orders.items():
            new_tuple = []
            for b in bond_tuple:
                if b > atom_id:
                    b -= 1
                new_tuple.append(b)
            new_bond_orders[tuple(new_tuple)] = bond_order
        self.bond_orders = new_bond_orders
        del self.hatoms[atom_id]
        self.changed()

    def remove_heavy_atoms(self, atom_ids):
        sorted_atom_ids = sorted(atom_ids, reverse=True)
        for atom_id in sorted_atom_ids:
            self.remove_heavy_atom(atom_id)

    def add_heavy_atom_chain(
        self,
        modified_atom_id,
        new_chain_atoms,
        new_chain_atom_valences=None,
        chain_bond_orders=None,
    ):
        bonded_chain = [modified_atom_id]
        last_added_id = self.nhatoms()
        num_added_atoms = len(new_chain_atoms)
        self.graph.add_vertices(num_added_atoms)
        for chain_id, new_chain_atom in enumerate(new_chain_atoms):
            bonded_chain.append(last_added_id)
            last_added_id += 1
            if new_chain_atom_valences is None:
                new_valence = default_valence(new_chain_atom)
            else:
                new_valence = new_chain_atom_valences[chain_id]
            self.hatoms.append(HeavyAtom(new_chain_atom, valence=new_valence))
            self.hatoms[-1].nhydrogens = self.hatoms[-1].valence

        for i in range(num_added_atoms):
            if chain_bond_orders is None:
                new_bond_order = 1
            else:
                new_bond_order = chain_bond_orders[i]
            self.change_bond_order(bonded_chain[i], bonded_chain[i + 1], new_bond_order)

    def replace_heavy_atom(
        self,
        replaced_atom_id,
        inserted_atom,
        inserted_valence=None,
        resonance_structure_id=None,
    ):
        self.adjust_resonance_valences_atom(
            replaced_atom_id, resonance_structure_id=resonance_structure_id
        )
        self.hatoms[replaced_atom_id].ncharge = int_atom_checked(inserted_atom)
        old_valence = self.hatoms[replaced_atom_id].valence
        if inserted_valence is None:
            inserted_valence = default_valence(inserted_atom)
        self.hatoms[replaced_atom_id].valence = inserted_valence
        self.change_hydrogen_number(replaced_atom_id, inserted_valence - old_valence)
        self.changed()

    def change_valence(self, modified_atom_id, new_valence):
        ha = self.hatoms[modified_atom_id]
        self.change_hydrogen_number(modified_atom_id, new_valence - ha.valence)
        ha.valence = new_valence
        self.changed()

    # For properties used to generate representations.
    def get_res_av_bond_orders(self, edge_list=None):
        if edge_list is None:
            edge_list = self.graph.get_edgelist()
        # Get dictionnaries for all resonance structures.
        res_struct_dict_list = self.resonance_structure_orders
        res_av_bos = np.ones(len(edge_list), dtype=float)
        all_av_res_struct_dict = {}
        for res_struct_dict in res_struct_dict_list:
            av_res_struct_dict = res_struct_dict[0]
            for add_rsd in res_struct_dict[1:]:
                for bond_edge, bond_order in add_rsd.items():
                    if bond_edge in av_res_struct_dict:
                        av_res_struct_dict[bond_edge] += bond_order
                    else:
                        av_res_struct_dict[bond_edge] = bond_order
            for edge in av_res_struct_dict:
                av_res_struct_dict[edge] = float(av_res_struct_dict[edge]) / len(res_struct_dict)
            all_av_res_struct_dict = {**all_av_res_struct_dict, **av_res_struct_dict}
        for eid, edge in enumerate(edge_list):
            if edge in all_av_res_struct_dict:
                res_av_bos[eid] += all_av_res_struct_dict[edge]
        return res_av_bos

    def shortest_paths(self, weights=None):
        if self.nhatoms() == 1:
            return np.array([[0.0]])
        else:
            return np.array(self.graph.shortest_paths(weights=weights))


def canonically_permuted_ChemGraph(cg: ChemGraph) -> ChemGraph:
    """
    Get version of cg where hatoms are canonically ordered.
    """
    cg.init_canonical_permutation()
    new_nuclear_charges = np.ones((cg.natoms(),), dtype=int)
    new_adj_mat = np.zeros((cg.natoms(), cg.natoms()), dtype=int)
    cur_h_id = cg.nhatoms()
    for hatom_canon_id, hatom_id in enumerate(cg.inv_canonical_permutation):
        ha = cg.hatoms[hatom_id]
        new_nuclear_charges[hatom_canon_id] = ha.ncharge
        for neigh in cg.neighbors(hatom_id):
            canon_neigh = cg.canonical_permutation[neigh]
            new_adj_mat[
                canon_neigh, hatom_canon_id
            ] = 1  # non-unity bond orders will be fixed during ChemGraph.__init__()
        for _ in range(ha.nhydrogens):
            new_adj_mat[hatom_canon_id, cur_h_id] = 1
            new_adj_mat[cur_h_id, hatom_canon_id] = 1
            cur_h_id += 1
    new_cg = ChemGraph(nuclear_charges=new_nuclear_charges, adj_mat=new_adj_mat, charge=cg.charge)
    if new_cg != cg:
        raise Exception
    return new_cg


def chemgraph_str2unchecked_adjmat_ncharges(input_string: str) -> tuple:
    """
    Converts a ChemGraph string representation into the adjacency matrix (with all bond orders set to one) and nuclear charges.

    Args:
        input_string : string to be converted
    Returs:
        tuple (adj_mat, nuclear_charges, charge) of adjacency matrix, nuclear charges, and charge.
    """
    input_string_split = input_string.split("_")
    if len(input_string_split) != 1:
        charge = int(input_string_split[1])
    else:
        charge = 0

    hatom_ncharges = []
    hydrogen_nums = []
    hatom_neighbors = []
    for hatom_str in input_string_split[0].split(":"):
        if "@" in hatom_str:
            hatom_str_split = hatom_str.split("@")
            hatom_str_remainder = hatom_str_split[0]
            hatom_neighbors.append([int(neigh_id) for neigh_id in hatom_str_split[1:]])
        else:
            hatom_str_remainder = hatom_str
            hatom_neighbors.append([])
        if "#" in hatom_str_remainder:
            hatom_str_split = hatom_str_remainder.split("#")
            hatom_ncharges.append(int(hatom_str_split[0]))
            hydrogen_nums.append(int(hatom_str_split[1]))
        else:
            hatom_ncharges.append(int(hatom_str_remainder))
            hydrogen_nums.append(0)
    nuclear_charges = np.append(
        np.array(hatom_ncharges), np.ones((sum(hydrogen_nums),), dtype=int)
    )
    natoms = len(nuclear_charges)
    adj_mat = np.zeros((natoms, natoms), dtype=int)
    last_h_id = len(hatom_ncharges)
    for i, (j_list, hnum) in enumerate(zip(hatom_neighbors, hydrogen_nums)):
        for j in j_list:
            adj_mat[i, j] = 1
            adj_mat[j, i] = 1
        for _ in range(hnum):
            adj_mat[i, last_h_id] = 1
            adj_mat[last_h_id, i] = 1
            last_h_id += 1

    return adj_mat, nuclear_charges, charge


def str2ChemGraph(input_string: str, shuffle=False) -> ChemGraph:
    """
    Converts a ChemGraph string representation into a ChemGraph object.
    input_string : string to be converted
    shuffle : whether atom positions should be shuffled, introduced for testing purposes.
    """
    unchecked_adjmat, ncharges, charge = chemgraph_str2unchecked_adjmat_ncharges(input_string)
    if shuffle:  # TODO should I use shuffled_chemgraph here?
        ids = list(range(len(ncharges)))
        random.shuffle(ids)
        ncharges = ncharges[ids]
        unchecked_adjmat = unchecked_adjmat[ids][:, ids]
    return ChemGraph(nuclear_charges=ncharges, adj_mat=unchecked_adjmat, charge=charge)


def shuffled_chemgraph(chemgraph_in: ChemGraph) -> ChemGraph:
    """
    Returns an identical chemgraph whose atoms have been shuffled.
    """
    ncharges = chemgraph_in.full_ncharges()
    adjmat = chemgraph_in.full_adjmat()
    ids = list(range(len(ncharges)))
    random.shuffle(ids)
    shuffled_ncharges = ncharges[ids]
    shuffled_adjmat = adjmat[ids][:, ids]
    return ChemGraph(
        nuclear_charges=shuffled_ncharges,
        adj_mat=shuffled_adjmat,
        charge=chemgraph_in.charge,
    )


# For splitting chemgraphs.
def split_chemgraph_no_dissociation_check(cg_input, membership_vector, copied=False):
    if copied:
        cg = cg_input
    else:
        cg = copy.deepcopy(cg_input)
    subgraphs_vertex_ids = sorted_by_membership(membership_vector)

    new_graph_list = []
    new_hatoms_list = []
    new_bond_orders_list = []
    new_bond_positions_orders_dict_list = []

    for vertex_ids in subgraphs_vertex_ids:
        new_graph_list.append(cg.graph.subgraph(vertex_ids))
        new_hatoms_list.append([cg.hatoms[vertex_id] for vertex_id in vertex_ids])
        new_bond_orders_list.append({})
        new_bond_positions_orders_dict_list.append({})

    # Create bond order dictionnaries:
    for vertex_id in range(cg.nhatoms()):
        mv1 = membership_vector[vertex_id]
        internal_id1 = subgraphs_vertex_ids[mv1].index(vertex_id)
        for neigh in cg.graph.neighbors(vertex_id):
            mv2 = membership_vector[neigh]
            internal_id2 = subgraphs_vertex_ids[mv2].index(neigh)
            cur_bond_order = cg.bond_order(vertex_id, neigh)
            if mv1 == mv2:
                new_bond_orders_list[mv1] = {
                    **new_bond_orders_list[mv1],
                    sorted_tuple(internal_id1, internal_id2): cur_bond_order,
                }
            else:
                if internal_id1 in new_bond_positions_orders_dict_list[mv1]:
                    new_bond_positions_orders_dict_list[mv1][internal_id1].append(cur_bond_order)
                else:
                    new_bond_positions_orders_dict_list[mv1][internal_id1] = [cur_bond_order]

    output = []
    for graph, hatoms, bond_orders, bond_positions_orders_dict in zip(
        new_graph_list,
        new_hatoms_list,
        new_bond_orders_list,
        new_bond_positions_orders_dict_list,
    ):
        bond_positions_orders = []
        for atom_id, connection_orders in bond_positions_orders_dict.items():
            for connection_order in connection_orders:
                bond_positions_orders.append((atom_id, connection_order))
                hatoms[atom_id].nhydrogens += connection_order
        new_cg = ChemGraph(
            graph=graph,
            hatoms=hatoms,
            bond_orders=bond_orders,
            charge=sum(ha.charge for ha in hatoms),
        )

        # Append virtual atoms at the place of broken bonds.
        for modified_atom, extra_bond_orders in bond_positions_orders_dict.items():
            for extra_bond_order in extra_bond_orders:
                new_cg.add_heavy_atom_chain(
                    modified_atom,
                    [0],
                    chain_bond_orders=[extra_bond_order],
                    new_chain_atom_valences=[extra_bond_order],
                )

        output.append(new_cg)
    return output


def split_chemgraph_into_connected_fragments(cg_input, copied=False):
    connected_fragment_membership_vector = cg_input.graph.components().membership
    return split_chemgraph_no_dissociation_check(
        cg_input, connected_fragment_membership_vector, copied=copied
    )


def combine_chemgraphs(cg1, cg2):
    new_hatoms = copy.deepcopy(cg1.hatoms + cg2.hatoms)
    new_graph = disjoint_union([cg1.graph, cg2.graph])
    id2_shift = cg1.nhatoms()
    new_bond_orders = copy.deepcopy(cg1.bond_orders)
    for bond_tuple, bond_order in cg2.bond_orders.items():
        new_bond_tuple = tuple(np.array(bond_tuple) + id2_shift)
        new_bond_orders[new_bond_tuple] = bond_order
    return ChemGraph(hatoms=new_hatoms, bond_orders=new_bond_orders, graph=new_graph)


def h2chemgraph():
    """
    ChemGraph representation of the H2 molecule.
    """
    # Very dirty, but there seems to be no better way.
    output = ChemGraph(nuclear_charges=[9, 9], adj_mat=[[0, 1], [1, 0]])
    for i in range(2):
        output.hatoms[i].ncharge = 1
    return output
