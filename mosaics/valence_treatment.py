# TODO: Perhaps in the future everything related to graph manipulation should
# be included into ExtGraphCompound instead.
# TODO Perhaps defining "canonically_permuted" inside a ChemGraph can simplify some expressions.
# TODO I have rewritten the way resonance structures are treated, as a result some subroutines used for representation generation
# may not work.

import itertools, copy, random
import numpy as np
from igraph import Graph
from igraph.operators import disjoint_union
from .periodic import (
    valences_int,
    period_int,
    s_int,
    p_int,
    unshared_pairs,
    coord_num_hybrid,
)
from .misc_procedures import (
    str_atom_corr,
    int_atom_checked,
    sorted_by_membership,
    sorted_tuple,
)


class InvalidAdjMat(Exception):
    pass


class InvalidChange(Exception):
    pass


# Introduced in case we, for example, started to consider F as a default addition instead.
DEFAULT_ELEMENT = 1

# To avoid equality expressions for two reals.
irrelevant_bond_order_difference = 1.0e-8


def avail_val_list(atom_id):
    return valences_int[int_atom_checked(atom_id)]


# Default valence for a given element.
def default_valence(atom_id):
    val_list = avail_val_list(atom_id)
    if isinstance(val_list, tuple):
        return val_list[0]
    else:
        return val_list


# Color obj_list in a way that each equal obj1 and obj2 were the same color.
# Used for defining canonical permutation of a graph.
def list2colors(obj_list):
    ids_objs = list(enumerate(obj_list))
    ids_objs.sort(key=lambda x: x[1])
    num_obj = len(ids_objs)
    colors = np.zeros(num_obj, dtype=int)
    cur_color = 0
    prev_obj = ids_objs[0][1]
    for i in range(1, num_obj):
        cur_obj = ids_objs[i][1]
        if cur_obj != prev_obj:
            cur_color += 1
            prev_obj = cur_obj
        colors[ids_objs[i][0]] = cur_color
    return colors


class HeavyAtom:
    def __init__(
        self,
        atom_symbol,
        valence=None,
        nhydrogens=0,
        coordination_number=None,
        possible_valences=None,
    ):
        """
        Class storing information about a heavy atom and the hydrogens connected to it.
        """
        self.ncharge = int_atom_checked(atom_symbol)
        if valence is None:
            valence = self.smallest_valid_valence(
                coordination_number=coordination_number
            )
        self.valence = valence
        self.nhydrogens = nhydrogens
        self.possible_valences = possible_valences
        self.changed()

    def changed(self):
        self.comparison_list = None

    # Valence-related.
    def avail_val_list(self):
        if self.ncharge == 0:
            return self.valence
        else:
            return avail_val_list(self.ncharge)

    def is_polyvalent(self):
        return isinstance(valences_int[self.ncharge], tuple)

    def valence_reasonable(self):
        val_list = self.avail_val_list()
        if isinstance(val_list, tuple):
            return self.valence in val_list
        else:
            return self.valence == val_list

    def smallest_valid_valence(self, coordination_number=None, show_id=False):
        if self.ncharge == 0:
            return self.valence

        val_list = self.avail_val_list()
        if isinstance(val_list, tuple):
            needed_id = None
            if coordination_number is None:
                needed_id = 0
            else:
                for i, valence in enumerate(val_list):
                    if valence >= coordination_number:
                        needed_id = i
                        break
                if needed_id is None:
                    raise InvalidAdjMat
            if show_id:
                return needed_id
            else:
                return val_list[needed_id]
        else:
            if coordination_number is not None:
                if coordination_number > val_list:
                    raise InvalidAdjMat
            if show_id:
                return -1
            else:
                return val_list

    def valence_val_id(self):
        vals = self.avail_val_list()
        if isinstance(vals, tuple):
            return self.avail_val_list().index(self.valence)
        else:
            return 0

    def max_valence(self):
        val_list = self.avail_val_list()
        if isinstance(val_list, tuple):
            return val_list[-1]
        else:
            return val_list

    def mincopy(self):
        """
        Not %100 sure whether this should be made __deepcopy__ instead.
        """
        return HeavyAtom(
            atom_symbol=self.ncharge, valence=self.valence, nhydrogens=self.nhydrogens
        )

    def element_name(self):
        return str_atom_corr(self.ncharge)

    # Procedures for ordering.
    def get_comparison_list(self):
        if self.comparison_list is None:
            #           This early draft was written to keep more similar hatoms closer to each other for representation purposes, but I am no longer sure it is useful.
            #           Commented out after an update that made .valence_val_id() dependent on resonance structure.
            #            if self.ncharge == 0:
            #                s = -1
            #                p = -1
            #                per = -1
            #            else:
            #                s = s_int[self.ncharge]
            #                p = p_int[self.ncharge]
            #                per = period_int[self.ncharge]
            #            self.comparison_list = [
            #                self.valence - self.nhydrogens,
            #                s,
            #                p,
            #                per,
            #                self.valence_val_id(),
            #            ]
            self.comparison_list = [self.ncharge, self.nhydrogens]
        return self.comparison_list

    def __lt__(self, ha2):
        return self.get_comparison_list() < ha2.get_comparison_list()

    def __gt__(self, ha2):
        return self.get_comparison_list() > ha2.get_comparison_list()

    def __eq__(self, ha2):
        return self.get_comparison_list() == ha2.get_comparison_list()

    # Procedures for printing.
    def __str__(self):
        return (
            "C:"
            + str(self.ncharge)
            + ",V:"
            + str(self.valence)
            + ",H:"
            + str(self.nhydrogens)
        )

    def __repr__(self):
        return str(self)


def str2HeavyAtom(ha_str: str):
    fields = ha_str.split(",")
    return HeavyAtom(
        int(fields[0][2:]), valence=int(fields[1][2:]), nhydrogens=int(fields[2][2:])
    )


# TODO check that the function is not duplicated elsewhere
def next_valence(
    ha: HeavyAtom, int_step: int = 1, valence_option_id: int or None = None
):
    """
    Next valence value.
    """
    val_list = ha.avail_val_list()
    if (valence_option_id is not None) and (ha.possible_valences is not None):
        cur_valence = ha.possible_valences[valence_option_id]
    else:
        cur_valence = ha.valence
    cur_val_id = val_list.index(cur_valence)
    new_val_id = cur_val_id + int_step
    if (new_val_id < 0) or (new_val_id >= len(val_list)):
        return None
    else:
        return val_list[new_val_id]


# Function saying how large can a bond order be between two atoms.
def max_bo(hatom1, hatom2):
    return 3


# Functions that should help defining a meaningful distance measure between Heavy_Atom objects. TO-DO: Still need those?


def hatom_state_coords(ha):
    return [
        period_int[ha.ncharge],
        s_int[ha.ncharge],
        p_int[ha.ncharge],
        ha.valence,
        ha.nhydrogens,
    ]


num_state_coords = {hatom_state_coords: 5}

# Auxiliary functions
def adj_mat2bond_orders(adj_mat):
    bos = {}
    for atom1, adj_mat_row in enumerate(adj_mat):
        for atom2, adj_mat_val in enumerate(adj_mat_row[:atom1]):
            if adj_mat_val != 0:
                bos[(atom2, atom1)] = adj_mat_val
    return bos


# Old draft. I'm no longer sure things should be done this way.
# class ValenceConfigurationCharacter:
#    def __init__(self, valence_values):
#        """
#        An auxiliary class introduced for comparisons of valence configurations and sorting them according to relevance.
#        """
#        self.sorted_valence_values = sorted(valence_values, reverse=True)
#
#    def __eq__(self, other_vcc):
#        return self.sorted_valence_values == other_vcc.sorted_valence_values
#
#    def __gt__(self, other_vcc):
#        if len(self.sorted_valence_values) != len(other_vcc.sorted_valence_values):
#            return len(self.sorted_valence_values) > len(
#                other_vcc.sorted_valence_values
#            )
#        for s1, s2 in zip(self.sorted_valence_values, other_vcc.sorted_valence_values):
#            if s1 != s2:
#                return s1 > s2
# Must be equal then.
#        return False
#
#    def __lt__(self, other_vcc):
#        return not self > other_vcc


def ValenceConfigurationCharacter(valence_values):
    return sum(valence_values)


# TODO perhaps used SortedDict from sortedcontainers more here?
class ChemGraph:
    def __init__(
        self,
        graph=None,
        hatoms=None,
        bond_orders=None,
        all_bond_orders=None,
        adj_mat=None,
        nuclear_charges=None,
        hydrogen_autofill=False,
        hydrogen_numbers=None,
    ):
        self.graph = graph
        self.hatoms = hatoms
        self.bond_orders = bond_orders
        self.all_bond_orders = all_bond_orders

        if self.hatoms is not None:
            if nuclear_charges is None:
                self.nuclear_charges = [ha.ncharge for ha in self.hatoms]
                if graph is None:
                    self.all_bond_orders = {}
                    for t, bo in self.bond_orders.items():
                        self.all_bond_orders[sorted_tuple(*t)] = bo
                    cur_h_id = len(self.nuclear_charges)
                    for ha_id, ha in enumerate(self.hatoms):
                        for _ in range(ha.nhydrogens):
                            self.nuclear_charges.append(1)
                            self.all_bond_orders[(ha_id, cur_h_id)] = 1
                            cur_h_id += 1
                    nuclear_charges = self.nuclear_charges

        if (self.all_bond_orders is None) and (self.bond_orders is None):
            if self.graph is None:
                bo_input = adj_mat2bond_orders(adj_mat)
                if hydrogen_autofill:
                    self.bond_orders = bo_input
                else:
                    self.all_bond_orders = bo_input
            else:
                self.bond_orders = {}
                for e in self.graph.get_edgelist():
                    self.bond_orders[e] = 1
        if (self.graph is None) or (self.hatoms is None):
            self.init_graph_natoms(
                np.array(nuclear_charges),
                hydrogen_autofill=hydrogen_autofill,
                hydrogen_numbers=hydrogen_numbers,
            )
        # TODO Check for ways to combine finding resonance structures with reassigning pi bonds.
        # Check that valences make sense.

        self.changed()

        self.init_resonance_structures()

    def init_graph_natoms(
        self, nuclear_charges, hydrogen_autofill=False, hydrogen_numbers=None
    ):
        self.hatoms = []
        if self.bond_orders is not None:
            heavy_bond_orders = self.bond_orders
        self.bond_orders = {}
        heavy_atom_dict = {}
        for atom_id, ncharge in enumerate(nuclear_charges):
            if ncharge != DEFAULT_ELEMENT:
                heavy_atom_dict[atom_id] = len(self.hatoms)
                self.hatoms.append(HeavyAtom(ncharge, valence=0))
        self.graph = Graph(n=len(self.hatoms), directed=False)
        if hydrogen_autofill:
            filled_bond_orders = heavy_bond_orders
        else:
            filled_bond_orders = self.all_bond_orders
        for bond_tuple, bond_order in filled_bond_orders.items():
            for ha_id1, ha_id2 in itertools.permutations(bond_tuple):
                if ha_id1 in heavy_atom_dict:
                    true_id = heavy_atom_dict[ha_id1]
                    self.hatoms[true_id].valence += bond_order
                    if ha_id2 in heavy_atom_dict:
                        if ha_id1 < ha_id2:
                            self.change_edge_order(
                                true_id, heavy_atom_dict[ha_id2], bond_order
                            )
                    else:
                        self.hatoms[true_id].nhydrogens += bond_order
        if hydrogen_autofill:
            if hydrogen_numbers is None:
                for ha_id, hatom in enumerate(self.hatoms):
                    cur_assigned_valence = hatom.valence
                    self.hatoms[ha_id].valence = hatom.smallest_valid_valence(
                        coordination_number=cur_assigned_valence
                    )
                    self.hatoms[ha_id].nhydrogens += (
                        self.hatoms[ha_id].valence - cur_assigned_valence
                    )
            else:
                for ha_id, nhydrogens in enumerate(hydrogen_numbers):
                    self.hatoms[ha_id].nhydrogens = nhydrogens
                    self.hatoms[ha_id].valence = nhydrogens
                for e in self.graph.get_edgelist():
                    self.hatoms[e[0]].valence += self.bond_orders[e]
                    self.hatoms[e[1]].valence += self.bond_orders[e]

    # If was modified (say, atoms added/removed), some data becomes outdated.
    def changed(self):
        self.canonical_permutation = None
        self.inv_canonical_permutation = None
        self.colors = None
        # TO-DO: perhaps sort by classes in the beginning instead?
        self.equivalence_vector = None
        self.pair_equivalence_matrix = None

        self.resonance_structure_orders = None
        self.resonance_structure_map = None
        self.resonance_structure_inverse_map = None
        self.resonance_structure_valence_vals = None

        for ha in self.hatoms:
            ha.possible_valences = None

        self.comparison_list = None

    # Checking graph's state.
    def valences_reasonable(self):
        for ha_id, ha in enumerate(self.hatoms):
            if not ha.valence_reasonable():
                return False
            cur_val = ha.nhydrogens
            for neigh in self.neighbors(ha_id):
                btuple = sorted_tuple(ha_id, neigh)
                cur_val += self.bond_orders[btuple]
            if ha.valence != cur_val:
                return False
        return True

    def polyvalent_atoms_present(self):
        for ha in self.hatoms:
            if ha.is_polyvalent():
                return True
        return False

    def coordination_number(self, hatom_id):
        return len(self.neighbors(hatom_id)) + self.hatoms[hatom_id].nhydrogens

    # Everything related to equivalences.
    def num_equiv_classes_from_arr(self, equiv_class_arr):
        return np.amax(equiv_class_arr) + 1

    def num_equiv_classes(self, atom_set_length):
        if atom_set_length == 1:
            return self.num_equiv_classes_from_arr(self.equivalence_vector)
        else:
            return self.num_equiv_classes_from_arr(self.pair_equivalence_matrix)

    def init_equivalence_vector(self):
        if self.equivalence_vector is None:
            self.equivalence_vector = -np.ones((self.nhatoms(),), dtype=int)

    def init_pair_equivalence_matrix(self):
        if self.pair_equivalence_matrix is None:
            self.pair_equivalence_matrix = -np.ones(
                (self.nhatoms(), self.nhatoms()), dtype=int
            )

    def init_equivalence_array(self, atom_set_length):
        if atom_set_length == 1:
            self.init_equivalence_vector()
        else:
            self.init_pair_equivalence_matrix()

    def equiv_arr(self, atom_set_length):
        if atom_set_length == 1:
            return self.equivalence_vector
        else:
            return self.pair_equivalence_matrix

    def checked_equivalence_vector(self):
        """
        Check that all equivalence classes have been initialized and return the resulting equivalence vector.
        """
        for i in range(self.nhatoms()):
            self.check_equivalence_class((i,))
        return self.equivalence_vector

    def sorted_colors(self, atom_set):
        self.init_colors()
        return sorted([self.colors[atom_id] for atom_id in atom_set])

    def atom_sets_equivalence_reasonable(self, atom_id_set1, atom_id_set2):
        if len(atom_id_set1) == 1:
            self.init_colors()
            return self.colors[atom_id_set1[0]] == self.colors[atom_id_set2[0]]
        else:
            if self.sorted_colors(atom_id_set1) != self.sorted_colors(atom_id_set2):
                return False
            if self.aa_all_bond_orders(*atom_id_set1) != self.aa_all_bond_orders(
                *atom_id_set2
            ):
                return False
            for atom_id_set2_permut in itertools.permutations(atom_id_set2):
                match = True
                for atom_id1, atom_id2 in zip(atom_id_set1, atom_id_set2_permut):
                    if not self.atom_pair_equivalent(atom_id1, atom_id2):
                        match = False
                        break
                if match:
                    return True
            return False

    def assign_equivalence_class(self, atom_id_set, assigned_val):
        if len(atom_id_set) == 1:
            self.equivalence_vector[atom_id_set] = assigned_val
        else:
            self.pair_equivalence_matrix[atom_id_set] = assigned_val
            self.pair_equivalence_matrix[atom_id_set[::-1]] = assigned_val

    def equiv_class_examples(self, atom_set_length, as_tuples=True):
        num_classes = self.num_equiv_classes(atom_set_length)
        equiv_class_arr = self.equiv_arr(atom_set_length)
        output = []
        for equiv_class_id in range(num_classes):
            where_output = np.where(equiv_class_arr == equiv_class_id)
            example_tuple = tuple(where_column[0] for where_column in where_output)
            if not as_tuples:
                example_tuple = example_tuple[0]
            output.append(example_tuple)
        return output

    def equiv_class_members(self, equiv_class_id, atom_set_length):
        return list(zip(*np.where(self.equiv_arr(atom_set_length) == equiv_class_id)))

    def min_id_equivalent_atom_unchecked(self, atom_id):
        if self.equivalence_vector is None:
            return atom_id
        equiv_class = self.unchecked_equivalence_class(atom_id)
        if equiv_class == -1:
            return atom_id
        else:
            return np.where(self.equivalence_vector == equiv_class)[0][0]

    def check_equivalence_class(self, atom_id_set):
        atom_set_length = len(atom_id_set)
        self.init_equivalence_array(atom_set_length)
        if self.unchecked_equivalence_class(atom_id_set) == -1:
            self.init_colors()
            for equiv_class_id, example_tuple in enumerate(
                self.equiv_class_examples(atom_set_length)
            ):
                if self.atom_sets_equivalence_reasonable(atom_id_set, example_tuple):
                    if self.uninit_atom_sets_equivalent(atom_id_set, example_tuple):
                        self.assign_equivalence_class(atom_id_set, equiv_class_id)
                        return
            self.assign_equivalence_class(
                atom_id_set, self.num_equiv_classes(atom_set_length)
            )

    def uninit_atom_sets_equivalent(self, atom_set1, atom_set2):
        self.init_colors()

        temp_colors1 = copy.deepcopy(self.colors)
        temp_colors2 = copy.deepcopy(self.colors)
        dummy_color = max(self.colors) + 1
        for atom_id1, atom_id2 in zip(atom_set1, atom_set2):
            temp_colors1[atom_id1] = dummy_color
            temp_colors2[atom_id2] = dummy_color
        return self.graph.isomorphic_vf2(
            self.graph, color1=temp_colors1, color2=temp_colors2
        )

    def atom_pair_equivalent(self, atom_id1, atom_id2):
        return self.atom_sets_equivalent([atom_id1], [atom_id2])

    def unchecked_equivalence_class(self, atom_set):
        if isinstance(atom_set, int) or (len(atom_set) == 1):
            return self.equivalence_vector[atom_set]
        else:
            return self.pair_equivalence_matrix[atom_set]

    def equivalence_class(self, atom_set):
        self.check_equivalence_class(atom_set)
        return self.unchecked_equivalence_class(atom_set)

    def atom_sets_equivalent(self, atom_set1, atom_set2):
        if len(atom_set1) == 2:
            return self.uninit_atom_sets_equivalent(atom_set1, atom_set2)
        return self.equivalence_class(atom_set1) == self.equivalence_class(atom_set2)

    # How many times atom_id is repeated inside a molecule.
    def atom_multiplicity(self, atom_id):
        return sum(
            self.atom_pair_equivalent(atom_id, other_atom_id)
            for other_atom_id in range(self.nhatoms())
        )

    def unrepeated_atom_list(self):
        for atom_id in range(self.nhatoms()):
            self.check_equivalence_class([atom_id])
        return self.equiv_class_examples(1, as_tuples=False)

    # Coordination number including unconnected electronic pairs. TO-DO: make sure it does not count pairs that contribute to an aromatic system?
    def effective_coordination_number(self, hatom_id):
        pairs = 0
        hatom = self.hatoms[hatom_id]
        ncharge = hatom.ncharge
        if ncharge in unshared_pairs:
            cur_dict = unshared_pairs[ncharge]
            valence = hatom.valence
            if valence in cur_dict:
                pairs = cur_dict[valence]
        return self.coordination_number(hatom_id) + pairs

    # Hybridization of heavy atom hatom_id.
    def hybridization(self, hatom_id):
        return coord_num_hybrid[self.effective_coordination_number(hatom_id)]

    def is_connected(self):
        return self.num_connected() == 1

    # Number of connected molecules.
    def num_connected(self):
        return len(self.graph.components())

    # Order of bond between atoms atom_id1 and atom_id2
    def bond_order(self, atom_id1, atom_id2):
        stuple = sorted_tuple(atom_id1, atom_id2)
        if stuple in self.bond_orders:
            return self.bond_orders[stuple]
        else:
            return 0

    def bond_order_float(self, atom_id1, atom_id2):
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

    def aa_all_bond_orders(self, atom_id1, atom_id2, unsorted=False):
        self.init_resonance_structures()

        stuple = sorted_tuple(atom_id1, atom_id2)
        if stuple in self.bond_orders:
            if stuple in self.resonance_structure_map:
                res_struct_ords = self.resonance_structure_orders[
                    self.resonance_structure_map[stuple]
                ]
                output = []
                for res_struct_ord in res_struct_ords:
                    if stuple in res_struct_ord:
                        new_bond_order = res_struct_ord[stuple] + 1
                    else:
                        new_bond_order = 1
                    if unsorted or (new_bond_order not in output):
                        output.append(new_bond_order)
                if unsorted:
                    return output
                else:
                    return sorted(output)
            else:
                return [self.bond_orders[stuple]]
        else:
            return [0]

    def nhatoms(self):
        """
        Number of heavy atoms.
        """
        return self.graph.vcount()

    def tot_nhydrogens(self):
        """
        Total number of hydrogens.
        """
        return sum(hatom.nhydrogens for hatom in self.hatoms)

    def natoms(self):
        """
        Total number of atoms.
        """
        return self.tot_nhydrogens() + self.nhatoms()

    def tot_ncovpairs(self):
        """
        Total number of binding electron pairs.
        """
        return self.tot_nhydrogens() + sum(self.bond_orders.values())

    # Dirty inheritance:
    def neighbors(self, hatom_id):
        return self.graph.neighbors(hatom_id)

    def num_neighbors(self, hatom_id):
        return self.graph.neighborhood_size(vertices=hatom_id, order=1) - 1

    # Basic commands for managing the graph.
    def set_edge_order(self, atom1, atom2, new_edge_order):
        true_bond_tuple = tuple(sorted_tuple(atom1, atom2))
        if new_edge_order < 0:
            raise InvalidChange
        if new_edge_order == 0:
            self.graph.delete_edges([true_bond_tuple])
            del self.bond_orders[true_bond_tuple]
        else:
            self.bond_orders[true_bond_tuple] = new_edge_order

    def change_edge_order(self, atom1, atom2, change=0):
        if change != 0:
            if atom1 == atom2:
                raise InvalidChange
            true_bond_tuple = tuple(sorted_tuple(atom1, atom2))
            try:
                cur_edge_order = self.bond_orders[true_bond_tuple]
            except KeyError:
                cur_edge_order = 0
                self.graph.add_edge(*true_bond_tuple)
            new_edge_order = cur_edge_order + change
            if new_edge_order < 0:
                raise InvalidChange
            self.set_edge_order(atom1, atom2, new_edge_order)

    def change_hydrogen_number(self, atom_id, hydrogen_number_change):
        self.hatoms[atom_id].nhydrogens += hydrogen_number_change
        self.hatoms[atom_id].changed()
        if self.hatoms[atom_id].nhydrogens < 0:
            raise InvalidChange

    def valence_config_valid(self, checked_valences):
        """
        Check whether heavy atom valences provided in checked_valences array are valid for some resonance structure.
        """
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

    def gen_coord_nums_extra_valence_ids(self, initialized_valences=True):
        coordination_numbers = []
        extra_valence_indices = []
        for hatom_id, hatom in enumerate(self.hatoms):
            cur_coord_number = self.coordination_number(hatom_id)
            max_valence = hatom.max_valence()
            if max_valence < cur_coord_number:
                raise InvalidAdjMat
            elif max_valence > cur_coord_number:
                if initialized_valences:
                    if hatom.possible_valences is None:
                        if hatom.valence == cur_coord_number:
                            continue
                coordination_numbers.append(cur_coord_number)
                extra_valence_indices.append(hatom_id)
            if not initialized_valences:
                hatom.valence = hatom.smallest_valid_valence(cur_coord_number)
        return coordination_numbers, extra_valence_indices

    def prelim_nonsigma_bonds(self):
        # Set all bond orders to one.
        for bond_tuple, bond_order in self.bond_orders.items():
            if bond_order > 1:
                self.change_edge_order(*bond_tuple, 1 - bond_order)
        # Find indices of atoms with spare non-sigma electrons. Also check coordination numbers are not above valence.
        (
            coordination_numbers,
            extra_valence_indices,
        ) = self.gen_coord_nums_extra_valence_ids(initialized_valences=False)

        if len(extra_valence_indices) == 0:  # no non-sigma bonds to reassign
            return
        (
            extra_val_ids_lists,
            coord_nums_lists,
            extra_val_subgraph_list,
        ) = self.extra_valence_subgraphs(extra_valence_indices, coordination_numbers)
        for extra_val_ids, coord_nums, extra_val_subgraph in zip(
            extra_val_ids_lists, coord_nums_lists, extra_val_subgraph_list
        ):
            self.reassign_nonsigma_bonds_subgraph(
                extra_val_ids, coord_nums, extra_val_subgraph
            )

    def extra_valence_subgraphs(self, extra_valence_indices, coordination_numbers):
        total_subgraph = self.graph.induced_subgraph(extra_valence_indices)
        ts_components = total_subgraph.components()
        members = ts_components.membership
        extra_val_subgraph_list = ts_components.subgraphs()
        extra_val_ids_lists = sorted_by_membership(members, extra_valence_indices)
        coord_nums_lists = sorted_by_membership(members, coordination_numbers)
        return extra_val_ids_lists, coord_nums_lists, extra_val_subgraph_list

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
                    self.hatoms[hatom_id].valence = self.hatoms[
                        hatom_id
                    ].possible_valences[valence_option_id]

    def single_atom_resonance_structure(self, hatom_id):
        """Which resonance structure region contains hatom_id."""
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

    # TODO: Maybe expand to include charge transfer?
    def init_resonance_structures(self):
        self.prelim_nonsigma_bonds()
        if (
            (self.resonance_structure_orders is not None)
            and (self.resonance_structure_valence_vals is not None)
            and (self.resonance_structure_map is not None)
            and (self.resonance_structure_inverse_map is not None)
        ):
            return

        self.resonance_structure_orders = []
        self.resonance_structure_valence_vals = []
        self.resonance_structure_map = {}
        self.resonance_structure_inverse_map = []

        (
            coordination_numbers,
            extra_valence_indices,
        ) = self.gen_coord_nums_extra_valence_ids()
        if len(extra_valence_indices) == 0:
            return
        (
            extra_val_ids_lists,
            coord_nums_lists,
            extra_val_subgraph_list,
        ) = self.extra_valence_subgraphs(extra_valence_indices, coordination_numbers)

        for extra_val_ids, coord_nums, extra_val_subgraph in zip(
            extra_val_ids_lists, coord_nums_lists, extra_val_subgraph_list
        ):
            cur_resonance_region_id = len(self.resonance_structure_orders)
            for i, val_id1 in enumerate(extra_val_ids):
                neighs = self.neighbors(val_id1)
                for val_id2 in extra_val_ids[:i]:
                    if val_id2 in neighs:
                        self.resonance_structure_map[
                            (val_id2, val_id1)
                        ] = cur_resonance_region_id
            num_valence_options, def_val_option = self.valence_change_range(
                extra_val_ids
            )
            if num_valence_options is None:
                valence_options = [None]
            else:
                valence_options = range(num_valence_options)
            tot_subgraph_res_struct = []
            tot_valence_vals = []
            for valence_option in valence_options:
                self.change_valence_option(extra_val_ids, valence_option)
                subgraph_res_struct_list = self.complete_valences_attempt(
                    extra_val_ids,
                    coord_nums,
                    extra_val_subgraph,
                    all_possibilities=True,
                )
                if subgraph_res_struct_list is None:
                    raise InvalidAdjMat
                tot_subgraph_res_struct += subgraph_res_struct_list
                for _ in range(len(subgraph_res_struct_list)):
                    tot_valence_vals.append(valence_option)

            self.resonance_structure_valence_vals.append(tot_valence_vals)
            self.resonance_structure_orders.append(tot_subgraph_res_struct)
            self.resonance_structure_inverse_map.append(extra_val_ids)
            # Change valence states of extra_val_ids to the default option that corresponds to the bond orders assigned through the bond_orders dictionnary.
            if def_val_option is not None:
                self.change_valence_option(extra_val_ids, def_val_option)

    def added_edges_list_to_dict(self, added_edges):
        add_bond_orders = {}
        for e in added_edges:
            se = sorted_tuple(*e)
            if se in add_bond_orders:
                add_bond_orders[se] += 1
                if add_bond_orders[se] == max_bo(
                    self.hatoms[se[0]], self.hatoms[se[1]]
                ):
                    return None
            else:
                add_bond_orders[se] = 1
        return add_bond_orders

    def possible_subgraph_heavy_atom_valences(self, extra_val_ids, coord_nums):
        """
        An auxiliary procedure that checks which valence values can be iterated over for a subset of the chemical graph
        and then sorts the valences lists to be iterated over in order of relevance.
        The order of relevance is determined by: 1. Sum of valence numbers. 2. For two valence configurations with the same sum, we pick the one with
        smallest
        """
        HeavyAtomValenceIterators = []
        IteratedValenceIds = []
        for hatom_id, coord_num in zip(extra_val_ids, coord_nums):
            self.hatoms[hatom_id].possible_valences = None
            needed_val_id = self.hatoms[hatom_id].smallest_valid_valence(
                coord_num, True
            )
            if needed_val_id != -1:
                HeavyAtomValenceIterators.append(
                    iter(self.hatoms[hatom_id].avail_val_list()[needed_val_id:])
                )
                IteratedValenceIds.append(hatom_id)
        HeavyAtomValences = list(itertools.product(*HeavyAtomValenceIterators))
        HeavyAtomValences.sort(key=lambda x: ValenceConfigurationCharacter(x))

        return HeavyAtomValences, IteratedValenceIds

    def reassign_nonsigma_bonds_subgraph(
        self, extra_val_ids, coord_nums, extra_val_subgraph
    ):
        (
            HeavyAtomValences,
            IteratedValenceIds,
        ) = self.possible_subgraph_heavy_atom_valences(extra_val_ids, coord_nums)
        min_found_vcc = None
        for HeavyAtomValencesList in HeavyAtomValences:
            if min_found_vcc is not None:
                if (
                    ValenceConfigurationCharacter(HeavyAtomValencesList)
                    != min_found_vcc
                ):
                    break
            # Assign all heavy atoms their current valences.
            for ha_id, ha_val in zip(IteratedValenceIds, HeavyAtomValencesList):
                self.hatoms[ha_id].valence = ha_val
            subgraph_resonance_struct = self.complete_valences_attempt(
                extra_val_ids, coord_nums, extra_val_subgraph
            )
            if subgraph_resonance_struct is not None:
                if min_found_vcc is None:
                    min_found_vcc = ValenceConfigurationCharacter(HeavyAtomValencesList)
                    saved_subgraph_resonance_struct = subgraph_resonance_struct
                    saved_heavy_atom_valences_list = HeavyAtomValencesList
                for ha_id, ha_val in zip(IteratedValenceIds, HeavyAtomValencesList):
                    if self.hatoms[ha_id].possible_valences is None:
                        self.hatoms[ha_id].possible_valences = []
                    self.hatoms[ha_id].possible_valences.append(ha_val)
        if min_found_vcc is None:
            print("Invalid molecule:")
            print("HeavyAtom list:", self.hatoms)
            print("Graph:", self.graph)
            raise InvalidAdjMat
        for ha_id, ha_val in zip(IteratedValenceIds, saved_heavy_atom_valences_list):
            ha = self.hatoms[ha_id]
            ha.valence = ha_val
            poss_vals = ha.possible_valences
            if len(poss_vals) != 1:
                if any(val != poss_vals[0] for val in poss_vals[1:]):
                    continue
            ha.possible_valences = None
        # Initialized bonds according to the last considered set of added edges. (The valences are initialized already.)
        for bond_tuple, bond_added_order in saved_subgraph_resonance_struct.items():
            self.change_edge_order(*bond_tuple, bond_added_order)

    def complete_valences_attempt(
        self, extra_val_ids, coord_nums, extra_val_subgraph, all_possibilities=False
    ):
        """
        For a subgraph of the chemical graph extra_val_subgraph which spawns over hatoms with ids extra_val_ids with
        coordination numbers coord_nums, generate assignment of all non-sigma valence electrons into nonsigma bonds.
        If all_possibilities is False returns one such resonance structure, otherwise enumerate and return all resonance structures.
        """

        output = None
        added_edges = []
        connection_opportunities = np.zeros(len(extra_val_ids), dtype=int)
        extra_valences = np.zeros(len(extra_val_ids), dtype=int)
        for i, (eval_id, coord_num) in enumerate(zip(extra_val_ids, coord_nums)):
            extra_valences[i] = self.hatoms[eval_id].valence - coord_num
        # TODO is it needed?
        if np.all(extra_valences == 0):
            return {}
        # Check how many neighboring atoms a given atom can be connected to with the nonsigma bonds.
        for cur_id, extra_valence in enumerate(extra_valences):
            if extra_valence != 0:
                neighs = extra_val_subgraph.neighbors(cur_id)
                for neigh in neighs:
                    if extra_valences[neigh] != 0:
                        connection_opportunities[cur_id] += 1
                if connection_opportunities[cur_id] == 0:
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
                closed_atom = np.where(connection_opportunities == min_connectivity)[0][
                    0
                ]
                potential_other_atoms = possible_closed_pairs(
                    closed_atom, extra_valences, extra_val_subgraph
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
                    connection_opportunities[:] = saved_connection_opportunities[
                        cur_decision_fork
                    ][:]
                    potential_other_atoms = copy.deepcopy(
                        saved_potential_other_atoms[cur_decision_fork]
                    )
                    closed_atom = saved_closed_atom[cur_decision_fork]
                    del added_edges[added_edges_stops[cur_decision_fork] :]
                else:
                    path_taken[cur_decision_fork] = 0
                    saved_extra_valences[cur_decision_fork] = np.copy(extra_valences)
                    saved_connection_opportunities[cur_decision_fork] = np.copy(
                        connection_opportunities
                    )
                    saved_potential_other_atoms[cur_decision_fork] = copy.deepcopy(
                        potential_other_atoms
                    )
                    saved_closed_atom[cur_decision_fork] = closed_atom
                    added_edges_stops[cur_decision_fork] = len(added_edges)
                choice = path_taken[cur_decision_fork]
            other_closed_atom = potential_other_atoms[choice]

            # Add the extra nonsigma bond.
            added_edges.append(
                (extra_val_ids[closed_atom], extra_val_ids[other_closed_atom])
            )
            # Delete the now tied valence atoms.
            for cur_id in [closed_atom, other_closed_atom]:
                extra_valences[cur_id] -= 1
                if extra_valences[cur_id] == 0:
                    connection_opportunities[cur_id] = 0
                    # Neighbors of an atom with "spent" extra valence electrons can no longer connect to it.
                    for neigh_id in extra_val_subgraph.neighbors(cur_id):
                        if connection_opportunities[neigh_id] != 0:
                            connection_opportunities[neigh_id] -= 1
            if np.all(extra_valences == 0):
                added_bonds_dict = self.added_edges_list_to_dict(added_edges)
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

    # More sophisticated commands that are to be called in the "modify" module.
    def change_bond_order(
        self, atom1, atom2, bond_order_change, resonance_structure_id=None
    ):
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

    def adjust_resonance_valences(
        self, resonance_structure_region, resonance_structure_id
    ):
        self.init_resonance_structures()
        changed_hatom_ids = self.resonance_structure_inverse_map[
            resonance_structure_region
        ]
        cur_resonance_struct_orders = self.resonance_structure_orders[
            resonance_structure_region
        ][resonance_structure_id]
        new_valence_option = self.resonance_structure_valence_vals[
            resonance_structure_region
        ][resonance_structure_id]
        self.change_valence_option(changed_hatom_ids, new_valence_option)
        for hatom_considered_num, hatom_id2 in enumerate(changed_hatom_ids):
            hatom2_neighbors = self.neighbors(hatom_id2)
            for hatom_id1 in changed_hatom_ids[:hatom_considered_num]:
                if hatom_id1 in hatom2_neighbors:
                    stuple = sorted_tuple(hatom_id1, hatom_id2)
                    if stuple in cur_resonance_struct_orders:
                        self.set_edge_order(
                            hatom_id1,
                            hatom_id2,
                            1 + cur_resonance_struct_orders[stuple],
                        )
                    else:
                        self.set_edge_order(hatom_id1, hatom_id2, 1)

    def adjust_resonance_valences_atom(
        self, atom_id, resonance_structure_id=None, valence_option_id=None
    ):
        if (resonance_structure_id is not None) or (valence_option_id is not None):
            adjusted_resonance_region = self.single_atom_resonance_structure(atom_id)
            if resonance_structure_id is None:
                resonance_structure_id = self.resonance_structure_valence_vals[
                    adjusted_resonance_region
                ].index(valence_option_id)
            self.adjust_resonance_valences(
                adjusted_resonance_region, resonance_structure_id
            )

    # TODO Do we need resonance structure invariance here?
    def remove_heavy_atom(self, atom_id, resonance_structure_id=None):
        self.adjust_resonance_valences_atom(
            atom_id, resonance_structure_id=resonance_structure_id
        )

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
                new_valence = None
            else:
                new_valence = new_chain_atom_valences[chain_id]
            self.hatoms.append(HeavyAtom(new_chain_atom, valence=new_valence))
            self.hatoms[-1].nhydrogens = self.hatoms[-1].valence

        # TODO find a way to avoid using *.changed() several times?
        self.changed()

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
        self.change_hydrogen_number(
            modified_atom_id, new_valence - self.hatoms[modified_atom_id].valence
        )
        self.hatoms[modified_atom_id].valence = new_valence

        self.changed()

    # Output properties that include hydrogens.

    def full_ncharges(self):
        output = np.ones(self.natoms(), dtype=int)
        for ha_id, ha in enumerate(self.hatoms):
            output[ha_id] = ha.ncharge
        return output

    def full_adjmat(self):
        natoms = self.natoms()
        output = np.zeros((natoms, natoms), dtype=int)
        for bond_tuple, bond_order in self.bond_orders.items():
            output[bond_tuple] = bond_order
            output[bond_tuple[::-1]] = bond_order
        cur_h_id = self.nhatoms()
        for ha_id, ha in enumerate(self.hatoms):
            for _ in range(ha.nhydrogens):
                output[ha_id, cur_h_id] = 1
                output[cur_h_id, ha_id] = 1
                cur_h_id += 1
        return output

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
                av_res_struct_dict[edge] = float(av_res_struct_dict[edge]) / len(
                    res_struct_dict
                )
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

    # Procedures used for sorting.
    def init_colors(self):
        if self.colors is None:
            self.colors = list2colors(self.hatoms)

    def init_canonical_permutation(self):
        if self.canonical_permutation is None:
            self.init_colors()
            self.canonical_permutation = (
                self.nhatoms()
                - 1
                - np.array(self.graph.canonical_permutation(color=self.colors))
            )
            self.inv_canonical_permutation = np.zeros(self.nhatoms(), dtype=int)
            for pos_counter, pos in enumerate(self.canonical_permutation):
                self.inv_canonical_permutation[pos] = pos_counter

    def get_inv_canonical_permutation(self):
        self.init_canonical_permutation()
        return self.inv_canonical_permutation

    def get_comparison_list(self):
        if self.comparison_list is None:
            self.init_canonical_permutation()
            self.comparison_list = []
            for perm_hatom_id, hatom_id in enumerate(self.inv_canonical_permutation):
                self.comparison_list += self.hatoms[hatom_id].get_comparison_list()
                perm_neighs = []
                for neigh_id in self.neighbors(hatom_id):
                    perm_id = self.canonical_permutation[neigh_id]
                    if perm_id > perm_hatom_id:
                        perm_neighs.append(perm_id)
                self.comparison_list += sorted(perm_neighs)
                self.comparison_list.append(len(perm_neighs))
        return self.comparison_list

    def copy_extra_data_to(self, other_cg, linear_storage=False):
        self.copy_equivalence_info_to(other_cg, linear_storage=linear_storage)

    def copy_equivalence_info_to(self, other_cg, linear_storage=False):
        if linear_storage:
            atom_set_length_ubound = 2
        else:
            atom_set_length_ubound = 3
        for atom_set_length in range(1, atom_set_length_ubound):
            if self.equiv_arr(atom_set_length) is None:
                continue
            else:
                other_cg.init_equivalence_array(atom_set_length)

            num_equiv_classes = self.num_equiv_classes(atom_set_length)

            other_num_equiv_classes = other_cg.num_equiv_classes(atom_set_length)

            copied_equivalence_classes = {}

            for other_equiv_class_id in range(other_num_equiv_classes):
                other_cg_equiv_class_members = other_cg.equiv_class_members(
                    other_equiv_class_id, atom_set_length
                )
                equiv_class_members = other_cg.match_atom_id_sets(
                    other_cg_equiv_class_members, self
                )
                for equiv_class_member in equiv_class_members:
                    equiv_class_id = self.equiv_arr(atom_set_length)[equiv_class_member]
                    if equiv_class_id != -1:
                        copied_equivalence_classes[
                            equiv_class_id
                        ] = other_equiv_class_id
                        break

            for equiv_class_id in range(num_equiv_classes):
                equiv_class_members = self.equiv_class_members(
                    equiv_class_id, atom_set_length
                )
                other_cg_equiv_class_members = self.match_atom_id_sets(
                    equiv_class_members, other_cg
                )
                if equiv_class_id in copied_equivalence_classes:
                    assigned_equiv_class_id = copied_equivalence_classes[equiv_class_id]
                else:
                    assigned_equiv_class_id = other_num_equiv_classes
                    other_num_equiv_classes += 1
                for other_cg_equiv_class_member in other_cg_equiv_class_members:
                    other_cg.assign_equivalence_class(
                        other_cg_equiv_class_member, assigned_equiv_class_id
                    )

    def match_atom_id_set(self, atom_id_set, other_cg):
        return tuple(
            other_cg.inv_canonical_permutation[self.canonical_permutation[atom_id]]
            for atom_id in atom_id_set
        )

    def match_atom_id_sets(self, atom_id_sets, other_cg):
        return [
            self.match_atom_id_set(atom_id_set, other_cg)
            for atom_id_set in atom_id_sets
        ]

    def canonical_atom_set_iterator(self, atom_set_length):
        return itertools.product(
            *[self.inv_canonical_permutation for _ in range(atom_set_length)]
        )

    def __hash__(self):
        # TODO replaced comparison_list with comparison_tuple?
        return hash(tuple(self.get_comparison_list()))

    def __lt__(self, ch2):
        return self.get_comparison_list() < ch2.get_comparison_list()

    def __gt__(self, ch2):
        return self.get_comparison_list() > ch2.get_comparison_list()

    def __eq__(self, ch2):
        return self.get_comparison_list() == ch2.get_comparison_list()

    def __str__(self):
        """
        The representation consists of substrings corresponding to each heavy atom in canonical ordering and connected by ":".
        Each of those substrings starts with the nuclear charge of the atom, continues after "#" with the number of hydrogens connected to
        the heavy atom (if not 0), and finishes with list of other atom indices preceded by "@" to which the current atom is connected and
        whose indices exceed the current atom's index.

        See examples/chemxpl/chemxpl_str_representation for examples of how it works.
        """
        self.init_canonical_permutation()
        hatom_strings = []

        for hatom_canon_id, hatom_id in enumerate(self.inv_canonical_permutation):
            canonical_neighbor_list = []
            for neigh_id in self.neighbors(hatom_id):
                canonical_neighbor_id = self.canonical_permutation[neigh_id]
                if canonical_neighbor_id > hatom_canon_id:
                    canonical_neighbor_list.append(canonical_neighbor_id)
            hatom_string = str(self.hatoms[hatom_id].ncharge)
            nhydrogens = self.hatoms[hatom_id].nhydrogens
            if nhydrogens != 0:
                hatom_string += "#" + str(nhydrogens)
            if len(canonical_neighbor_list) != 0:
                canonical_neighbor_list.sort()
                hatom_string += "@" + "@".join(
                    str(cn_id) for cn_id in canonical_neighbor_list
                )
            hatom_strings.append(hatom_string)
        return ":".join(hatom_strings)

    def __repr__(self):
        return str(self)


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
    new_cg = ChemGraph(nuclear_charges=new_nuclear_charges, adj_mat=new_adj_mat)
    if new_cg != cg:
        raise Exception
    return new_cg


def chemgraph_str2unchecked_adjmat_ncharges(input_string: str) -> tuple:
    """
    Converts a ChemGraph string representation into the adjacency matrix (with all bond orders set to one) and nuclear charges.
    input_string : string to be converted
    """
    hatom_ncharges = []
    hydrogen_nums = []
    hatom_neighbors = []
    for hatom_str in input_string.split(":"):
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

    return adj_mat, nuclear_charges


def str2ChemGraph(input_string: str, shuffle=False) -> ChemGraph:
    """
    Converts a ChemGraph string representation into a ChemGraph object.
    input_string : string to be converted
    shuffle : whether atom positions should be shuffled, introduced for testing purposes.
    """
    unchecked_adjmat, ncharges = chemgraph_str2unchecked_adjmat_ncharges(input_string)
    if shuffle:  # TODO should I use shuffled_chemgraph here?
        ids = list(range(len(ncharges)))
        random.shuffle(ids)
        ncharges = ncharges[ids]
        unchecked_adjmat = unchecked_adjmat[ids][:, ids]
    return ChemGraph(nuclear_charges=ncharges, adj_mat=unchecked_adjmat)


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
    return ChemGraph(nuclear_charges=shuffled_ncharges, adj_mat=shuffled_adjmat)


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
                    new_bond_positions_orders_dict_list[mv1][internal_id1].append(
                        cur_bond_order
                    )
                else:
                    new_bond_positions_orders_dict_list[mv1][internal_id1] = [
                        cur_bond_order
                    ]

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
        new_cg = ChemGraph(graph=graph, hatoms=hatoms, bond_orders=bond_orders)

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


def combine_chemgraphs(cg1, cg2):
    new_hatoms = copy.deepcopy(cg1.hatoms + cg2.hatoms)
    new_graph = disjoint_union([cg1.graph, cg2.graph])
    id2_shift = cg1.nhatoms()
    new_bond_orders = copy.deepcopy(cg1.bond_orders)
    for bond_tuple, bond_order in cg2.bond_orders.items():
        new_bond_tuple = tuple(np.array(bond_tuple) + id2_shift)
        new_bond_orders[new_bond_tuple] = bond_order
    return ChemGraph(hatoms=new_hatoms, bond_orders=new_bond_orders, graph=new_graph)


def connection_forbidden(nc1, nc2, forbidden_bonds):
    if (nc1 is None) or (nc2 is None) or (forbidden_bonds is None):
        return False
    nc_tuple = sorted_tuple(int_atom_checked(nc1), int_atom_checked(nc2))
    return nc_tuple in forbidden_bonds


def h2chemgraph():
    """
    ChemGraph representation of the H2 molecule.
    """
    # Very dirty, but there seems to be no better way.
    output = ChemGraph(nuclear_charges=[9, 9], adj_mat=[[0, 1], [1, 0]])
    for i in range(2):
        output.hatoms[i].ncharge = 1
    return output


#   Utility functions
def possible_closed_pairs(closed_atom, extra_valences, extra_val_subgraph):
    output = []
    for i in extra_val_subgraph.neighbors(closed_atom):
        if extra_valences[i] != 0:
            output.append(i)
    return output
