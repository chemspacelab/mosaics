# TODO Currently, assigning equivalence classes to all nodes scales as O(nhatoms**2). I think that smarter coding can bring that down to O(log(nhatoms)), but for now we have more important methodological problems.

# TODO Should be possible to optimize further by comparing HeavyAtom neighborhood representations instead of colors (should decrease the number of canonical permutation evaluations during simulation).

import itertools
import operator

import numpy as np
from igraph import Graph

from ..misc_procedures import (
    VERBOSITY,
    VERBOSITY_MUTED,
    InvalidAdjMat,
    list2colors,
    permutation_inverse,
    set_verbosity,
    sorted_tuple,
)
from ..periodic import coord_num_hybrid, unshared_pairs
from .heavy_atom import HeavyAtom, default_valence


class InvalidChange(Exception):
    pass


# Introduced in case we, for example, started to consider F as a default addition instead.
DEFAULT_ELEMENT = 1

# Dummy equivalence class preliminarly assigned to hatoms and pairs on hatoms.
unassigned_equivalence_class_id = -1

# Choice of algorithm for isomorphism determination.
bliss = "BLISS"
vf2 = "VF2"
isomorphism_algorithm = vf2
available_isomorphism_algorithms = [bliss, vf2]


def set_isomorphism_algorithm(new_isomorphism_algorithm: str):
    """
    Set algorithm for finding isomorphisms (VF2 or BLISS).
    """
    global isomorphism_algorithm
    assert new_isomorphism_algorithm in available_isomorphism_algorithms
    isomorphism_algorithm = new_isomorphism_algorithm


# How large is the neighborhood of a HeavyAtom used to define its color.
color_defining_neighborhood_radius = 0


def set_color_defining_neighborhood_radius(new_color_defining_neighborhood_radius):
    """
    Set how large is the radius of the neighborhood of a HeavyAtom used to generate its integer list representation.
    The latter is used to determine whether two HeavyAtom objects inside ChemGraph should be assigned the same color.
    Larger neighborhoods are more expensive to process, but also decrease how often isomorphism algorithms used here
    will have to deal with similarly-colored nodes. Optimal color_defining_neighborhood_radius value is probably
    application-dependent.
    """
    global color_defining_neighborhood_radius
    color_defining_neighborhood_radius = new_color_defining_neighborhood_radius


using_two_level_comparison = True


def set_using_two_level_comparison(new_using_two_level_comparison: bool):
    """
    Use ChemGraph comparison with "trivial" and "complex" colors or not.
    """
    global using_two_level_comparison
    using_two_level_comparison = new_using_two_level_comparison


def set_misc_global_variables(
    isomorphism_algorithm=vf2,
    color_defining_neighborhood_radius=0,
    using_two_level_comparison=True,
    VERBOSITY=VERBOSITY_MUTED,
):
    """
    Set all global variables in the module.
    """
    set_isomorphism_algorithm(isomorphism_algorithm)
    set_color_defining_neighborhood_radius(color_defining_neighborhood_radius)
    set_using_two_level_comparison(using_two_level_comparison)
    set_verbosity(VERBOSITY)


def misc_global_variables_current_kwargs():
    return {
        "isomorphism_algorithm": isomorphism_algorithm,
        "color_defining_neighborhood_radius": color_defining_neighborhood_radius,
        "using_two_level_comparison": using_two_level_comparison,
        "VERBOSITY": VERBOSITY,
    }


def canonical_permutation_with_inverse(graph: Graph, colors):
    """
    Return canonical permutation in terms of both forward and inverse arrays.
    """
    inv_canonical_permutation = np.array(graph.canonical_permutation(color=colors))
    canonical_permutation = permutation_inverse(inv_canonical_permutation)
    return canonical_permutation, inv_canonical_permutation


def ha_graph_comparison_list(
    graph,
    ha_trivial_comparison_lists,
    canonical_permutation,
    inv_canonical_permutation,
    charge=None,
):
    """
    Create an integer list uniquely representing a graph with HeavyAtom objects as nodes with a known canonical permutation.
    Used to define instances of ChemGraph along with node neighborhoods.
    """
    comparison_list = [len(ha_trivial_comparison_lists)]
    for perm_hatom_id, hatom_id in enumerate(inv_canonical_permutation):
        comparison_list += list(ha_trivial_comparison_lists[hatom_id])
        perm_neighs = []
        for neigh_id in graph.neighbors(hatom_id):
            perm_id = canonical_permutation[neigh_id]
            if perm_id > perm_hatom_id:
                perm_neighs.append(perm_id)
        comparison_list.append(len(perm_neighs))
        comparison_list += sorted(perm_neighs)
    if charge is not None:
        comparison_list.append(charge)
    return comparison_list


# Auxiliary functions
def adj_mat2bond_list(adj_mat):
    bond_list = []
    for atom1, adj_mat_row in enumerate(adj_mat):
        for atom2, adj_mat_val in enumerate(adj_mat_row[:atom1]):
            if adj_mat_val != 0:
                bond_list.append((atom2, atom1))
    return bond_list


# TODO perhaps used SortedDict from sortedcontainers more here?
class BaseChemGraph:
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
            bond_list (dict or None): bond orders between all atoms (for the initialized resonance structure).
            adj_mat (numpy.array or None): adjacency matrix between all atoms (including hydrogens).
            nuclear_charges (numpy.array or None): all nuclear charges (including hydrogens).
        """

        self.graph = graph
        self.hatoms = hatoms
        self.bond_orders = bond_orders
        self.charge = charge

        if (self.graph is None) and (self.hatoms is None):
            self.init_graph_natoms(np.array(nuclear_charges), bond_list=bond_list, adj_mat=adj_mat)
        self.changed()

    def init_graph_natoms(self, nuclear_charges, bond_list=None, adj_mat=None):
        if bond_list is None:
            assert adj_mat is not None
            bond_list = adj_mat2bond_list(adj_mat)
        self.hatoms = []
        heavy_atom_dict = {}
        for atom_id, ncharge in enumerate(nuclear_charges):
            if ncharge != DEFAULT_ELEMENT:
                heavy_atom_dict[atom_id] = len(self.hatoms)
                self.hatoms.append(HeavyAtom(ncharge))
        self.graph = Graph(n=len(self.hatoms), directed=False)
        for ha_id1, ha_id2 in bond_list:
            if ha_id1 in heavy_atom_dict:
                true_id = heavy_atom_dict[ha_id1]
                if ha_id2 in heavy_atom_dict:
                    self.graph.add_edge(true_id, heavy_atom_dict[ha_id2])
                    continue
            else:
                if ha_id2 not in heavy_atom_dict:
                    # this means the graph contains molecular hydrogen; the code was not designed for that.
                    raise InvalidAdjMat("Contains molecular hydrogen.")
                assert ha_id2 in heavy_atom_dict
                true_id = heavy_atom_dict[ha_id2]
            self.hatoms[true_id].nhydrogens += 1

    def changed(self):
        """
        Initialize (or re-initialize) temporary data.
        """
        # Canonical ordering of HeavyAtom members.
        self.canonical_permutation = None
        self.inv_canonical_permutation = None
        # List storing neighborhood-based representations of atoms.
        self.ha_comparison_lists = [None for _ in range(self.nhatoms())]
        # Colors calculated from self.ha_comparison_lists.
        self.colors = None
        # HeavyAtom object representations.
        self.ha_trivial_comparison_lists = [ha.get_comparison_list() for ha in self.hatoms]
        # Colors calculated from ha_trivial_comparison_lists.
        self.trivial_colors = list2colors(self.ha_trivial_comparison_lists)

        # Related to storing equivalence classes of HeavyAtom members and their pairs.
        # TODO some of these lines might be excessive.
        self.equivalence_vector = None
        self.pair_equivalence_matrix = None
        # Information on resonance structures.
        self.resonance_structure_orders = None
        self.resonance_structure_map = None
        self.resonance_structure_inverse_map = None
        self.resonance_structure_valence_vals = None
        self.resonance_structure_charge_feasibilities = None
        self.overall_charge_feasibility = None
        self.resonance_structures_merged = None
        # Might be excessive?
        for ha in self.hatoms:
            ha.possible_valences = None
            ha.possible_charges = None

        # Comparison list based on stochimetry of different HeavyAtom neighborhoods.
        self.stochiometry_comparison_list = None
        # Comparison list based on proper canonical ordering.
        self.comparison_list = None
        # Number of automorphisms.
        self.log_permutation_factor = None
        # Temporary color arrays used to check vertex equivalence.
        self.temp_colors1 = None
        self.temp_colors2 = None

    def init_ha_comparison_list(self, central_ha_id):
        """
        Check that a neighborhood-based representation has been assigned to HeavyAtom with index ha_id.
        """
        if self.ha_comparison_lists[central_ha_id] is not None:
            return
        if color_defining_neighborhood_radius == 0:
            self.ha_comparison_lists[central_ha_id] = self.ha_trivial_comparison_lists[
                central_ha_id
            ]
            return
        # Isolate neighborhood of interest.
        neighborhood_ids = sorted(
            self.graph.neighborhood(central_ha_id, color_defining_neighborhood_radius)
        )
        neighborhood_subgraph = self.graph.subgraph(neighborhood_ids)
        # Initiate colors of the neighborhood.
        neigh_trivial_comparison_lists = []
        central_ha_id_subgraph = 0
        for subgraph_hatom_id, hatom_id in enumerate(neighborhood_ids):
            neigh_trivial_comparison_lists.append(self.ha_trivial_comparison_lists[hatom_id])
            if hatom_id == central_ha_id:
                central_ha_id_subgraph = subgraph_hatom_id
        neigh_trivial_colors = list2colors(neigh_trivial_comparison_lists)
        neigh_trivial_colors[central_ha_id_subgraph] = max(neigh_trivial_colors) + 1
        # Get canonical permutation.
        (
            neigh_canonical_permutation,
            neigh_inv_canonical_permutation,
        ) = canonical_permutation_with_inverse(neighborhood_subgraph, neigh_trivial_colors)
        # Use canonical permutation to create comparison list the same way it is done for the global ChemGraph.
        self.ha_comparison_lists[central_ha_id] = ha_graph_comparison_list(
            neighborhood_subgraph,
            neigh_trivial_comparison_lists,
            neigh_canonical_permutation,
            neigh_inv_canonical_permutation,
        )
        # TODO Not %100 this line is necessary
        self.ha_comparison_lists[central_ha_id].append(
            neigh_canonical_permutation[central_ha_id_subgraph]
        )

    def init_all_ha_comparison_lists(self):
        """
        Check all neighborhood-based representations are initialized.
        """
        for ha_id in range(self.nhatoms()):
            self.init_ha_comparison_list(ha_id)

    def reinit_temp_colors(self):
        self.temp_colors1 = np.copy(self.colors)
        self.temp_colors2 = np.copy(self.colors)

    def init_colors(self):
        """
        Initialize 'proper' colors based on neighborhoods of HeavyAtom instances.
        """
        if self.colors is not None:
            return
        self.init_all_ha_comparison_lists()

        self.colors = list2colors(self.ha_comparison_lists)
        self.reinit_temp_colors()

    def overwrite_colors(self, new_colors):
        assert len(new_colors) == self.nhatoms()
        self.colors = np.array(new_colors)
        self.reinit_temp_colors()

    def init_stochiometry_comparison_list(self):
        """
        Initialize list used to compare stochiometry (in terms of either nodes or, if color_defining_neighborhood_radius !=0, in terms of neighborhoods). Created to avoid calculating canonical permutations of the entire molecular graph when possible.
        """
        if self.stochiometry_comparison_list is not None:
            return
        self.stochiometry_comparison_list = []
        self.init_all_ha_comparison_lists()
        comp_list_nums = {}
        for comp_list in self.ha_comparison_lists:
            comp_tuple = tuple(comp_list)
            if comp_tuple not in comp_list_nums:
                comp_list_nums[comp_tuple] = 0
            comp_list_nums[comp_tuple] += 1
        ordered_comp_list_nums = sorted(list(comp_list_nums.items()))
        for comp_tuple, num in ordered_comp_list_nums:
            self.stochiometry_comparison_list.append(num)
            self.stochiometry_comparison_list.append(len(comp_tuple))
            self.stochiometry_comparison_list += list(comp_tuple)

    def get_stochiometry_comparison_list(self):
        self.init_stochiometry_comparison_list()
        return self.stochiometry_comparison_list

    def polyvalent_atoms_present(self):
        for ha in self.hatoms:
            if ha.is_polyvalent():
                return True
        return False

    def coordination_number(self, hatom_id):
        return len(self.neighbors(hatom_id)) + self.hatoms[hatom_id].nhydrogens

    # Everything related to equivalences between nodes or bonds.
    def num_equiv_classes_from_arr(self, equiv_class_arr):
        max_class_id = np.amax(equiv_class_arr)
        if max_class_id == unassigned_equivalence_class_id:
            return 0
        return max_class_id + 1

    def num_equiv_classes(self, atom_set_length):
        if atom_set_length == 1:
            return self.num_equiv_classes_from_arr(self.equivalence_vector)
        else:
            return self.num_equiv_classes_from_arr(self.pair_equivalence_matrix)

    def init_equivalence_vector(self):
        if self.equivalence_vector is None:
            self.equivalence_vector = np.repeat(unassigned_equivalence_class_id, self.nhatoms())

    def init_pair_equivalence_matrix(self):
        if self.pair_equivalence_matrix is None:
            self.pair_equivalence_matrix = np.repeat(
                unassigned_equivalence_class_id, self.nhatoms() ** 2
            ).reshape((self.nhatoms(), self.nhatoms()))

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

    def check_all_atom_equivalence_classes(self):
        for i in range(self.nhatoms()):
            self.check_equivalence_class((i,))

    def checked_equivalence_vector(self):
        """
        Check that all equivalence classes have been initialized and return the resulting equivalence vector.
        """
        self.check_all_atom_equivalence_classes()
        return self.equivalence_vector

    def sorted_colors(self, atom_set):
        self.init_colors()
        return sorted([self.colors[atom_id] for atom_id in atom_set])

    def bond_orders_equivalent(self, *args):
        # only makes sense for the inheritor ChemGraph class.
        return True

    def atom_sets_equivalence_reasonable(self, atom_id_set1, atom_id_set2):
        if len(atom_id_set1) == 1:
            self.init_colors()
            return self.colors[atom_id_set1[0]] == self.colors[atom_id_set2[0]]
        else:
            if self.sorted_colors(atom_id_set1) != self.sorted_colors(atom_id_set2):
                return False
            if not self.bond_orders_equivalent(atom_id_set1, atom_id_set2):
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

    def atom_sets_iterator(self, atom_set_length):
        return itertools.combinations(range(self.nhatoms()), atom_set_length)

    # TODO Do we still need equivalence classes of bonds?
    def equiv_class_examples(self, atom_set_length, as_tuples=True):
        num_classes = self.num_equiv_classes(atom_set_length)
        if num_classes == 0:
            return []
        if as_tuples:
            output = np.empty((num_classes, atom_set_length), dtype=int)
        else:
            output = np.empty((num_classes,), dtype=int)
        class_represented = np.zeros((num_classes,), dtype=bool)
        equiv_class_arr = self.equiv_arr(atom_set_length)

        cur_example_id = 0

        for atom_set in self.atom_sets_iterator(atom_set_length):
            cur_equiv_class = equiv_class_arr[atom_set]
            if cur_equiv_class == unassigned_equivalence_class_id:
                continue
            if class_represented[cur_equiv_class]:
                continue
            class_represented[cur_equiv_class] = True
            if as_tuples:
                output[cur_example_id, :] = atom_set[:]
            else:
                output[cur_example_id] = atom_set[0]
            cur_example_id += 1
            if cur_example_id == num_classes:
                return output

        # All examples should've been found at this point.
        raise Exception

    def equiv_class_members(self, equiv_class_id, atom_set_length):
        return list(zip(*np.where(self.equiv_arr(atom_set_length) == equiv_class_id)))

    def min_id_equivalent_atom_unchecked(self, atom_id):
        if self.equivalence_vector is None:
            return atom_id
        equiv_class = self.unchecked_equivalence_class(atom_id)
        if equiv_class == unassigned_equivalence_class_id:
            return atom_id
        else:
            return np.where(self.equivalence_vector == equiv_class)[0][0]

    def check_equivalence_class(self, atom_id_set):
        atom_set_length = len(atom_id_set)
        self.init_equivalence_array(atom_set_length)
        if self.unchecked_equivalence_class(atom_id_set) == unassigned_equivalence_class_id:
            self.init_colors()
            for example_tuple in self.equiv_class_examples(atom_set_length):
                if not self.atom_sets_equivalence_reasonable(atom_id_set, example_tuple):
                    continue
                if self.uninit_atom_sets_equivalent(atom_id_set, example_tuple):
                    equiv_class_id = self.unchecked_equivalence_class(example_tuple)
                    self.assign_equivalence_class(atom_id_set, equiv_class_id)
                    return
            self.assign_equivalence_class(atom_id_set, self.num_equiv_classes(atom_set_length))

    def automorphism_check(self, **kwargs):
        # TODO : post 3.10 match would be better here.
        if isomorphism_algorithm == bliss:
            return self.graph.isomorphic_bliss(**kwargs)
        if isomorphism_algorithm == vf2:
            return self.graph.isomorphic_vf2(**kwargs)
        raise Exception

    def uninit_atom_sets_equivalent(self, atom_set1, atom_set2):
        self.init_colors()

        dummy_color_addition = max(self.colors) + 1
        for atom_id1, atom_id2 in zip(atom_set1, atom_set2):
            self.temp_colors1[atom_id1] += dummy_color_addition
            self.temp_colors2[atom_id2] += dummy_color_addition
        are_equivalent = self.automorphism_check(
            color1=self.temp_colors1, color2=self.temp_colors2
        )
        for atom_id1, atom_id2 in zip(atom_set1, atom_set2):
            self.temp_colors1[atom_id1] = self.colors[atom_id1]
            self.temp_colors2[atom_id2] = self.colors[atom_id2]
        return are_equivalent

    def sorted_atom_set_color_list(self, atom_set):
        self.init_colors()
        return sorted(self.colors[atom_id] for atom_id in atom_set)

    def uninit_atom_sets_equivalent_wcolor_check(self, atom_set1, atom_set2):
        if len(atom_set1) != len(atom_set2):
            return False
        self.init_colors()
        if self.sorted_atom_set_color_list(atom_set1) != self.sorted_atom_set_color_list(
            atom_set2
        ):
            return False
        return self.uninit_atom_sets_equivalent(atom_set1, atom_set2)

    def atom_pair_equivalent(self, atom_id1, atom_id2):
        return self.atom_sets_equivalent([atom_id1], [atom_id2])

    def unchecked_equivalence_class(self, atom_set):
        if isinstance(atom_set, int) or (len(atom_set) == 1):
            return self.equivalence_vector[atom_set]
        else:
            return self.pair_equivalence_matrix[tuple(atom_set)]

    def equivalence_class(self, atom_set):
        self.check_equivalence_class(atom_set)
        return self.unchecked_equivalence_class(atom_set)

    def atom_sets_equivalent(self, atom_set1, atom_set2):
        if len(atom_set1) == 2:
            return self.uninit_atom_sets_equivalent(atom_set1, atom_set2)
        return self.equivalence_class(atom_set1) == self.equivalence_class(atom_set2)

    def init_log_permutation_factor(self):
        self.init_colors()
        self.log_permutation_factor = np.log(self.graph.count_isomorphisms_vf2(color1=self.colors))

    def get_log_permutation_factor(self):
        """
        Logarithm of the number of distinct ways the molecule can be permuted.
        """
        if self.log_permutation_factor is None:
            self.init_log_permutation_factor()
        return self.log_permutation_factor

    # How many times atom_id is repeated inside a molecule.
    def atom_multiplicity(self, atom_id):
        return sum(
            self.atom_pair_equivalent(atom_id, other_atom_id)
            for other_atom_id in range(self.nhatoms())
        )

    def unrepeated_atom_list(self):
        self.check_all_atom_equivalence_classes()
        return self.equiv_class_examples(1, as_tuples=False)

    def effective_coordination_number(self, hatom_id):
        """
        Coordination number including unconnected electronic pairs.

        TODO 1. make sure it does not count pairs that contribute to an aromatic system 2. was never properly tested with charges
        """
        pairs = 0
        hatom = self.hatoms[hatom_id]
        ncharge = hatom.ncharge
        if ncharge in unshared_pairs:
            cur_dict = unshared_pairs[ncharge]
            valence = hatom.valence
            if valence in cur_dict:
                pairs = cur_dict[valence]
        return self.coordination_number(hatom_id) + pairs

    def hybridization(self, hatom_id):
        """
        Hybridization of heavy atom hatom_id.

        TODO untested for aromatics and charged species
        """
        return coord_num_hybrid[self.effective_coordination_number(hatom_id)]

    def is_connected(self) -> bool:
        """
        Whether instance contains just one molecule.
        """
        return self.graph.is_connected()

    def num_connected(self) -> int:
        """
        Number of connected molecules inside instance.
        """
        return len(self.graph.components())

    def hatom_default_valence(self, atom_id):
        return default_valence(self.hatoms[atom_id].ncharge)

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

    def tot_nelectrons(self):
        """
        Total number of electrons in the molecule.
        """
        return self.tot_nhydrogens() + sum(ha.ncharge for ha in self.hatoms) - self.charge

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

    def are_neighbors(self, hatom_id1, hatom_id2):
        return self.graph.are_connected(hatom_id1, hatom_id2)

    # Basic commands for managing the graph.
    def set_edge_order(self, atom1, atom2, new_edge_order):
        """
        Change molecular graph's edge order.
        """
        true_bond_tuple = tuple(sorted_tuple(atom1, atom2))
        if new_edge_order < 0:
            raise InvalidChange
        if new_edge_order == 0:
            if self.graph.are_connected(*true_bond_tuple):
                self.graph.delete_edges([true_bond_tuple])
            del self.bond_orders[true_bond_tuple]
        else:
            if not self.graph.are_connected(*true_bond_tuple):
                self.graph.add_edge(*true_bond_tuple)
            self.bond_orders[true_bond_tuple] = new_edge_order

    def assign_extra_edge_orders(self, changed_hatom_ids, extra_order_dict):
        """
        (Only used in ChemGraph for resonance structure assignment.)
        """
        for hatom_considered_num, hatom_id2 in enumerate(changed_hatom_ids):
            hatom2_neighbors = self.neighbors(hatom_id2)
            for hatom_id1 in changed_hatom_ids[:hatom_considered_num]:
                if hatom_id1 in hatom2_neighbors:
                    stuple = sorted_tuple(hatom_id1, hatom_id2)
                    if stuple in extra_order_dict:
                        self.set_edge_order(
                            hatom_id1,
                            hatom_id2,
                            1 + extra_order_dict[stuple],
                        )
                    else:
                        self.set_edge_order(hatom_id1, hatom_id2, 1)

    def change_edge_order(self, atom1, atom2, change=0):
        """
        Change molecular graph's edge order.
        """
        if change == 0:
            return
        if atom1 == atom2:
            raise InvalidChange
        true_bond_tuple = tuple(sorted_tuple(atom1, atom2))
        try:
            cur_edge_order = self.bond_orders[true_bond_tuple]
        except KeyError:
            cur_edge_order = 0
        new_edge_order = cur_edge_order + change
        if new_edge_order < 0:
            raise InvalidChange
        self.set_edge_order(atom1, atom2, new_edge_order)

    def change_hydrogen_number(self, atom_id, hydrogen_number_change):
        self.hatoms[atom_id].nhydrogens += hydrogen_number_change
        if self.hatoms[atom_id].nhydrogens < 0:
            raise InvalidChange

    def append_heavy_atom(self, atom_ncharge, nhydrogens=None):
        """
        Simply append a new heavy atom with nuclear charge of atom_ncharge and nhydrogens attached hydrogens without checking correctness of the resulting valences or creating any new bonds.
        """
        if nhydrogens is None:
            nhydrogens = default_valence(atom_ncharge)
        self.graph.add_vertex()
        self.hatoms.append(HeavyAtom(atom_ncharge, nhydrogens=nhydrogens))

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

    def init_canonical_permutation(self):
        if self.canonical_permutation is not None:
            return
        self.init_colors()
        (
            self.canonical_permutation,
            self.inv_canonical_permutation,
        ) = canonical_permutation_with_inverse(self.graph, self.colors)

    def get_canonical_permutation(self):
        self.init_canonical_permutation()
        return self.canonical_permutation

    def get_inv_canonical_permutation(self):
        self.init_canonical_permutation()
        return self.inv_canonical_permutation

    def get_comparison_list(self):
        if self.comparison_list is None:
            self.init_canonical_permutation()
            self.comparison_list = ha_graph_comparison_list(
                self.graph,
                self.ha_trivial_comparison_lists,
                self.canonical_permutation,
                self.inv_canonical_permutation,
                charge=self.charge,
            )
        return self.comparison_list

    def copy_extra_data_to(self, other_cg, linear_storage=False):
        self.copy_equivalence_info_to(other_cg, linear_storage=linear_storage)

    def match_atom_id_set(self, atom_id_set, other_cg):
        return tuple(
            other_cg.inv_canonical_permutation[self.canonical_permutation[atom_id]]
            for atom_id in atom_id_set
        )

    def match_atom_id_sets(self, atom_id_sets, other_cg):
        return [self.match_atom_id_set(atom_id_set, other_cg) for atom_id_set in atom_id_sets]

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
                    if equiv_class_id != unassigned_equivalence_class_id:
                        copied_equivalence_classes[equiv_class_id] = other_equiv_class_id
                        break

            for equiv_class_id in range(num_equiv_classes):
                equiv_class_members = self.equiv_class_members(equiv_class_id, atom_set_length)
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

    def canonical_atom_set_iterator(self, atom_set_length):
        return itertools.product(*[self.inv_canonical_permutation for _ in range(atom_set_length)])

    def __hash__(self):
        # TODO replace comparison_list with comparison_tuple?
        return hash(tuple(self.get_comparison_list()))

    # for ordering graphs
    def brute_comparison(self, cg2, comp_operator) -> bool:
        return comp_operator(self.get_comparison_list(), cg2.get_comparison_list())

    def comparison(self, cg2, comp_operator) -> bool:
        """
        Perform comparison operation on two chemical graphs by first comparing stochiometries, then comparison lists.
        """
        if self.charge != cg2.charge:
            return comp_operator(self.charge, cg2.charge)
        if not using_two_level_comparison:
            return self.brute_comparison(cg2, comp_operator)
        cg1_stoch = self.get_stochiometry_comparison_list()
        cg2_stoch = cg2.get_stochiometry_comparison_list()
        if cg1_stoch == cg2_stoch:
            return self.brute_comparison(cg2, comp_operator)
        else:
            return comp_operator(cg1_stoch, cg2_stoch)

    def __lt__(self, cg2):
        return self.comparison(cg2, operator.lt)

    def __gt__(self, cg2):
        return self.comparison(cg2, operator.gt)

    def __eq__(self, cg2):
        return self.comparison(cg2, operator.eq)

    def __str__(self):
        """
        The representation consists of substrings corresponding to each heavy atom in canonical ordering and connected by ":".
        Each of those substrings starts with the nuclear charge of the atom, continues after "#" with the number of hydrogens connected to
        the heavy atom (if not 0), and finishes with list of other atom indices preceded by "@" to which the current atom is connected and
        whose indices exceed the current atom's index. If the molecule's charge is not 0 it is added in the end after "_" sign.

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
            hatom_string = str(self.hatoms[hatom_id])
            if len(canonical_neighbor_list) != 0:
                canonical_neighbor_list.sort()
                hatom_string += "@" + "@".join(str(cn_id) for cn_id in canonical_neighbor_list)
            hatom_strings.append(hatom_string)
        rep_str = ":".join(hatom_strings)
        if self.charge != 0:
            rep_str += "_" + str(self.charge)
        return rep_str

    def __repr__(self):
        return str(self)
