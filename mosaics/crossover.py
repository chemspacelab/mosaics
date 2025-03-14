# Everything related to crossover moves.
# TODO: Konstantin: I know the number of objects created here can be cut down significantly, but
# at this point more polishing could be excessive.

import bisect
import itertools
import random
from copy import deepcopy

import numpy as np
from igraph.operators import disjoint_union
from sortedcontainers import SortedList

from .ext_graph_compound import atom_multiplicity_in_list, connection_forbidden
from .misc_procedures import VERBOSITY, VERBOSITY_MUTED, intlog
from .valence_treatment import ChemGraph, InvalidChange, max_bo, sorted_by_membership, sorted_tuple


class Frag2FragMapping:
    def __init__(self, new_membership_vector, old_membership_vector, frag_id=0):
        """
        Auxiliary class for mapping HeavyAtom objects from the same fragment used for two FragmentPair objects.
        """
        self.old_frag_ids = np.where(old_membership_vector == frag_id)[0]
        self.new_frag_ids = np.where(new_membership_vector == frag_id)[0]

    def __call__(self, old_id):
        """
        The new id of HeavyAtom which went by old_id in the old fragment pair.
        """
        inside_id = bisect.bisect_left(self.old_frag_ids, old_id)
        return self.new_frag_ids[inside_id]


class Frag2FragBondMapping:
    def __init__(self, core_init_frag, other_init_frag, new_core_membership_vector):
        """
        Auxiliary class for showing which pairs of bond tuples were created after exchanging atoms between FragmentPair objects.
        core_init_frag : FragmentPair object whose core was used in the new ChemGraph.
        other_init_frag : FragmentPair object with which the exchange was made.
        new_core_membership_vector : membership vector of the new ChemGraph created with the core of core_init_frag.
        """
        self.core_to_new = Frag2FragMapping(
            new_core_membership_vector,
            core_init_frag.membership_vector,
            frag_id=core_init_frag.core_membership_vector_value,
        )
        self.other_to_new = Frag2FragMapping(
            new_core_membership_vector,
            other_init_frag.membership_vector,
            frag_id=other_init_frag.remainder_membership_vector_value,
        )

    def new_bond_tuple(self, old_tuple_core, old_tuple_other):
        internal_id1 = self.core_to_new(old_tuple_core[0])
        internal_id2 = self.other_to_new(old_tuple_other[1])
        return (internal_id1, internal_id2)

    def new_bond_tuples(self, old_tuple_list1, old_tuple_list2):
        output = []
        for old_tuple1, old_tuple2 in zip(old_tuple_list1, old_tuple_list2):
            new_bond_tuple = self.new_bond_tuple(old_tuple1, old_tuple2)
            # This is done to prevent non-invertible formation of a bond of non-unity order.
            if new_bond_tuple in output:
                return None
            output.append(new_bond_tuple)
        return output


class FragmentPairAffectedBondStatus:
    def __init__(self, bond_tuple_dict, valences=None):
        """
        Auxiliary class storing information about bond orders and valences affected by ChemGraph object being ``cut'' into a FragmentPair.
        """
        self.bond_tuple_dict = bond_tuple_dict
        if valences is None:
            valences = {}
        self.valences = valences

    def __eq__(self, other_fpabs):
        return (self.bond_tuple_dict == other_fpabs.bond_tuple_dict) and (
            self.valences == other_fpabs.valences
        )

    def __str__(self):
        return (
            "<FragmentPairAffectedBondStatus:"
            + str(self.bond_tuple_dict)
            + ":"
            + str(self.valences)
            + ">"
        )

    def __repr__(self):
        return str(self)


def bond_list_to_atom_tuple(bond_list):
    output = SortedList()
    for bt in bond_list:
        for atom_id in bt:
            if atom_id not in output:
                output.add(atom_id)
    return tuple(output)


class FragmentPair:
    def __init__(
        self,
        cg: ChemGraph,
        origin_point: int,
        neighborhood_size: int = 0,
    ):
        """
        Pair of fragments formed from one molecule split around the membership vector.
        cg : ChemGraph - the split molecule
        origin_point : int - origin point of the "core" fragment
        neighborhood_size : int - distance between origin point and nodes included into the "core" fragment
        """
        self.chemgraph = cg
        # Check that resonance structures are initialized.
        self.chemgraph.init_resonance_structures()

        self.origin_point = origin_point
        self.neighborhood_size = neighborhood_size

        # constants for how different parts of FragmentPair are marked in self.membership_vector
        self.core_membership_vector_value = 0
        self.remainder_membership_vector_value = 1
        self.border_membership_vector_value = -1

        self.membership_vector = np.repeat(self.remainder_membership_vector_value, cg.nhatoms())
        self.membership_vector[origin_point] = self.core_membership_vector_value

        self.affected_bonds = [
            (self.origin_point, neigh) for neigh in self.chemgraph.neighbors(origin_point)
        ]

        self.bond_list_equivalence_class_examples = []
        self.bond_list_equivalence_class_dict = {}

        while self.core_size() < neighborhood_size:
            self.expand_core(update_affected_status=False)
            if self.no_fragment_neighbors():
                # Can happen before self.core_size() == neighborhood_size if self.cg is disjoint.
                break

        # Two list of vertices corresponding to the two fragments.
        self.sorted_vertices = None
        self.init_affected_status_info()

    def no_fragment_neighbors(self):
        return len(self.affected_bonds) == 0

    def expand_core(self, update_affected_status=True):
        for old_affected_bond in self.affected_bonds:
            new_id = old_affected_bond[1]
            if self.is_remainder(new_id):
                self.membership_vector[new_id] = self.border_membership_vector_value
        self.affected_bonds = []
        for border_id in np.where(self.membership_vector == self.border_membership_vector_value)[
            0
        ]:
            self.membership_vector[border_id] = self.core_membership_vector_value
            for neigh in self.chemgraph.neighbors(border_id):
                if self.is_remainder(neigh):
                    self.affected_bonds.append((border_id, neigh))
        if update_affected_status:
            self.init_affected_status_info()

    def is_remainder(self, i):
        return self.membership_vector[i] == self.remainder_membership_vector_value

    def core_size(self):
        return np.sum(self.membership_vector == self.core_membership_vector_value)

    def init_affected_status_info(self):
        """
        Initialize information about bonds and valences that are parts of resonance structures affected by fragmentation.
        """
        self.affected_resonance_structures = []
        resonance_structure_orders_iterators = []

        resonance_structure_affected_bonds = (
            {}
        )  # lists of bonds affected by different resonance structure regions

        saved_all_bond_orders = {}

        default_bond_order_dict = {}  # orders of bonds not affected by resonance structures

        cg = self.chemgraph

        for bond_tuple in self.affected_bonds:
            bond_stuple = sorted_tuple(*bond_tuple)
            if bond_stuple in cg.resonance_structure_map:
                rsr_id = cg.resonance_structure_map[bond_stuple]
                if rsr_id not in saved_all_bond_orders:
                    saved_all_bond_orders[rsr_id] = {}
                    self.affected_resonance_structures.append(rsr_id)
                    resonance_structure_orders_iterators.append(
                        range(len(cg.resonance_structure_orders[rsr_id]))
                    )
                if rsr_id not in resonance_structure_affected_bonds:
                    resonance_structure_affected_bonds[rsr_id] = []
                resonance_structure_affected_bonds[rsr_id].append(bond_tuple)
                saved_all_bond_orders[rsr_id][bond_tuple] = cg.aa_all_bond_orders(
                    *bond_stuple, unsorted=True
                )

            else:
                cur_bo = cg.bond_orders[bond_stuple]
                if cur_bo in default_bond_order_dict:
                    default_bond_order_dict[cur_bo].append(bond_tuple)
                else:
                    default_bond_order_dict[cur_bo] = [bond_tuple]

        if len(resonance_structure_orders_iterators) == 0:
            self.affected_status = [FragmentPairAffectedBondStatus(default_bond_order_dict)]
            return

        self.affected_status = []
        for resonance_structure_orders_ids in itertools.product(
            *resonance_structure_orders_iterators
        ):
            new_status = FragmentPairAffectedBondStatus(deepcopy(default_bond_order_dict))
            for res_reg_id, rso_id in zip(
                self.affected_resonance_structures, resonance_structure_orders_ids
            ):
                ha_ids = cg.resonance_structure_inverse_map[res_reg_id]
                val_pos = cg.resonance_structure_valence_vals[res_reg_id][rso_id]
                for ha_id in ha_ids:
                    poss_valences = cg.hatoms[ha_id].possible_valences
                    if poss_valences is not None:
                        new_status.valences[ha_id] = poss_valences[val_pos]
                for btuple, bos in saved_all_bond_orders[res_reg_id].items():
                    cur_bo = bos[rso_id]
                    if cur_bo in new_status.bond_tuple_dict:
                        new_status.bond_tuple_dict[cur_bo].append(btuple)
                    else:
                        new_status.bond_tuple_dict[cur_bo] = [btuple]

            if new_status not in self.affected_status:
                self.affected_status.append(new_status)

    def assign_equivalence_class_to_atom_tuple(self, atom_tuple):
        found_equivalence_class = None
        for equiv_cl_id, other_at in enumerate(self.bond_list_equivalence_class_examples):
            if self.chemgraph.uninit_atom_sets_equivalent_wcolor_check(other_at, atom_tuple):
                found_equivalence_class = equiv_cl_id
                break
        if found_equivalence_class is None:
            found_equivalence_class = len(self.bond_list_equivalence_class_examples)
            self.bond_list_equivalence_class_examples.append(atom_tuple)
        self.bond_list_equivalence_class_dict[atom_tuple] = found_equivalence_class

    def get_bond_list_equivalence_class(self, bond_list):
        atom_list = bond_list_to_atom_tuple(bond_list)
        if atom_list not in self.bond_list_equivalence_class_dict:
            self.assign_equivalence_class_to_atom_tuple(atom_list)
        return self.bond_list_equivalence_class_dict[atom_list]

    def equivalence_status_representation(self, status: FragmentPairAffectedBondStatus):
        equiv_rep = {}
        for bo, bt_list in status.bond_tuple_dict.items():
            equiv_rep[bo] = self.get_bond_list_equivalence_class(bt_list)
        return equiv_rep

    # TODO For now it is only used in linear-scaling implementation of crossover moves, but perhaps should be used in the original version too.
    def get_equiv_checked_affected_statuses(self):
        """
        Get members of self.affected_status that are not symmetrically equivalent.
        """
        checked_status_list = []
        checked_status_equivalence_rep_list = []
        for status in self.affected_status:
            equiv_rep = self.equivalence_status_representation(status)
            if equiv_rep in checked_status_equivalence_rep_list:
                continue
            checked_status_list.append(status)
            checked_status_equivalence_rep_list.append(equiv_rep)
        return checked_status_list

    def get_sorted_vertices(self, frag_id):
        if self.sorted_vertices is None:
            self.sorted_vertices = sorted_by_membership(self.membership_vector)
        return self.sorted_vertices[frag_id]

    def adjusted_ha_valences(self, status_id, membership_id):
        vals = self.affected_status[status_id].valences
        output = []
        for i in self.get_sorted_vertices(membership_id):
            if i in vals:
                output.append(vals[i])
            else:
                output.append(self.chemgraph.hatoms[i].valence)
        return output

    def get_frag_size(self, frag_id):
        return len(self.get_sorted_vertices(frag_id))

    def get_hatoms_sublist(self, frag_id):
        return [
            self.chemgraph.hatoms[ha_id].mincopy() for ha_id in self.get_sorted_vertices(frag_id)
        ]

    def get_frag_subgraph(self, frag_id):
        return self.chemgraph.graph.subgraph(self.get_sorted_vertices(frag_id))

    def crossover(
        self,
        other_fp,
        switched_bond_tuples_self: int,
        switched_bond_tuples_other: int,
        affected_status_id_self: int,
        affected_status_id_other: int,
        **other_kwargs,
    ):
        """
        Couple to another fragment.
        switched_bond_tuples_self, switched_bond_tuples_other - bonds corresponding to which heavy atom index tuples are switched between different fragments.
        affected_bonds_id_self, affected_bonds_id_other - which sets of affected bond orders the fragments are initialized with.
        """

        frag_id_self = self.core_membership_vector_value
        frag_id_other = other_fp.remainder_membership_vector_value

        nhatoms_self = self.get_frag_size(frag_id_self)
        nhatoms_other = other_fp.get_frag_size(frag_id_other)

        new_membership_vector = np.append(
            np.repeat(frag_id_self, nhatoms_self),
            np.repeat(frag_id_other, nhatoms_other),
        )

        # Check which bonds need to be created.
        bond_mapping = Frag2FragBondMapping(self, other_fp, new_membership_vector)

        created_bonds = bond_mapping.new_bond_tuples(
            switched_bond_tuples_self, switched_bond_tuples_other
        )
        if created_bonds is None:
            return None, None

        # "Sew" two graphs together with the created bonds.
        new_graph = disjoint_union(
            [
                self.get_frag_subgraph(frag_id_self),
                other_fp.get_frag_subgraph(frag_id_other),
            ]
        )

        for new_bond_tuple in created_bonds:
            new_graph.add_edge(*new_bond_tuple)

        new_hatoms = self.get_hatoms_sublist(frag_id_self) + other_fp.get_hatoms_sublist(
            frag_id_other
        )

        new_ChemGraph = ChemGraph(hatoms=new_hatoms, graph=new_graph)

        new_hatoms_old_valences = self.adjusted_ha_valences(
            affected_status_id_self, frag_id_self
        ) + other_fp.adjusted_ha_valences(affected_status_id_other, frag_id_other)

        # Lastly, check that re-initialization does not decrease the bond order.
        if not new_ChemGraph.valence_config_valid(new_hatoms_old_valences):
            return None, None

        return new_ChemGraph, new_membership_vector

    def ncharge(self, hatom_id):
        return self.chemgraph.hatoms[hatom_id].ncharge


def possible_fragment_size_bounds(cg):
    """
    Possible fragment sizes for a ChemGraph objects satisfying enforced constraints. Cut compared to the previous versions for the sake of detailed balance simplicity.
    """
    return [1, cg.nhatoms() - 1]


def valid_cross_connection(cg1, cg2, tlist1, tlist2, bo, forbidden_bonds=None):
    for t1, t2 in zip(tlist1, tlist2):
        for new_bond in [(t1[0], t2[1]), (t1[1], t2[0])]:
            ha1 = cg1.hatoms[new_bond[0]]
            ha2 = cg2.hatoms[new_bond[1]]
            if bo > max_bo(ha1, ha2):
                return False
            if connection_forbidden(ha1.ncharge, ha2.ncharge, forbidden_bonds):
                return False
    return True


def bo_tuple_dicts_shapes_match(bo_tuple_dict1, bo_tuple_dict2):
    """
    Check that the two dictionnaries are of similar dimensionality.
    """
    if len(bo_tuple_dict1) != len(bo_tuple_dict2):
        return False
    for bo1, bo_tuples1 in bo_tuple_dict1.items():
        if bo1 not in bo_tuple_dict2:
            return False
        if len(bo_tuple_dict2[bo1]) != len(bo_tuples1):
            return False
    return True


class BondOrderSortedTuplePermutations:
    def __init__(
        self,
        status1: FragmentPairAffectedBondStatus,
        status2: FragmentPairAffectedBondStatus,
        chemgraph1: ChemGraph,
        chemgraph2: ChemGraph,
        forbidden_bonds=None,
    ):
        """
        Generates permutations of tuples grouped by bond orders making sure no forbidden bonds are created.
        """
        self.bo_tuple_dict1 = status1.bond_tuple_dict
        self.bo_tuple_dict2 = status2.bond_tuple_dict
        # Check that the two dictionnaries are of similar dimensionality.
        self.non_empty = bo_tuple_dicts_shapes_match(self.bo_tuple_dict1, self.bo_tuple_dict2)

        if not self.non_empty:
            return

        # Initialize necessary quantities.
        self.chemgraph1 = chemgraph1
        self.chemgraph2 = chemgraph2
        self.sorted_bos = sorted(self.bo_tuple_dict1.keys())

        self.forbidden_bonds = forbidden_bonds
        iterators = []
        for bo in self.sorted_bos:
            bo_tuples1 = self.bo_tuple_dict1[bo]
            iterators.append(itertools.permutations(bo_tuples1))
        self.iterator_product = itertools.product(*iterators)

    def check_non_empty(self):
        if not self.non_empty:
            return False
        try:
            _ = self.__next__()
            return True
        except StopIteration:
            return False

    def __iter__(self):
        if self.non_empty:
            return self
        else:
            return iter(())

    def __next__(self):
        while True:
            finished = True
            tuples1 = []
            tuples2 = []
            for t1, bo in zip(self.iterator_product.__next__(), self.sorted_bos):
                t2 = self.bo_tuple_dict2[bo]
                if not valid_cross_connection(
                    self.chemgraph1,
                    self.chemgraph2,
                    t1,
                    t2,
                    bo,
                    forbidden_bonds=self.forbidden_bonds,
                ):
                    finished = False
                    break
                tuples1 += list(t1)
                tuples2 += list(t2)
            if finished:
                return tuples1, tuples2


def crossover_outcomes(cg_pair, chosen_sizes, origin_points, forbidden_bonds=None):
    frag1 = FragmentPair(cg_pair[0], origin_points[0], neighborhood_size=chosen_sizes[0])
    frag2 = FragmentPair(cg_pair[1], origin_points[1], neighborhood_size=chosen_sizes[1])

    new_chemgraph_pairs = []
    new_origin_points = None
    for status_id1, status1 in enumerate(frag1.affected_status):
        for status_id2, status2 in enumerate(frag2.affected_status):
            for tuples1, tuples2 in BondOrderSortedTuplePermutations(
                status1, status2, *cg_pair, forbidden_bonds=forbidden_bonds
            ):
                new_chemgraph_1, new_membership_vector_1 = frag1.crossover(
                    frag2, tuples1, tuples2, status_id1, status_id2
                )
                new_chemgraph_2, new_membership_vector_2 = frag2.crossover(
                    frag1, tuples2, tuples1, status_id2, status_id1
                )
                if (new_membership_vector_1 is None) or (new_membership_vector_2 is None):
                    continue
                if new_origin_points is None:
                    map1 = Frag2FragMapping(
                        new_membership_vector_1,
                        frag1.membership_vector,
                        frag_id=frag1.core_membership_vector_value,
                    )
                    map2 = Frag2FragMapping(
                        new_membership_vector_2,
                        frag2.membership_vector,
                        frag_id=frag2.core_membership_vector_value,
                    )
                    new_origin_points = (
                        map1(frag1.origin_point),
                        map2(frag2.origin_point),
                    )
                new_pair = (new_chemgraph_1, new_chemgraph_2)
                if new_pair not in new_chemgraph_pairs:
                    new_chemgraph_pairs.append(new_pair)
    if new_origin_points is None:
        return None, None
    else:
        return new_chemgraph_pairs, new_origin_points


class RandomTupleBondReconnector:
    def __init__(self, status1, status2):
        bo_tuple_dict1 = status1.bond_tuple_dict
        bo_tuple_dict2 = status2.bond_tuple_dict

        self.tuples_correspondence = {}
        for bo1, tuples1 in bo_tuple_dict1.items():
            self.tuples_correspondence[sorted_tuple(*tuples1)] = bo_tuple_dict2[bo1]

        self.reconnecting_list = None

    def __eq__(self, other_rtbr):
        for self_tuples1, self_tuples2 in self.tuples_correspondence.items():
            if self_tuples1 not in other_rtbr.tuples_correspondence:
                return False
            if self_tuples2 != other_rtbr.tuples_correspondence[self_tuples1]:
                return False
        return True

    def init_shuffled_reconnecting_list(self):
        self.reconnecting_list = []
        for tuples1, tuples2 in self.tuples_correspondence.items():
            random.shuffle(tuples2)
            new_tuples1 = list(tuples1)
            random.shuffle(new_tuples1)
            self.reconnecting_list.append([new_tuples1, tuples2])
        random.shuffle(self.reconnecting_list)

    def check_bond_satisfaction(self, frag1, frag2, forbidden_bonds=None):
        created_bonds1 = SortedList()
        created_bonds2 = SortedList()
        for reconnecting_tuples_list_pair in self.reconnecting_list:
            for tuples1, tuples2 in zip(*reconnecting_tuples_list_pair):
                created_bond1 = (tuples1[0], tuples2[1])
                created_bond2 = (tuples2[0], tuples1[1])
                if created_bond1 in created_bonds1:
                    return False
                else:
                    created_bonds1.add(created_bond1)
                if created_bond2 in created_bonds2:
                    return False
                else:
                    created_bonds2.add(created_bond2)
                if forbidden_bonds is not None:
                    nc11 = frag1.ncharge(tuples1[0])
                    nc22 = frag2.ncharge(tuples2[1])
                    if connection_forbidden(nc11, nc22, forbidden_bonds=forbidden_bonds):
                        return False
                    nc21 = frag2.ncharge(tuples2[0])
                    nc12 = frag1.ncharge(tuples1[1])
                    if connection_forbidden(nc21, nc12, forbidden_bonds=forbidden_bonds):
                        return False
        return True

    def inverse_reconnecting_list(
        self, init_fragpair1, init_fragpair2, new_core_membership1, new_core_membership2
    ):
        core1_bond_mapping = Frag2FragBondMapping(
            init_fragpair1, init_fragpair2, new_core_membership1
        )
        core2_bond_mapping = Frag2FragBondMapping(
            init_fragpair2, init_fragpair1, new_core_membership2
        )
        output = []
        for [tuples1, tuples2] in self.reconnecting_list:
            new_tuples1 = core1_bond_mapping.new_bond_tuples(tuples1, tuples2)
            new_tuples2 = core2_bond_mapping.new_bond_tuples(tuples2, tuples1)
            output.append([new_tuples1, new_tuples2])
        return output

    def equivalence_representation(self, frag1: FragmentPair, frag2: FragmentPair):
        output = SortedList()
        for tuple_list1, tuple_list2 in self.tuples_correspondence.items():
            output.add(
                (
                    frag1.get_bond_list_equivalence_class(tuple_list1),
                    frag2.get_bond_list_equivalence_class(tuple_list2),
                )
            )
        return list(output)

    def __str__(self):
        return (
            "<RandomTupleBondReconnector,tuples_correspondence:"
            + str(self.tuples_correspondence)
            + ">"
        )

    def __repr__(self):
        return str(self)


def matching_status_reconnectors(frag1, frag2):
    valid_reconnectors = []
    valid_reconnector_equiv_reps = []
    valid_status_ids = []
    for status_id1, status1 in enumerate(frag1.get_equiv_checked_affected_statuses()):
        for status_id2, status2 in enumerate(frag2.get_equiv_checked_affected_statuses()):
            if not bo_tuple_dicts_shapes_match(status1.bond_tuple_dict, status2.bond_tuple_dict):
                continue
            cur_reconnector = RandomTupleBondReconnector(status1, status2)
            cur_reconnector_equiv_rep = cur_reconnector.equivalence_representation(frag1, frag2)
            if cur_reconnector_equiv_rep not in valid_reconnector_equiv_reps:
                valid_reconnector_equiv_reps.append(cur_reconnector_equiv_rep)
                valid_reconnectors.append(cur_reconnector)
                valid_status_ids.append((status_id1, status_id2))
    return valid_reconnectors, valid_status_ids


def matching_status_reconnectors_wfrags(cg_pair, origin_points, chosen_sizes):
    frag1 = FragmentPair(cg_pair[0], origin_points[0], neighborhood_size=chosen_sizes[0])
    frag2 = FragmentPair(cg_pair[1], origin_points[1], neighborhood_size=chosen_sizes[1])
    valid_reconnectors, valid_status_ids = matching_status_reconnectors(frag1, frag2)
    return valid_reconnectors, valid_status_ids, frag1, frag2


# TODO: Check this function is used everywhere?
def get_exchanged_tuples(t1, t2):
    return (t1[0], t2[1]), (t2[0], t1[1])


# TODO might be used to replace FragmentPair.crossover?
class FragmentPairReconnectingBlob:
    def __init__(
        self,
        frag1: FragmentPair,
        frag2: FragmentPair,
        shuffled_reconnector: RandomTupleBondReconnector,
        affected_status_id1: int,
        affected_status_id2: int,
    ):
        """
        Auxiliary class used to combine two FragmentPair objects into a single "blob" where reconnection of bonds is done.
        After reconnection two new ChemGraph objects are recovered along with the new origin points.
        """
        # new_frag1 and new_frag2 are the new FragPair objects with "cores" borrowed from frag1 and frag2.
        self.new_frag1_membership_value = frag1.core_membership_vector_value
        self.new_frag2_membership_value = frag1.remainder_membership_vector_value
        self.new_frag1_origin_point = frag1.origin_point
        self.new_frag2_origin_point = frag2.origin_point + frag1.chemgraph.nhatoms()
        # Assign memberships
        frag2_membership_vector_addition = np.copy(frag2.membership_vector)
        for hatom_id, membership in enumerate(frag2_membership_vector_addition):
            if membership == frag2.core_membership_vector_value:
                new_membership = self.new_frag2_membership_value
            else:
                new_membership = self.new_frag1_membership_value
            frag2_membership_vector_addition[hatom_id] = new_membership
        self.membership_vector = np.append(
            np.copy(frag1.membership_vector), frag2_membership_vector_addition
        )
        # The "resolved" membership vector version makes "red" and "blue" fragment nodes distinguishable.
        self.resolved_membership_vector = np.copy(self.membership_vector)
        self.resolved_membership_vector[frag1.chemgraph.nhatoms() :] += 2

        # Initialize ChemGraph associated with the blob.
        self.blob_chemgraph = deepcopy(frag1.chemgraph)
        self.blob_chemgraph.hatoms += deepcopy(frag2.chemgraph.hatoms)
        # TODO do we need deepcopy here?
        self.blob_chemgraph.graph = disjoint_union(
            [self.blob_chemgraph.graph, deepcopy(frag2.chemgraph.graph)]
        )
        self.blob_chemgraph.changed()
        # How bonds are reconnected.
        self.forward_reconnection = []
        hatom_id_shift = frag1.chemgraph.nhatoms()
        for [
            exchanged_tuple_list1,
            exchanged_tuple_list2,
        ] in shuffled_reconnector.reconnecting_list:
            exchanged_tuple_true_list2 = [
                (bond_tuple[0] + hatom_id_shift, bond_tuple[1] + hatom_id_shift)
                for bond_tuple in exchanged_tuple_list2
            ]
            self.forward_reconnection.append([exchanged_tuple_list1, exchanged_tuple_true_list2])
        self.backward_reconnection = []

        # Logarithm of probability ratio of the forward and backward reconnection being proposed.
        self.log_prob_balance = 0.0
        # Need to save old valences to check that they are satisfied in the new compounds.
        self.old_valences_lists = []
        self.old_valences_lists.append(
            frag1.adjusted_ha_valences(affected_status_id1, frag1.core_membership_vector_value)
            + frag2.adjusted_ha_valences(
                affected_status_id2, frag2.remainder_membership_vector_value
            )
        )
        self.old_valences_lists.append(
            frag1.adjusted_ha_valences(
                affected_status_id1, frag1.remainder_membership_vector_value
            )
            + frag2.adjusted_ha_valences(affected_status_id2, frag2.core_membership_vector_value)
        )

        self.init_chemgraph_colors()

    def init_chemgraph_colors(self):
        """
        Initialize colors of the underlying ChemGraph as if the "red" and "blue" fragments are properly disconnected and distinguished from each other.
        Only differs from calling self.blob_chemgraph.init_colors if color_defining_neighborhood_radius != 0.
        """
        all_deleted_bonds = []
        for [exchanged_tuple_list1, exchanged_tuple_list2] in self.forward_reconnection:
            all_deleted_bonds += exchanged_tuple_list1
            all_deleted_bonds += exchanged_tuple_list2

        self.blob_chemgraph.graph.delete_edges(all_deleted_bonds)
        self.blob_chemgraph.colors = None
        self.blob_chemgraph.init_colors()
        self.blob_chemgraph.graph.add_edges(all_deleted_bonds)

        # Shift the colors to make fragment nodes distinguishable.
        frag_sorted_colors = sorted_by_membership(
            self.resolved_membership_vector, self.blob_chemgraph.colors
        )
        color_shift = np.zeros((4,), dtype=int)
        for frag_id, frag_colors in enumerate(frag_sorted_colors[:3]):
            max_color = max(frag_colors)
            color_shift[frag_id + 1] = color_shift[frag_id] + max_color + 1
        new_colors = np.copy(self.blob_chemgraph.colors)
        for node_id in range(self.blob_chemgraph.nhatoms()):
            frag_id = self.resolved_membership_vector[node_id]
            new_colors[node_id] += color_shift[frag_id]

        # Updated temp_colors accordingly.
        self.blob_chemgraph.overwrite_colors(new_colors)

    def log_prob_bond_choice(self, chosen_bond, other_bond_choices):
        equivalence_counter = 1
        for other_bond in other_bond_choices:
            if self.blob_chemgraph.uninit_atom_sets_equivalent_wcolor_check(
                chosen_bond, other_bond
            ):
                equivalence_counter += 1
        return intlog(equivalence_counter) - intlog(len(other_bond_choices) + 1)

    def reconnect_tuples(self, exchanged_tuple_list1, exchanged_tuple_list2):
        backward_exchanged_tuples1 = []
        backward_exchanged_tuples2 = []
        for exchange_id, (exchanged_tuple1, exchanged_tuple2) in enumerate(
            zip(exchanged_tuple_list1, exchanged_tuple_list2)
        ):
            # Contribution from forward shuffle choice.
            self.log_prob_balance += self.log_prob_bond_choice(
                exchanged_tuple1, exchanged_tuple_list1[exchange_id + 1 :]
            )
            self.log_prob_balance += self.log_prob_bond_choice(
                exchanged_tuple2, exchanged_tuple_list2[exchange_id + 1 :]
            )
            # Making the necessary bond changes.
            new_bond1, new_bond2 = get_exchanged_tuples(exchanged_tuple1, exchanged_tuple2)
            self.blob_chemgraph.graph.add_edges([new_bond1, new_bond2])
            # Updating the lists of created bonds.
            backward_exchanged_tuples1.append(new_bond1)
            backward_exchanged_tuples2.append(new_bond2)

        self.backward_reconnection.append([backward_exchanged_tuples1, backward_exchanged_tuples2])

    def disconnect_extra_tuples(self, backward_tuple_list1, backward_tuple_list2):
        # Not %100 how useful it is, but making sure forward and backward connections are marked in same order.
        final_exchanged_tuple_list1 = backward_tuple_list1[::-1]
        final_exchanged_tuple_list2 = backward_tuple_list2[::-1]
        for exchange_id, (exchanged_tuple1, exchanged_tuple2) in enumerate(
            zip(final_exchanged_tuple_list1, final_exchanged_tuple_list2)
        ):
            broken_bond1, broken_bond2 = get_exchanged_tuples(exchanged_tuple1, exchanged_tuple2)
            self.blob_chemgraph.graph.delete_edges([broken_bond1, broken_bond2])
            self.log_prob_balance -= self.log_prob_bond_choice(
                exchanged_tuple1, final_exchanged_tuple_list1[:exchange_id]
            )
            self.log_prob_balance -= self.log_prob_bond_choice(
                exchanged_tuple2, final_exchanged_tuple_list2[:exchange_id]
            )

    def reconnect_all_tuples(self):
        self.log_prob_balance = 0.0
        for [exchanged_tuple_list1, exchanged_tuple_list2] in self.forward_reconnection:
            self.reconnect_tuples(exchanged_tuple_list1, exchanged_tuple_list2)
        for [backward_tuple_list1, backward_tuple_list2] in self.backward_reconnection[::-1]:
            self.disconnect_extra_tuples(backward_tuple_list1, backward_tuple_list2)

    def chemgraphs_origin_points(self):
        cg_pair = []
        new_origin_points = []
        for frag_member_value, origin_point, old_valences_list in zip(
            [self.new_frag1_membership_value, self.new_frag2_membership_value],
            [self.new_frag1_origin_point, self.new_frag2_origin_point],
            self.old_valences_lists,
        ):
            frag_members = np.where(self.membership_vector == frag_member_value)[0]
            new_graph = self.blob_chemgraph.graph.subgraph(frag_members)
            new_hatoms = [self.blob_chemgraph.hatoms[frag_member] for frag_member in frag_members]
            new_cg = ChemGraph(hatoms=new_hatoms, graph=new_graph)

            # Check that valences were valid.
            if not new_cg.valence_config_valid(old_valences_list):
                return None, None

            new_origin_point = np.where(frag_members == origin_point)[0][0]
            cg_pair.append(new_cg)
            new_origin_points.append(new_origin_point)
        return cg_pair, new_origin_points


def crossover_sample_random_outcome(cg_pair, chosen_sizes, origin_points, forbidden_bonds=None):
    """
    Make num_pair_generation_attempts attempts to create a cross-coupling from , then randomly choose among the resulting random cross-couplings.
    Also returns: (1) probability factor related to the number of permutations (2) probability factor corresponding to status choice.
    Note that probability factor corresponding to reconnector.shuffle cancels out.
    """
    frag1 = FragmentPair(cg_pair[0], origin_points[0], neighborhood_size=chosen_sizes[0])
    frag2 = FragmentPair(cg_pair[1], origin_points[1], neighborhood_size=chosen_sizes[1])
    (
        valid_reconnectors,
        valid_status_ids,
        frag1,
        frag2,
    ) = matching_status_reconnectors_wfrags(cg_pair, origin_points, chosen_sizes)

    num_reconnectors = len(valid_reconnectors)

    if len(valid_reconnectors) == 0:
        return None, None, None, None

    final_reconnector_id = np.random.randint(len(valid_reconnectors))
    final_reconnector = valid_reconnectors[final_reconnector_id]
    final_status_id1, final_status_id2 = valid_status_ids[final_reconnector_id]

    final_reconnector.init_shuffled_reconnecting_list()

    if not final_reconnector.check_bond_satisfaction(
        frag1, frag2, forbidden_bonds=forbidden_bonds
    ):
        return None, None, None, None

    fragment_blob = FragmentPairReconnectingBlob(
        frag1, frag2, final_reconnector, final_status_id1, final_status_id2
    )

    fragment_blob.reconnect_all_tuples()

    new_chemgraph_pair, new_origin_points = fragment_blob.chemgraphs_origin_points()

    return (
        new_chemgraph_pair,
        new_origin_points,
        num_reconnectors,
        fragment_blob.log_prob_balance,
    )


def frag_swap_size_compliant(cg, frag_size_leaving, frag_size_adding, nhatoms_range=None):
    """
    Check whether swapping core fragments of defined status leaves a GraphCompound satisfying number of nodes constraint.
    """
    if nhatoms_range is not None:
        new_size = cg.nhatoms() - frag_size_leaving + frag_size_adding
        if new_size > nhatoms_range[1]:
            return False
        if new_size < nhatoms_range[0]:
            return False
    return True


def frag_size_status_list(cg, origin_point, max_num_affected_bonds=None):
    """
    Map all possible ways a ChemGraph can be broken into FragmentPair satisfying input constraints.
    """
    frag_size_bounds = possible_fragment_size_bounds(cg)
    output = []
    temp_fp = FragmentPair(cg, origin_point)
    while temp_fp.core_size() <= frag_size_bounds[1]:
        if temp_fp.core_size() >= frag_size_bounds[0]:
            if (max_num_affected_bonds is None) or (
                len(temp_fp.affected_bonds) <= max_num_affected_bonds
            ):
                output.append((temp_fp.core_size(), temp_fp.affected_status))
        temp_fp.expand_core()
        if temp_fp.no_fragment_neighbors():
            break
    return output


def contains_matching_status(cg1, cg2, status_list1, status_list2, forbidden_bonds=None):
    for status1 in status_list1:
        for status2 in status_list2:
            if BondOrderSortedTuplePermutations(
                status1, status2, cg1, cg2, forbidden_bonds=forbidden_bonds
            ).check_non_empty():
                return True
    return False


def matching_frag_size_status_list(
    cg_pair,
    origin_points,
    nhatoms_range=None,
    forbidden_bonds=None,
    smallest_exchange_size=2,
    **frag_size_status_list_kwargs,
):
    unfiltered_frag_size_status_lists = [
        frag_size_status_list(cg, origin_point, **frag_size_status_list_kwargs)
        for cg, origin_point in zip(cg_pair, origin_points)
    ]
    output = []
    for (frag_size1, status_list1), (frag_size2, status_list2) in itertools.product(
        *unfiltered_frag_size_status_lists
    ):
        if (frag_size1 < smallest_exchange_size) and (frag_size2 < smallest_exchange_size):
            continue
        if (frag_size1 > cg_pair[0].nhatoms() - smallest_exchange_size) and (
            frag_size2 > cg_pair[1].nhatoms() - smallest_exchange_size
        ):
            continue
        if not (
            frag_swap_size_compliant(
                cg_pair[0], frag_size1, frag_size2, nhatoms_range=nhatoms_range
            )
            and frag_swap_size_compliant(
                cg_pair[1], frag_size2, frag_size1, nhatoms_range=nhatoms_range
            )
        ):
            continue
        if contains_matching_status(
            *cg_pair, status_list1, status_list2, forbidden_bonds=forbidden_bonds
        ):
            output.append((frag_size1, frag_size2))
    return output


def possible_origin_points(cg: ChemGraph, linear_scaling_crossover_moves=False):
    if linear_scaling_crossover_moves:
        return list(range(cg.nhatoms()))
    else:
        return cg.unrepeated_atom_list()


# TODO would introduction of visited_tps similar to randomized_change help here?
def randomized_crossover(
    cg_pair: list or tuple,
    crossover_smallest_exchange_size=2,
    forbidden_bonds: list or None = None,
    nhatoms_range: list or None = None,
    crossover_max_num_affected_bonds: int or None = None,
    linear_scaling_crossover_moves: bool = True,
    **dummy_kwargs,
):
    """
    Break two ChemGraph objects into two FragmentPair objects; the fragments are then re-coupled into two new ChemGraph objects.
    """
    internal_kwargs = {
        "nhatoms_range": nhatoms_range,
        "forbidden_bonds": forbidden_bonds,
        "smallest_exchange_size": crossover_smallest_exchange_size,
        "max_num_affected_bonds": crossover_max_num_affected_bonds,
    }

    if any(cg.nhatoms() == 1 for cg in cg_pair):
        return None, None

    # Choose "origin points" neighborhoods of which will be marked "red"
    # TODO For now I'm preserving integer multiplication to avoid breaking tests. Summing logarithms should be better though.
    #    log_tot_choice_prob_ratio = 0.0
    tot_choice_prob_ratio = 1.0
    origin_points = []
    for cg in cg_pair:
        cg_possible_origin_points = possible_origin_points(
            cg, linear_scaling_crossover_moves=linear_scaling_crossover_moves
        )
        #        log_tot_choice_prob_ratio -= llenlog(cg_ual)
        tot_choice_prob_ratio /= len(cg_possible_origin_points)
        chosen_origin_point = random.choice(cg_possible_origin_points)
        if linear_scaling_crossover_moves:
            tot_choice_prob_ratio *= atom_multiplicity_in_list(cg, chosen_origin_point)
        origin_points.append(chosen_origin_point)

    # Generate lists containing possible fragment sizes and the corresponding bond status.
    forward_mfssl = matching_frag_size_status_list(cg_pair, origin_points, **internal_kwargs)

    if len(forward_mfssl) == 0:
        return None, None
    #    log_tot_choice_prob_ratio -= llenlog(forward_mfssl)
    tot_choice_prob_ratio /= len(forward_mfssl)
    chosen_sizes = random.choice(forward_mfssl)

    if linear_scaling_crossover_moves:
        (
            new_cg_pair,
            new_origin_points,
            num_reconnectors,
            reconnection_prob_balance,
        ) = crossover_sample_random_outcome(
            cg_pair, chosen_sizes, origin_points, forbidden_bonds=forbidden_bonds
        )
    else:
        new_cg_pairs, new_origin_points = crossover_outcomes(
            cg_pair, chosen_sizes, origin_points, forbidden_bonds=forbidden_bonds
        )
    if new_origin_points is None:
        return None, None

    #    log_tot_choice_prob_ratio -= llenlog(new_cg_pairs)
    if linear_scaling_crossover_moves:
        tot_choice_prob_ratio /= num_reconnectors
    else:
        tot_choice_prob_ratio /= len(new_cg_pairs)
        new_cg_pair = random.choice(new_cg_pairs)

    # Account for inverse choice probability.
    for new_cg, new_origin_point in zip(new_cg_pair, new_origin_points):
        #        log_tot_choice_prob_ratio += llenlog(new_cg.unrepeated_atom_list())
        new_possible_origin_points = possible_origin_points(
            new_cg, linear_scaling_crossover_moves=linear_scaling_crossover_moves
        )
        tot_choice_prob_ratio *= len(new_possible_origin_points)
        if linear_scaling_crossover_moves:
            tot_choice_prob_ratio /= atom_multiplicity_in_list(new_cg, new_origin_point)

    inverse_mfssl = matching_frag_size_status_list(
        new_cg_pair, new_origin_points, **internal_kwargs
    )

    tot_choice_prob_ratio *= len(inverse_mfssl)

    if linear_scaling_crossover_moves:
        valid_reconnectors, _, _, _ = matching_status_reconnectors_wfrags(
            new_cg_pair, new_origin_points, chosen_sizes
        )
        tot_choice_prob_ratio *= len(valid_reconnectors)
    else:
        inverse_cg_pairs, _ = crossover_outcomes(
            new_cg_pair,
            chosen_sizes,
            new_origin_points,
            forbidden_bonds=forbidden_bonds,
        )
        #    log_tot_choice_prob_ratio += llenlog(inverse_cg_pairs)
        tot_choice_prob_ratio *= len(inverse_cg_pairs)

    log_tot_choice_prob_ratio = np.log(tot_choice_prob_ratio)
    if np.isinf(log_tot_choice_prob_ratio):
        if VERBOSITY != VERBOSITY_MUTED:
            print("NONINVERTIBLE CROSS-COUPLING PROPOSED:")
            print("INITIAL CHEMGRAPHS:", cg_pair)
            print("PROPOSED CHEMGRAPHS:", new_cg_pair)
        raise InvalidChange

    if linear_scaling_crossover_moves:
        log_tot_choice_prob_ratio += reconnection_prob_balance

    return new_cg_pair, log_tot_choice_prob_ratio
