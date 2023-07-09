# Everything related to cross-couplings.
from .valence_treatment import (
    ChemGraph,
    sorted_by_membership,
    sorted_tuple,
    connection_forbidden,
    max_bo,
)
import numpy as np
import random, bisect, itertools
from igraph.operators import disjoint_union
from copy import deepcopy


class Frag2FragMapping:
    def __init__(self, new_membership_vector, old_membership_vector, frag_id=0):
        """
        Auxiliary class for mapping HeavyAtom objects from the same fragment used for two FragmentPair objects.
        """
        self.old_frag_ids = np.where(old_membership_vector == frag_id)[0]
        self.new_frag_ids = np.where(new_membership_vector == frag_id)[0]

    def __call__(self, old_id):
        inside_id = bisect.bisect_left(self.old_frag_ids, old_id)
        return self.new_frag_ids[inside_id]


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

        self.chemgraph.init_resonance_structures()

        self.origin_point = origin_point
        self.neighborhood_size = neighborhood_size

        # constants for how different parts of FragmentPair are marked in self.membership_vector
        self.core_membership_vector_value = 0
        self.remainder_membership_vector_value = 1
        self.border_membership_vector_value = -1

        self.membership_vector = np.repeat(
            self.remainder_membership_vector_value, cg.nhatoms()
        )
        self.membership_vector[origin_point] = self.core_membership_vector_value

        self.affected_bonds = [
            (self.origin_point, neigh)
            for neigh in self.chemgraph.neighbors(origin_point)
        ]

        while self.core_size() < neighborhood_size:
            self.expand_core(update_affected_status=False)

        # Two list of vertices corresponding to the two grahments.
        self.sorted_vertices = None
        self.init_affected_status_info()

    def expand_core(self, update_affected_status=True):
        for old_affected_bond in self.affected_bonds:
            new_id = old_affected_bond[1]
            if self.is_remainder(new_id):
                self.membership_vector[new_id] = self.border_membership_vector_value
        self.affected_bonds = []
        for border_id in np.where(
            self.membership_vector == self.border_membership_vector_value
        )[0]:
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
        self.affected_resonance_structures = []
        resonance_structure_orders_iterators = []

        resonance_structure_affected_bonds = {}
        saved_all_bond_orders = {}

        default_bond_order_dict = {}

        for bond_tuple in self.affected_bonds:
            bond_stuple = sorted_tuple(*bond_tuple)
            if bond_stuple in self.chemgraph.resonance_structure_map:
                rsr_id = self.chemgraph.resonance_structure_map[bond_stuple]
                if rsr_id not in saved_all_bond_orders:
                    saved_all_bond_orders[rsr_id] = {}
                    self.affected_resonance_structures.append(rsr_id)
                    resonance_structure_orders_iterators.append(
                        range(len(self.chemgraph.resonance_structure_orders[rsr_id]))
                    )
                if rsr_id not in resonance_structure_affected_bonds:
                    resonance_structure_affected_bonds[rsr_id] = []
                resonance_structure_affected_bonds[rsr_id].append(bond_tuple)
                saved_all_bond_orders[rsr_id][
                    bond_tuple
                ] = self.chemgraph.aa_all_bond_orders(*bond_stuple, unsorted=True)

            else:
                cur_bo = self.chemgraph.bond_orders[bond_stuple]
                if cur_bo in default_bond_order_dict:
                    default_bond_order_dict[cur_bo].append(bond_tuple)
                else:
                    default_bond_order_dict[cur_bo] = [bond_tuple]

        if len(resonance_structure_orders_iterators) == 0:
            self.affected_status = [
                FragmentPairAffectedBondStatus(default_bond_order_dict)
            ]
            return

        self.affected_status = []
        for resonance_structure_orders_ids in itertools.product(
            *resonance_structure_orders_iterators
        ):
            new_status = FragmentPairAffectedBondStatus(
                deepcopy(default_bond_order_dict)
            )
            for res_reg_id, rso_id in zip(
                self.affected_resonance_structures, resonance_structure_orders_ids
            ):
                ha_ids = self.chemgraph.resonance_structure_inverse_map[res_reg_id]
                val_pos = self.chemgraph.resonance_structure_valence_vals[res_reg_id][
                    rso_id
                ]
                for ha_id in ha_ids:
                    poss_valences = self.chemgraph.hatoms[ha_id].possible_valences
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
            self.chemgraph.hatoms[ha_id].mincopy()
            for ha_id in self.get_sorted_vertices(frag_id)
        ]

    def get_frag_subgraph(self, frag_id):
        return self.chemgraph.graph.subgraph(self.get_sorted_vertices(frag_id))

    def cross_couple(
        self,
        other_fp,
        switched_bond_tuples_self: int,
        switched_bond_tuples_other: int,
        affected_status_id_self: int,
        affected_status_id_other: int,
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

        self_to_new = Frag2FragMapping(
            new_membership_vector, self.membership_vector, frag_id=frag_id_self
        )
        other_to_new = Frag2FragMapping(
            new_membership_vector, other_fp.membership_vector, frag_id=frag_id_other
        )

        # Check which bonds need to be created.
        created_bonds = []

        for btuple_self, btuple_other in zip(
            switched_bond_tuples_self, switched_bond_tuples_other
        ):
            internal_id1 = self_to_new(btuple_self[0])
            internal_id2 = other_to_new(btuple_other[1])

            new_bond_tuple = (internal_id1, internal_id2)
            if new_bond_tuple in created_bonds:
                # This is done to prevent non-invertible formation of a bond of non-unity order.
                return None, None
            else:
                created_bonds.append(new_bond_tuple)

        # "Sew" two graphs together with the created bonds.
        new_graph = disjoint_union(
            [
                self.get_frag_subgraph(frag_id_self),
                other_fp.get_frag_subgraph(frag_id_other),
            ]
        )

        for new_bond_tuple in created_bonds:
            new_graph.add_edge(*new_bond_tuple)

        new_hatoms = self.get_hatoms_sublist(
            frag_id_self
        ) + other_fp.get_hatoms_sublist(frag_id_other)

        new_ChemGraph = ChemGraph(hatoms=new_hatoms, graph=new_graph)

        new_hatoms_old_valences = self.adjusted_ha_valences(
            affected_status_id_self, frag_id_self
        ) + other_fp.adjusted_ha_valences(affected_status_id_other, frag_id_other)

        # Lastly, check that re-initialization does not decrease the bond order.
        if new_ChemGraph.valence_config_valid(new_hatoms_old_valences):
            return new_ChemGraph, new_membership_vector
        else:
            return None, None


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
        self.non_empty = False
        self.bo_tuple_dict1 = status1.bond_tuple_dict
        self.bo_tuple_dict2 = status2.bond_tuple_dict
        # Check that the two dictionnaries are of similar dimensionality.
        if len(self.bo_tuple_dict1) != len(self.bo_tuple_dict2):
            return
        for bo1, bo_tuples1 in self.bo_tuple_dict1.items():
            if bo1 not in self.bo_tuple_dict2:
                return
            if len(self.bo_tuple_dict2[bo1]) != len(bo_tuples1):
                return
        self.non_empty = True

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


def cross_couple_outcomes(cg_pair, chosen_sizes, origin_points, forbidden_bonds=None):
    frag1 = FragmentPair(
        cg_pair[0], origin_points[0], neighborhood_size=chosen_sizes[0]
    )
    frag2 = FragmentPair(
        cg_pair[1], origin_points[1], neighborhood_size=chosen_sizes[1]
    )

    new_chemgraph_pairs = []
    new_origin_points = None
    for status_id1, status1 in enumerate(frag1.affected_status):
        for status_id2, status2 in enumerate(frag2.affected_status):
            for tuples1, tuples2 in BondOrderSortedTuplePermutations(
                status1, status2, *cg_pair, forbidden_bonds=forbidden_bonds
            ):
                new_chemgraph_1, new_membership_vector_1 = frag1.cross_couple(
                    frag2, tuples1, tuples2, status_id1, status_id2
                )
                new_chemgraph_2, new_membership_vector_2 = frag2.cross_couple(
                    frag1, tuples2, tuples1, status_id2, status_id1
                )
                if (new_membership_vector_1 is None) or (
                    new_membership_vector_2 is None
                ):
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


def frag_swap_size_compliant(
    cg, frag_size_leaving, frag_size_adding, nhatoms_range=None
):
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


def frag_size_status_list(cg, origin_point, max_num_affected_bonds=3):
    """
    Map all possible ways a ChemGraph can be broken into FragmentPair satisfying input constraints.
    """
    frag_size_bounds = possible_fragment_size_bounds(cg)
    output = []
    temp_fp = FragmentPair(cg, origin_point)
    while temp_fp.core_size() <= frag_size_bounds[1]:
        if temp_fp.core_size() >= frag_size_bounds[0]:
            if len(temp_fp.affected_bonds) <= max_num_affected_bonds:
                output.append((temp_fp.core_size(), temp_fp.affected_status))
        temp_fp.expand_core()
    return output


def contains_matching_status(
    cg1, cg2, status_list1, status_list2, forbidden_bonds=None
):
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
        if (frag_size1 < smallest_exchange_size) and (
            frag_size2 < smallest_exchange_size
        ):
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


# TODO would introduction of visited_tps similar to randomized_change help here?
def randomized_cross_coupling(
    cg_pair: list or tuple,
    cross_coupling_smallest_exchange_size=2,
    forbidden_bonds: list or None = None,
    nhatoms_range: list or None = None,
    cross_coupling_max_num_affected_bonds: int = 3,
    **dummy_kwargs,
):
    """
    Break two ChemGraph objects into two FragmentPair objects; the fragments are then re-coupled into two new ChemGraph objects.
    """
    internal_kwargs = {
        "nhatoms_range": nhatoms_range,
        "forbidden_bonds": forbidden_bonds,
        "smallest_exchange_size": cross_coupling_smallest_exchange_size,
        "max_num_affected_bonds": cross_coupling_max_num_affected_bonds,
    }

    if any(cg.nhatoms() == 1 for cg in cg_pair):
        return None, None

    # Choose "origin points" neighborhoods of which will be marked "red"
    tot_choice_prob_ratio = 1.0
    origin_points = []
    for cg in cg_pair:
        cg_ual = cg.unrepeated_atom_list()
        tot_choice_prob_ratio /= len(cg_ual)
        origin_points.append(random.choice(cg_ual))

    # Generate lists containing possible fragment sizes and the corresponding bond status.
    forward_mfssl = matching_frag_size_status_list(
        cg_pair, origin_points, **internal_kwargs
    )

    if len(forward_mfssl) == 0:
        return None, None
    tot_choice_prob_ratio /= len(forward_mfssl)
    chosen_sizes = random.choice(forward_mfssl)

    new_cg_pairs, new_origin_points = cross_couple_outcomes(
        cg_pair, chosen_sizes, origin_points, forbidden_bonds=forbidden_bonds
    )
    if new_origin_points is None:
        return None, None

    tot_choice_prob_ratio /= len(new_cg_pairs)

    new_cg_pair = random.choice(new_cg_pairs)

    # Account for inverse choice probability.
    for new_cg in new_cg_pair:
        tot_choice_prob_ratio *= len(new_cg.unrepeated_atom_list())

    inverse_mfssl = matching_frag_size_status_list(
        new_cg_pair, new_origin_points, **internal_kwargs
    )

    tot_choice_prob_ratio *= len(inverse_mfssl)

    inverse_cg_pairs, _ = cross_couple_outcomes(
        new_cg_pair, chosen_sizes, new_origin_points, forbidden_bonds=forbidden_bonds
    )
    tot_choice_prob_ratio *= len(inverse_cg_pairs)

    try:
        log_tot_choice_prob_ratio = np.log(tot_choice_prob_ratio)
    except:
        print("NONINVERTIBLE CROSS-COUPLING PROPOSED:")
        print("INITIAL CHEMGRAPHS:", cg_pair)
        print("PROPOSED CHEMGRAPHS:", new_cg_pair)
        quit()

    return new_cg_pair, log_tot_choice_prob_ratio
