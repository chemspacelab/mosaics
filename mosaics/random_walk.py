# TODO For forward and backward probabilities, comment more on where different signs come from.
# TODO 1. Account for statistical noise of input data. 2. Many theory levels?
# TODO Somehow unify all instances where simulation histogram is modified.
# TODO sorted betas + parallel tempering just neighboring points as an option?

# TODO add a toy problem with a history-dependent minimized function to check that re-import from restart works properly.
# TODO add a restart example with dumped histogram.

# TODO check that linear_storage option is used correctly everywhere (should it be used consistently?)

# TODO separate MC_step counters for different MC move types?

import operator
import os
import random
from copy import deepcopy
from types import FunctionType
from typing import Union

import numpy as np
from scipy.special import expit
from sortedcontainers import SortedDict, SortedList

from .chem_graph import InvalidChange, str2ChemGraph
from .chem_graph.heavy_atom import default_valence, int_atom_checked
from .crossover import randomized_crossover
from .elementary_mutations import (
    add_heavy_atom_chain,
    available_added_atom_bos,
    change_valence,
    gen_atom_removal_possible_hnums,
    gen_val_change_pos_ncharges,
    remove_heavy_atom,
)
from .ext_graph_compound import egc_valid_wrt_change_params
from .misc_procedures import VERBOSITY, VERBOSITY_MUTED, sorted_tuple
from .modify import (
    TrajectoryPoint,
    full_change_list,
    global_step_traj_storage_label,
    inverse_procedure,
    nonglobal_step_traj_storage_label,
    randomized_change,
)
from .trajectory_analysis import ordered_trajectory, ordered_trajectory_ids
from .utils import dump2pkl, loadpkl, pkl_compress_ending

default_minfunc_name = "MIN_FUNC_NAME"


class CandidateCompound:
    def __init__(self, tp: TrajectoryPoint, func_val: float):
        """
        Auxiliary class for more convenient maintenance of candidate compound list.
        NOTE: The comparison operators are not strictly transitive, but work well enough for maintaining a candidate list.
        tp : Trajectory point object.
        func_val : value of the minimized function.
        """
        self.tp = tp
        self.func_val = func_val

    def __eq__(self, cc2):
        return compare_candidates(self, cc2, operator.eq)

    def __gt__(self, cc2):
        return compare_candidates(self, cc2, operator.gt)

    def __lt__(self, cc2):
        return compare_candidates(self, cc2, operator.lt)

    def __str__(self):
        return (
            "(CandidateCompound,func_val:"
            + str(self.func_val)
            + ",ChemGraph:"
            + str(self.tp.egc.chemgraph)
            + ")"
        )

    def __repr__(self):
        return str(self)


def compare_candidates(
    cand1: CandidateCompound, cand2: CandidateCompound, comparison_operator
) -> bool:
    """
    Perform comparison on two CandidateCompound objects.
    """
    if (cand1.tp == cand2.tp) or (cand1.func_val == cand2.func_val):
        return comparison_operator(cand1.tp, cand2.tp)
    return comparison_operator(cand1.func_val, cand2.func_val)


def maintain_sorted_CandidateCompound_list(
    obj,
    trajectory_point: TrajectoryPoint = None,
    min_func_val=None,
    candidate: CandidateCompound = None,
):
    """
    A subroutine shared between RandomWalk and DistributedRandomWalk objects for maintaining their saved CandidateCompound lists.
    """
    if (obj.num_saved_candidates is None) and (obj.saved_candidates_max_difference is None):
        return
    if candidate is not None:
        min_func_val = candidate.func_val

    if min_func_val is None:
        return

    if candidate is None:
        candidate = CandidateCompound(tp=deepcopy(trajectory_point), func_val=min_func_val)

    if (obj.num_saved_candidates is not None) and (
        len(obj.saved_candidates) >= obj.num_saved_candidates
    ):
        if min_func_val > obj.saved_candidates[-1].func_val:
            return
    if obj.saved_candidates_max_difference is not None:
        new_lower_minfunc_bound = None
        if len(obj.saved_candidates) != 0:
            if (
                min_func_val - obj.saved_candidates[0].func_val
                > obj.saved_candidates_max_difference
            ):
                return
            if min_func_val < obj.saved_candidates[0].func_val:
                new_lower_minfunc_bound = min_func_val

    if candidate in obj.saved_candidates:
        return

    obj.saved_candidates.add(candidate)

    if (obj.num_saved_candidates is not None) and (
        len(obj.saved_candidates) > obj.num_saved_candidates
    ):
        del obj.saved_candidates[obj.num_saved_candidates :]

    if obj.saved_candidates_max_difference is not None:
        if new_lower_minfunc_bound is not None:
            # Delete tail candidates with too large minimized function values.
            new_upper_bound = new_lower_minfunc_bound + obj.saved_candidates_max_difference
            deleted_indices_bound = None
            for i, cand in enumerate(obj.saved_candidates):
                if cand.func_val > new_upper_bound:
                    deleted_indices_bound = i
                    break
            if deleted_indices_bound is not None:
                del obj.saved_candidates[deleted_indices_bound:]


def str2CandidateCompound(str_in):
    stripped_string = str_in.strip()
    spl_stripped_string = stripped_string.split(",ChemGraph:")
    chemgraph_str = spl_stripped_string[1][:-1]
    chemgraph = str2ChemGraph(chemgraph_str)
    func_val_str = spl_stripped_string[0].split(",func_val:")[1]
    func_val = eval(func_val_str)
    tp = TrajectoryPoint(cg=chemgraph, func_val=func_val)
    return CandidateCompound(tp, func_val)


def tidy_forbidden_bonds(forbidden_bonds):
    if forbidden_bonds is None:
        return None
    output = SortedList()
    for fb in forbidden_bonds:
        output.add(sorted_tuple(*fb))
    return output


#   Auxiliary exception classes.
class SoftExitCalled(Exception):
    """
    Exception raised by RandomWalk object if soft exit request is passed.
    """


class InvalidStartingMolecules(Exception):
    """
    Exception raised by RandomWalk object if it is initialized with starting molecules that do not fit the parameters for random change.
    """


# TODO: The expression appears in distributed_random_walk once, but not in RandomWalk class itself, the latter using alternative expression.
# Using  Metropolis_acceptance_probability in RandomWalk instead might be better for readability, not sure.
def Metropolis_acceptance_probability(log_prob_balance):
    """
    Metropolis acceptance probability for a given balance of logarithm of proposition probability + effective potential function.
    """
    if log_prob_balance > 0.0:
        return 1.0
    return np.exp(log_prob_balance)


class RandomWalk:
    def __init__(
        self,
        init_egcs: list or None = None,
        init_tps: list or None = None,
        bias_coeff: float or None = None,
        vbeta_bias_coeff: float or None = None,
        bias_pot_all_replicas: bool = True,
        randomized_change_params: dict = {},
        starting_histogram: list or None = None,
        conserve_stochiometry: bool = False,
        bound_enforcing_coeff: float or None = None,
        keep_histogram: bool = False,
        histogram_save_rejected: bool = True,
        betas: list or None = None,
        min_function: FunctionType or None = None,
        no_min_function_lookup: bool = False,
        num_replicas: int or None = None,
        no_exploration: bool = False,
        restricted_tps: list or None = None,
        min_function_name: str = default_minfunc_name,
        num_saved_candidates: int or None = None,
        saved_candidates_max_difference: float or None = None,
        keep_full_trajectory: bool = False,
        keep_detailed_global_steps: bool = False,
        restart_file: str or None = None,
        make_restart_frequency: int or None = None,
        soft_exit_check_frequency: int or None = None,
        delete_temp_data: list or None = None,
        max_histogram_size: int or None = None,
        histogram_dump_portion: float = 0.5,
        histogram_dump_file_prefix: str = "",
        track_histogram_size: bool = False,
        visit_num_count_acceptance: bool = False,
        linear_storage: bool = True,
        compress_restart: bool = False,
        greedy_delete_checked_paths: bool = False,
        canonize_trajectory_points: bool = False,
        debug: bool = False,
    ):
        """
        Class that generates a trajectory over chemical space.
        init_egcs : initial positions of the simulation, in ExtGraphCompound format.
        init_tps : initial positions of the simulation, in TrajectoryPoint format.
        betas : values of beta used in the extended tempering ensemble; "None" corresponds to a virtual beta (greedily minimized replica).
        bias_coeff : biasing potential applied to push real beta replicas out of local minima
        vbeta_bias_coeff : biasing potential applied to push virtual beta replicas out of local minima
        bias_pot_all_replicas : whether the biasing potential is calculated from sum of visits of all replicas rather than the replica considered
        bound_enforcing_coeff : biasing coefficient for "soft constraints"; currently the only one functioning properly allows biasing simulation run in nhatoms_range towards nhatoms values in final_nhatoms_range (see randomized_change_params)
        min_function : minimized function
        min_function_name : name of minimized function (the label used for minimized function value in TrajectoryPoint object's calculated_data)
        no_min_function_lookup : the code evaluates the minimized function without checking whether it had been calculated for the property of interest before
        keep_histogram : store information about all considered molecules; mandatory for using biasing potentials
        histogram_save_rejected : if True then both accepted and rejected chemical graphs are saved into the histogram
        num_saved_candidates : if not None determines how many best candidates are kept in the saved_candidates attributes
        saved_candidates_max_difference : if not None determines maximum difference of minimized function values for the best and worst of saved best candidates
        keep_full_trajectory : save not just number of times a trajectory point was visited, but also all steps ids when the step was made
        keep_detailed_global_steps : save information of trajectory movement up to non-global steps
        restart_file : name of restart file to which the object is dumped at make_restart
        make_restart_frequency : if not None object will call make_restart each make_restart_frequency global steps.
        soft_exit_check_frequency : if not None the code will check presence of "EXIT" in the running directory each soft_exit_check_frequency steps; if "EXIT" exists the object calls make_restart and raises SoftExitCalled
        delete_temp_data : if not None after each minimized function evaluation for a TrajectoryPoint object self will delete TrajectoryPoint's calculated_data fields with those identifiers.
        max_histogram_size : if not None sets the maximal size for the histogram that, when exceeded, triggers dumping the histogram
        histogram_dump_portion : defines how much of the histogram is dumped if it exceeds the current size.
        histogram_dump_file_prefix : sets the prefix from which the name of the pickle file where histogram is dumped if its maximal size is exceeded
        track_histogram_size : print current size of the histogram after each global MC step
        visit_num_count_acceptance : if True number of visit numbers (used in biasing potential) is counted during each accept_reject_move call rather than each global step
        linear_storage : whether objects saved to the histogram contain data whose size scales more than linearly with molecule size
        compress_restart : whether restart files are compressed by default
        randomized_change_params : parameters defining the sampled chemical space and how the sampling is done; see description of init_randomized_params for more thorough explanation.
        greedy_delete_checked_paths : for greedy replicas take a modification path with simple moves only once.
        canonize_trajectory_points : keep all trajectory points canonized; only useful for testing purposes.
        debug : simulation is run with extra checks (testing purposes).
        """
        self.num_replicas = num_replicas
        self.betas = betas
        if self.num_replicas is None:
            if self.betas is not None:
                self.num_replicas = len(self.betas)
            else:
                if isinstance(init_egcs, list):
                    self.num_replicas = len(init_egcs)
                else:
                    self.num_replicas = 1

        if self.betas is not None:
            if self.num_replicas == 1:
                self.all_betas_same = True
            else:
                self.all_betas_same = all(
                    self.betas_same([0, other_beta_id])
                    for other_beta_id in range(1, self.num_replicas)
                )

        self.keep_full_trajectory = keep_full_trajectory
        self.keep_detailed_global_steps = keep_detailed_global_steps

        self.MC_step_counter = 0
        self.global_MC_step_counter = 0

        if isinstance(self.betas, list):
            assert len(self.betas) == self.num_replicas

        self.keep_histogram = keep_histogram
        self.histogram_save_rejected = histogram_save_rejected
        self.visit_num_count_acceptance = visit_num_count_acceptance
        self.linear_storage = linear_storage

        self.no_exploration = no_exploration
        if self.no_exploration:
            if restricted_tps is None:
                raise Exception
            else:
                self.restricted_tps = restricted_tps

        self.bias_coeff = bias_coeff
        self.vbeta_bias_coeff = vbeta_bias_coeff
        self.bias_pot_all_replicas = bias_pot_all_replicas

        self.randomized_change_params = randomized_change_params
        self.init_randomized_change_params(randomized_change_params)

        self.bound_enforcing_coeff = bound_enforcing_coeff

        # TODO previous implementation deprecated; perhaps re-implement as a ``soft'' constraint?
        # Or make a special dictionnary of changes that conserve stochiometry.
        self.conserve_stochiometry = conserve_stochiometry

        self.hydrogen_nums = None

        self.min_function = min_function
        self.min_function_name = min_function_name
        if self.min_function is not None:
            self.min_function_dict = {self.min_function_name: self.min_function}
        self.no_min_function_lookup = no_min_function_lookup
        self.num_saved_candidates = num_saved_candidates
        self.saved_candidates_max_difference = saved_candidates_max_difference
        if (self.num_saved_candidates is not None) or (
            self.saved_candidates_max_difference is not None
        ):
            self.saved_candidates = SortedList()

        # For storing statistics on move success.
        self.num_attempted_crossovers = 0
        self.num_valid_crossovers = 0
        self.num_accepted_crossovers = 0

        self.num_attempted_simple_moves = 0
        self.num_valid_simple_moves = 0
        self.num_accepted_simple_moves = 0

        self.num_attempted_tempering_swaps = 0
        self.num_accepted_tempering_swaps = 0

        self.moves_since_changed = np.zeros((self.num_replicas,), dtype=int)

        # Related to making restart files and checking for soft exit.
        self.restart_file = restart_file
        self.make_restart_frequency = make_restart_frequency
        self.compress_restart = compress_restart
        self.soft_exit_check_frequency = soft_exit_check_frequency
        self.global_steps_since_last = {}

        self.max_histogram_size = max_histogram_size
        self.histogram_dump_portion = histogram_dump_portion
        self.histogram_dump_file_prefix = histogram_dump_file_prefix
        self.track_histogram_size = track_histogram_size

        # Histogram initialization.
        if starting_histogram is None:
            if self.keep_histogram:
                self.histogram = SortedList()
            else:
                self.histogram = None
        else:
            self.histogram = starting_histogram

        self.delete_temp_data = delete_temp_data
        self.greedy_delete_checked_paths = greedy_delete_checked_paths

        self.debug = debug
        self.canonize_trajectory_points = canonize_trajectory_points

        self.init_cur_tps(init_egcs=init_egcs, init_tps=init_tps)

    def init_randomized_change_params(self, randomized_change_params=None):
        """
        Initialize parameters for how the chemical space is sampled. If randomized_change_params is None do nothing; otherwise it is a dictionnary with following entries:
        change_prob_dict : which changes are used simple MC moves (see full_change_list, minimized_change_list, and valence_ha_change_list global variables for examples).
        possible_elements : symbols of heavy atom elements that can be found inside molecules.
        added_bond_orders : as atoms are added to the molecule they can be connected to it with bonds of an order in added_bond_orders
        chain_addition_tuple_possibilities : minor parameter choosing the procedure for how heavy atom sceleton is grown. Setting it to "False" (the default value) should accelerate discovery rates.
        forbidden_bonds : nuclear charge pairs that are forbidden to connect with a covalent bond.
        not_protonated : nuclear charges of atoms that should not be covalently connected to hydrogens.
        bond_order_changes : by how much a bond can change during a simple MC step (e.g. [-1, 1]).
        bond_order_valence_changes : by how much a bond can change during steps that change bond order and valence of an atom (e.g. [-2, 2]).
        max_fragment_num : how many disconnected fragments (e.g. molecules) a chemical graph is allowed to break into.
        added_bond_orders_val_change : when creating atoms to be connected to a molecule's atom with a change of valence of the latter what the possible bond orders are.
        crossover_max_num_affected_bonds : when a molecule is broken into two fragments what is the maximum number of bonds that can be broken.
        crossover_smallest_exchange_size : do not perform a cross-coupling move if less than this number of atoms is exchanged on both sides.
        linear_scaling_elementary_mutations : O(nhatoms) scaling of elementary mutations at the cost of not accounting for graph invariance at trial step generation.
        linear_scaling_crossover_moves : O(nhatoms) scaling of crossover moves at the cost of not accounting for graph invariance and forbidden bonds at trial step generation.
        save_equivalence_data :
        """
        if randomized_change_params is not None:
            self.randomized_change_params = randomized_change_params
            # used_randomized_change_params contains randomized_change_params as well as some temporary data to optimize code's performance.
            self.used_randomized_change_params = deepcopy(self.randomized_change_params)
            if "forbidden_bonds" in self.used_randomized_change_params:
                self.used_randomized_change_params["forbidden_bonds"] = tidy_forbidden_bonds(
                    self.used_randomized_change_params["forbidden_bonds"]
                )

            self.used_randomized_change_params_check_defaults(
                check_kw_validity=True,
                change_prob_dict=full_change_list,
                possible_elements=["C"],
                forbidden_bonds=None,
                not_protonated=None,
                added_bond_orders=[1],
                chain_addition_tuple_possibilities=False,
                bond_order_changes=[-1, 1],
                bond_order_valence_changes=[-2, 2],
                nhatoms_range=None,
                final_nhatoms_range=None,
                max_fragment_num=1,
                added_bond_orders_val_change=[1, 2],
                crossover_max_num_affected_bonds=None,
                crossover_smallest_exchange_size=2,
                linear_scaling_elementary_mutations=True,
                linear_scaling_crossover_moves=True,
                save_equivalence_data=False,
            )

            # Some convenient aliases.
            cur_change_dict = self.used_randomized_change_params["change_prob_dict"]

            # Initialize some auxiliary arguments that allow the code to run just a little faster.
            cur_added_bond_orders = self.used_randomized_change_params["added_bond_orders"]
            cur_not_protonated = self.used_randomized_change_params["not_protonated"]
            cur_possible_elements = self.used_randomized_change_params["possible_elements"]
            cur_possible_ncharges = [
                int_atom_checked(possible_element) for possible_element in cur_possible_elements
            ]
            cur_default_valences = {}
            for ncharge in cur_possible_ncharges:
                cur_default_valences[ncharge] = default_valence(ncharge)

            # Check that each change operation has an inverse.
            for change_func in cur_change_dict:
                inv_change_func = inverse_procedure[change_func]
                if inv_change_func not in cur_change_dict:
                    if VERBOSITY != VERBOSITY_MUTED:
                        print(
                            "WARNING, inverse not found in randomized_change_params for:",
                            change_func,
                            " adding inverse.",
                        )
                    if isinstance(cur_change_dict, dict):
                        cur_change_dict[inv_change_func] = cur_change_dict[change_func]
                    else:
                        cur_change_dict.append(inv_change_func)

            # Initialize some auxiliary arguments that allow the code to run just a little faster.
            self.used_randomized_change_params["possible_ncharges"] = cur_possible_ncharges
            self.used_randomized_change_params["default_valences"] = cur_default_valences

            if (add_heavy_atom_chain in cur_change_dict) or (remove_heavy_atom in cur_change_dict):
                avail_added_bond_orders = {}
                atom_removal_possible_hnums = {}
                for ncharge, def_val in zip(cur_possible_ncharges, cur_default_valences.values()):
                    avail_added_bond_orders[ncharge] = available_added_atom_bos(
                        ncharge,
                        cur_added_bond_orders,
                        not_protonated=cur_not_protonated,
                    )
                    atom_removal_possible_hnums[ncharge] = gen_atom_removal_possible_hnums(
                        cur_added_bond_orders, def_val
                    )
                self.used_randomized_change_params_check_defaults(
                    avail_added_bond_orders=avail_added_bond_orders,
                    atom_removal_possible_hnums=atom_removal_possible_hnums,
                )

            if change_valence in cur_change_dict:
                self.used_randomized_change_params[
                    "val_change_poss_ncharges"
                ] = gen_val_change_pos_ncharges(
                    cur_possible_elements, not_protonated=cur_not_protonated
                )

    def used_randomized_change_params_check_defaults(
        self, check_kw_validity=False, **other_kwargs
    ):
        if check_kw_validity:
            for kw in self.used_randomized_change_params:
                if kw not in other_kwargs:
                    raise Exception("Randomized change parameter ", kw, " is invalid.")
        for kw, def_val in other_kwargs.items():
            if kw not in self.used_randomized_change_params:
                self.used_randomized_change_params[kw] = def_val

    def init_cur_tps(self, init_egcs=None, init_tps=None):
        """
        Set current positions of self's trajectory from init_egcs while checking that the resulting trajectory points are valid.
        init_egcs : a list of ExtGraphCompound objects to be set as positions; if None the procedure terminates without doing anything.
        """
        if init_tps is None:
            if init_egcs is None:
                return
            init_tps = [TrajectoryPoint(egc=egc) for egc in init_egcs]
        self.cur_tps = []
        for replica_id, added_tp in enumerate(init_tps):
            if not self.egc_valid_wrt_change_params(added_tp.egc):
                raise InvalidStartingMolecules
            if self.canonize_trajectory_points:
                added_tp.canonize_chemgraph()
            if self.no_exploration:
                if added_tp not in self.restricted_tps:
                    raise InvalidStartingMolecules
            #            self.hist_check_tp(added_tp)
            added_tp = self.hist_checked_tp(added_tp)
            if self.min_function is not None:
                # Initialize the minimized function's value in the new trajectory point and check that it is not None
                cur_min_func_val = self.eval_min_func(added_tp, replica_id)
                if cur_min_func_val is None:
                    raise InvalidStartingMolecules
            self.cur_tps.append(added_tp)
            self.update_saved_candidates(added_tp)
        self.update_histogram(list(range(self.num_replicas)))
        self.update_global_histogram()

    def egc_valid_wrt_change_params(self, egc):
        """
        Check that egc is valid with respect to chemical space specifications.
        """
        return egc_valid_wrt_change_params(egc, **self.randomized_change_params)

    # Acceptance rejection rules.
    def accept_reject_move(self, new_tps, prob_balance, replica_ids=[0], swap=False):
        if self.debug:
            self.check_move_validity(new_tps, replica_ids)
        if self.canonize_trajectory_points:
            for i in range(len(new_tps)):
                new_tps[i].canonize_chemgraph()

        self.MC_step_counter += 1

        accepted = self.acceptance_rule(new_tps, prob_balance, replica_ids=replica_ids)
        if accepted:
            for new_tp, replica_id in zip(new_tps, replica_ids):
                self.cur_tps[replica_id] = new_tp
                self.moves_since_changed[replica_id] = 0
        else:
            for new_tp, replica_id in zip(new_tps, replica_ids):
                self.moves_since_changed[replica_id] += 1
                if swap:
                    continue
                if self.keep_histogram and self.histogram_save_rejected:
                    tp_in_histogram = new_tp in self.histogram
                    if tp_in_histogram:
                        tp_index = self.histogram.index(new_tp)
                        new_tp.copy_extra_data_to(
                            self.histogram[tp_index], omit_data=self.delete_temp_data
                        )
                    else:
                        self.add_to_histogram(new_tp, replica_id)

        if self.keep_histogram:
            self.update_histogram(replica_ids)

        return accepted

    def check_move_validity(self, new_tps, replica_ids):
        """
        Introduced for testing purposes.
        """
        for new_tp in new_tps:
            if not self.egc_valid_wrt_change_params(new_tp.egc):
                print("INCONSISTENT TRAJECTORY POINTS")
                print("INITIAL:")
                print(*[self.cur_tps[replica_id] for replica_id in replica_ids])
                print("PROPOSED:")
                print(*new_tps)
                raise InvalidChange

    def acceptance_rule(self, new_tps, prob_balance, replica_ids=[0]):
        if self.no_exploration:
            for new_tp in new_tps:
                if new_tp not in self.restricted_tps:
                    return False
        new_tot_pot_vals = [
            self.tot_pot(new_tp, replica_id) for new_tp, replica_id in zip(new_tps, replica_ids)
        ]

        # Check we have not created any invalid molecules.
        if None in new_tot_pot_vals:
            return False

        prev_tot_pot_vals = [
            self.tot_pot(self.cur_tps[replica_id], replica_id) for replica_id in replica_ids
        ]

        if (self.betas is not None) and self.virtual_beta_present(replica_ids):
            vnew_tot_pot_vals = []
            vprev_tot_pot_vals = []
            for replica_id, new_tot_pot_val, prev_tot_pot_val in zip(
                replica_ids, new_tot_pot_vals, prev_tot_pot_vals
            ):
                if self.virtual_beta_id(replica_id):
                    vnew_tot_pot_vals.append(new_tot_pot_val)
                    vprev_tot_pot_vals.append(prev_tot_pot_val)
            vnew_tot_pot_vals.sort()
            vprev_tot_pot_vals.sort()
            for vnew_tpv, vprev_tpv in zip(vnew_tot_pot_vals, vprev_tot_pot_vals):
                if vnew_tpv != vprev_tpv:
                    return vnew_tpv < vprev_tpv

        delta_pot = prob_balance + sum(new_tot_pot_vals) - sum(prev_tot_pot_vals)

        if delta_pot <= 0.0:
            return True
        else:
            #            Uncommenting this line make the code consistent with old tests.
            #            return -np.log(np.random.random()) > delta_pot
            return np.random.exponential() > delta_pot

    def virtual_beta_present(self, beta_ids):
        return any(self.virtual_beta_ids(beta_ids))

    def virtual_beta_id(self, beta_id):
        if self.betas is None:
            return False
        else:
            return self.betas[beta_id] is None

    def virtual_beta_ids(self, beta_ids):
        return [self.virtual_beta_id(beta_id) for beta_id in beta_ids]

    def betas_same(self, beta_ids):
        vb_ids = self.virtual_beta_ids(beta_ids)
        if any(vb_ids):
            return all(vb_ids)
        else:
            return self.betas[beta_ids[0]] == self.betas[beta_ids[1]]

    def tot_pot(self, tp, replica_id, init_bias=0.0):
        """
        Total potential including minimized function, constraining and biasing potentials.
        """
        # 2023:01:12 With the current forms of eval_min_func and biasing_potential, function can be optimized
        # by finding self.histogram.index(tp) only once here. Not doing that to preserve generality of current tot_pot.
        tot_pot = init_bias
        if self.min_function is not None:
            min_func_val = self.eval_min_func(tp, replica_id)
            if min_func_val is None:
                return None
            if self.betas[replica_id] is not None:
                min_func_val *= self.betas[replica_id]
            tot_pot += min_func_val
        tot_pot += self.biasing_potential(tp, replica_id)
        if self.bound_enforcing_coeff is not None:
            tot_pot += self.bound_enforcing_pot(tp, replica_id)
        return tot_pot

    def bound_enforcing_pot(self, tp, replica_id):
        return self.bound_enforcing_coeff * self.bound_outlie(tp.egc, replica_id)

    def bound_outlie(self, egc, replica_id):
        output = 0
        if self.hydrogen_nums is not None:
            if egc.chemgraph.tot_nhydrogens() != self.hydrogen_nums[replica_id]:
                output += abs(egc.chemgraph.tot_nhydrogens() - self.hydrogen_nums[replica_id])
        if egc.chemgraph.num_connected != 1:
            output += egc.chemgraph.num_connected() - 1
        if "final_nhatoms_range" in self.used_randomized_change_params:
            final_nhatoms_range = self.used_randomized_change_params["final_nhatoms_range"]
            if egc.num_heavy_atoms() > final_nhatoms_range[1]:
                output += egc.num_heavy_atoms() - final_nhatoms_range[1]
            else:
                if egc.num_heavy_atoms() < final_nhatoms_range[0]:
                    output += final_nhatoms_range[0] - egc.num_heavy_atoms()
        return output

    def min_over_virtual(self, tot_pot_vals, replica_ids):
        output = None
        for tot_pot_val, replica_id in zip(tot_pot_vals, replica_ids):
            if (
                self.virtual_beta_id(replica_id)
                and (output is not None)
                and (output > tot_pot_val)
            ):
                output = tot_pot_val
        return output

    def tp_pair_order_prob(self, replica_ids, tp_pair=None):
        """
        Probability that replicas are occupied either by the respective cur_tps members of tp_pair, as opposed to the positions being swapped.
        """
        if tp_pair is None:
            tp_pair = [self.cur_tps[replica_id] for replica_id in replica_ids]
        cur_tot_pot_vals = [
            self.tot_pot(tp, replica_id) for tp, replica_id in zip(tp_pair, replica_ids)
        ]
        if None in cur_tot_pot_vals:
            return None
        switched_tot_pot_vals = [
            self.tot_pot(tp, replica_id) for tp, replica_id in zip(tp_pair, replica_ids[::-1])
        ]  # The reason we don't just switch cur_tot_pot_vals is to account for potential change of biasing potential between replicas.
        if self.virtual_beta_present(replica_ids):
            if all(self.virtual_beta_ids(replica_ids)):
                return 0.5
            else:
                cur_virt_min = self.min_over_virtual(cur_tot_pot_vals, replica_ids)
                switched_virt_min = self.min_over_virtual(switched_tot_pot_vals, replica_ids)
                if cur_virt_min == switched_virt_min:
                    return 0.5
                if cur_virt_min < switched_virt_min:
                    return 1.0
                else:
                    return 0.0
        else:
            delta_pot = sum(cur_tot_pot_vals) - sum(switched_tot_pot_vals)
            return expit(-delta_pot)

    # Basic move procedures.
    def MC_step(self, replica_id=0, **dummy_kwargs):
        self.num_attempted_simple_moves += 1

        if self.greedy_delete_checked_paths:
            delete_chosen_mod_path = self.virtual_beta_id(replica_id)
        else:
            delete_chosen_mod_path = False
        changed_tp = self.cur_tps[replica_id]
        new_tp, prob_balance = randomized_change(
            changed_tp,
            visited_tp_list=self.histogram,
            delete_chosen_mod_path=delete_chosen_mod_path,
            **self.used_randomized_change_params
        )
        if new_tp is None:
            return False

        self.num_valid_simple_moves += 1

        new_tp = self.hist_checked_tp(new_tp)
        accepted = self.accept_reject_move([new_tp], prob_balance, replica_ids=[replica_id])
        if accepted:
            self.num_accepted_simple_moves += 1

        return accepted

    def trial_crossover_MC_step(self, replica_ids):
        """
        Trial move part of the crossover step.
        """
        old_cg_pair = [self.cur_tps[replica_id].egc.chemgraph for replica_id in replica_ids]
        new_cg_pair, prob_balance = randomized_crossover(
            old_cg_pair, visited_tp_list=self.histogram, **self.used_randomized_change_params
        )
        if new_cg_pair is None:
            return None, None

        #        new_pair_tps=[TrajectoryPoint(cg=new_cg) for new_cg in new_cg_pair]
        #        self.hist_check_tps(new_pair_tps)
        new_pair_tps = self.hist_checked_tps(
            [TrajectoryPoint(cg=new_cg) for new_cg in new_cg_pair]
        )
        if self.betas is not None:
            new_pair_shuffle_prob = self.tp_pair_order_prob(replica_ids, tp_pair=new_pair_tps)
            if new_pair_shuffle_prob is None:  # a minimized function value is invalid
                return None, None
            if random.random() > new_pair_shuffle_prob:  # shuffle
                new_pair_shuffle_prob = 1.0 - new_pair_shuffle_prob
                new_pair_tps = new_pair_tps[::-1]
            old_pair_shuffle_prob = self.tp_pair_order_prob(replica_ids)
            prob_balance += np.log(new_pair_shuffle_prob / old_pair_shuffle_prob)
        return new_pair_tps, prob_balance

    def crossover_MC_step(self, replica_ids):
        """
        Attempt a cross-coupled MC step.
        """
        self.num_attempted_crossovers += 1

        new_pair_tps, prob_balance = self.trial_crossover_MC_step(replica_ids)

        if new_pair_tps is None:
            return False

        self.num_valid_crossovers += 1

        accepted = self.accept_reject_move(new_pair_tps, prob_balance, replica_ids=replica_ids)
        if accepted:
            self.num_accepted_crossovers += 1
        return accepted

    # Procedures for changing entire list of Trajectory Points at once.
    def MC_step_all(self, **mc_step_kwargs):
        output = []
        for replica_id in range(self.num_replicas):
            output.append(self.MC_step(**mc_step_kwargs, replica_id=replica_id))
        return output

    def random_changed_replica_pair(self):
        return random.sample(range(self.num_replicas), 2)

    def crossover_MC_step_all(
        self, num_crossover_attempts=1, randomized_change_params=None, **dummy_kwargs
    ):
        self.init_randomized_change_params(randomized_change_params=randomized_change_params)
        for _ in range(num_crossover_attempts):
            changed_replica_ids = self.random_changed_replica_pair()
            # The swap before is to avoid situations where the pair's
            # initial ordering's probability is 0 in crossover move,
            # the second is for detailed balance concerns.
            self.parallel_tempering_swap(changed_replica_ids)
            self.crossover_MC_step(changed_replica_ids)
            self.parallel_tempering_swap(changed_replica_ids)

    def parallel_tempering_swap(self, replica_ids):
        self.num_attempted_tempering_swaps += 1

        trial_tps = [self.cur_tps[replica_ids[1]], self.cur_tps[replica_ids[0]]]
        accepted = self.accept_reject_move(trial_tps, 0.0, replica_ids=replica_ids, swap=True)
        if accepted:
            self.num_accepted_tempering_swaps += 1

        return accepted

    def parallel_tempering(self, num_parallel_tempering_attempts=1, **dummy_kwargs):
        if (self.min_function is not None) and (not self.all_betas_same):
            for _ in range(num_parallel_tempering_attempts):
                invalid_choice = True
                while invalid_choice:
                    replica_ids = self.random_changed_replica_pair()
                    invalid_choice = self.betas_same(replica_ids)
                _ = self.parallel_tempering_swap(replica_ids)

    def global_change_dict(self):
        return {
            "simple": self.MC_step_all,
            "crossover": self.crossover_MC_step_all,
            "tempering": self.parallel_tempering,
        }

    def global_random_change(
        self, prob_dict={"simple": 0.5, "crossover": 0.25, "tempering": 0.25}, **other_kwargs
    ):
        self.global_MC_step_counter += 1

        cur_procedure = random.choices(list(prob_dict), weights=list(prob_dict.values()))[0]
        global_change_dict = self.global_change_dict()
        if cur_procedure in global_change_dict:
            global_change_dict[cur_procedure](**other_kwargs)
        else:
            raise Exception("Unknown option picked.")
        if self.keep_histogram and self.track_histogram_size:
            print(
                "HIST SIZE:",
                self.global_MC_step_counter,
                cur_procedure,
                len(self.histogram),
            )
        self.update_global_histogram()
        if self.make_restart_frequency is not None:
            self.check_make_restart()
        if self.soft_exit_check_frequency is not None:
            self.check_soft_exit()
        if self.max_histogram_size is not None:
            if len(self.histogram) > self.max_histogram_size:
                self.dump_extra_histogram()
        return cur_procedure

    def complete_simulation(self, num_global_MC_steps=0, **other_kwargs):
        """
        Call global_random_change until the number of global MC steps is reached or exceeded.
        (Introduced to make restarting scripts easier.)
        """
        while self.global_MC_step_counter < num_global_MC_steps:
            self.global_random_change(**other_kwargs)

    def eval_min_func(self, tp, replica_id):
        """
        Either evaluate minimized function or look it up.
        """
        if self.no_min_function_lookup:
            output = self.min_function(tp)
        else:
            output = tp.calc_or_lookup(self.min_function_dict)[self.min_function_name]

        # If we are keeping track of the histogram make sure all calculated data is saved there.
        # TODO combine things with update_histogram?
        if self.keep_histogram:
            tp_in_histogram = tp in self.histogram
            if tp_in_histogram:
                tp_index = self.histogram.index(tp)
                tp.copy_extra_data_to(self.histogram[tp_index], omit_data=self.delete_temp_data)
                # TODO make an update call info procedure?
                self.histogram[
                    tp_index
                ].last_tot_pot_call_global_MC_step = self.global_MC_step_counter
            else:
                self.add_to_histogram(tp, replica_id)

        if self.delete_temp_data is not None:
            for dtd_identifier in self.delete_temp_data:
                if dtd_identifier in tp.calculated_data:
                    del tp.calculated_data[dtd_identifier]

        self.update_saved_candidates(tp)

        return output

    def merge_histogram(self, other_histogram):
        """
        Merge current histogram with another one (for example recovered from a histogram dump file).
        """
        for tp in other_histogram:
            if tp in self.histogram:
                tp_index = self.histogram.index(tp)
                self.histogram[tp_index].merge_visit_data(tp)
            else:
                self.histogram.add(tp)

    def biasing_potential(self, tp, replica_id):
        cur_beta_virtual = self.virtual_beta_id(replica_id)
        if cur_beta_virtual:
            used_bias_coeff = self.vbeta_bias_coeff
        else:
            used_bias_coeff = self.bias_coeff

        if used_bias_coeff is None:
            return 0.0
        if (self.histogram is None) or (tp not in self.histogram):
            return 0.0
        tp_index = self.histogram.index(tp)
        tp_in_hist = self.histogram[tp_index]
        if tp_in_hist.num_visits is None:
            return 0.0
        tp_in_hist.last_tot_pot_call_global_MC_step = (
            self.global_MC_step_counter
        )  # TODO is it needed?
        if self.bias_pot_all_replicas:
            cur_visit_num = 0
            for other_replica_id in range(self.num_replicas):
                if cur_beta_virtual == self.virtual_beta_id(other_replica_id):
                    cur_visit_num += tp_in_hist.num_visits[other_replica_id]
            cur_visit_num = float(cur_visit_num)
        else:
            cur_visit_num = float(tp_in_hist.num_visits[replica_id])
        return cur_visit_num * used_bias_coeff

    def add_to_histogram(self, trajectory_point_in, replica_id):
        trajectory_point_in.first_MC_step_encounter = self.MC_step_counter
        trajectory_point_in.first_global_MC_step_encounter = self.global_MC_step_counter
        trajectory_point_in.first_encounter_replica = replica_id
        trajectory_point_in.last_tot_pot_call_global_MC_step = self.global_MC_step_counter

        added_tp = deepcopy(trajectory_point_in)

        if self.delete_temp_data is not None:
            for del_entry in self.delete_temp_data:
                if del_entry in added_tp.calculated_data:
                    del added_tp.calculated_data[del_entry]

        self.histogram.add(added_tp)

    def update_histogram(self, replica_ids):
        if self.keep_histogram:
            for replica_id in replica_ids:
                cur_tp = self.cur_tps[replica_id]
                tp_in_hist = cur_tp in self.histogram
                if not tp_in_hist:
                    self.add_to_histogram(cur_tp, replica_id)
                cur_tp_index = self.histogram.index(cur_tp)
                if tp_in_hist:
                    cur_tp.copy_extra_data_to(
                        self.histogram[cur_tp_index],
                        linear_storage=self.linear_storage,
                        omit_data=self.delete_temp_data,
                    )
                else:
                    if self.linear_storage:
                        self.histogram[cur_tp_index].clear_possibility_info()
                        self.histogram[cur_tp_index].egc.chemgraph.pair_equivalence_matrix = None

                if self.histogram[cur_tp_index].first_MC_step_acceptance is None:
                    self.histogram[cur_tp_index].first_MC_step_acceptance = self.MC_step_counter
                    self.histogram[
                        cur_tp_index
                    ].first_global_MC_step_acceptance = self.global_MC_step_counter
                    self.histogram[cur_tp_index].first_acceptance_replica = replica_id

                if self.keep_full_trajectory and self.keep_detailed_global_steps:
                    self.histogram[cur_tp_index].add_visit_step_id(
                        self.MC_step_counter,
                        replica_id,
                        step_type=nonglobal_step_traj_storage_label,
                    )
                if self.visit_num_count_acceptance:
                    self.update_num_visits(cur_tp_index, replica_id)

    def update_global_histogram(self):
        if self.keep_histogram:
            for replica_id, cur_tp in enumerate(self.cur_tps):
                cur_tp_index = self.histogram.index(cur_tp)
                if self.keep_full_trajectory:
                    self.histogram[cur_tp_index].add_visit_step_id(
                        self.global_MC_step_counter,
                        replica_id,
                        step_type=global_step_traj_storage_label,
                    )
                if not self.visit_num_count_acceptance:
                    self.update_num_visits(cur_tp_index, replica_id)

    def update_num_visits(self, tp_index, replica_id):
        if self.histogram[tp_index].num_visits is None:
            self.histogram[tp_index].num_visits = np.zeros((self.num_replicas,), dtype=int)
        self.histogram[tp_index].num_visits[replica_id] += 1

    #    def hist_check_tp(self, tp):
    #        """
    #        Check whether there is data in histogram related to a TrajectoryPoint.
    #        """
    #        if self.histogram is None:
    #            return
    #        if tp in self.histogram:
    #            tp_in_hist=self.histogram[self.histogram.index(tp)]
    #            tp_in_hist.copy_extra_data_to(tp)

    #    def hist_check_tps(self, tp_list):
    #        """
    #        Check whether there is data in histogram related to TrajectoryPoint objects.
    #        """
    #        if self.histogram is None:
    #            return
    #        for tp in tp_list:
    #            self.hist_check_tp(tp)
    # TODO The variant of the code that used hist_check_tps instead of hist_checked_tps seemed slightly slower for some reason,
    # might change should the code face a major revision. CHECK VISIT NUMBERS ARE INCLUDED IN COPY_EXTRA_DATA_TO!
    def hist_checked_tp(self, tp):
        if (self.histogram is None) or (tp not in self.histogram):
            return tp
        return deepcopy(self.histogram[self.histogram.index(tp)])

    def hist_checked_tps(self, tp_list):
        """
        Return a version of tp_list where all entries are replaced with references to self.histogram.
        tp_list : list of TrajectoryPoint objects
        """
        if self.histogram is None:
            return tp_list
        return [self.hist_checked_tp(tp) for tp in tp_list]

    def clear_histogram_visit_data(self):
        for tp in self.histogram:
            tp.num_visits = None
            if self.keep_full_trajectory:
                tp.visit_step_ids = None

    def update_saved_candidates(self, tp_in):
        maintain_sorted_CandidateCompound_list(
            self, tp_in, tp_in.calculated_data[self.min_function_name]
        )

    # Some properties for more convenient trajectory analysis.
    def ordered_trajectory(self):
        return ordered_trajectory(
            self.histogram,
            global_MC_step_counter=self.global_MC_step_counter,
            num_replicas=self.num_replicas,
        )

    def ordered_trajectory_ids(self):
        assert self.keep_full_trajectory
        return ordered_trajectory_ids(
            self.histogram,
            global_MC_step_counter=self.global_MC_step_counter,
            num_replicas=self.num_replicas,
        )

    # Quality-of-life-related.
    def frequency_checker(self, identifier, frequency):
        if identifier not in self.global_steps_since_last:
            self.global_steps_since_last[identifier] = 1
        output = self.global_steps_since_last[identifier] == frequency
        if output:
            self.global_steps_since_last[identifier] = 1
        else:
            self.global_steps_since_last[identifier] += 1
        return output

    def check_make_restart(self):
        if self.frequency_checker("make_restart", self.make_restart_frequency):
            self.make_restart()

    def check_soft_exit(self):
        if self.frequency_checker("soft_exit", self.soft_exit_check_frequency):
            if os.path.isfile("EXIT"):
                self.make_restart()
                raise SoftExitCalled

    def make_restart(self, restart_file: str or None = None, tarball: bool or None = None):
        """
        Create a file containing all information needed to restart the simulation from the current point.
        restart_file : name of the file where the dump is created; if None self.restart_file is used
        """
        if restart_file is None:
            restart_file = self.restart_file
        saved_data = {
            "cur_tps": self.cur_tps,
            "MC_step_counter": self.MC_step_counter,
            "global_MC_step_counter": self.global_MC_step_counter,
            "num_attempted_crossovers": self.num_attempted_crossovers,
            "num_valid_crossovers": self.num_valid_crossovers,
            "num_accepted_crossovers": self.num_accepted_crossovers,
            "num_attempted_simple_moves": self.num_attempted_simple_moves,
            "num_valid_simple_moves": self.num_valid_simple_moves,
            "num_accepted_simple_moves": self.num_accepted_simple_moves,
            "num_attempted_tempering_swaps": self.num_attempted_tempering_swaps,
            "num_accepted_tempering_swaps": self.num_accepted_tempering_swaps,
            "moves_since_changed": self.moves_since_changed,
            "global_steps_since_last": self.global_steps_since_last,
            "numpy_rng_state": np.random.get_state(),
            "random_rng_state": random.getstate(),
            "min_function_name": self.min_function_name,
            "min_function": self.min_function,
            "betas": self.betas,
        }
        if self.keep_histogram:
            saved_data = {**saved_data, "histogram": self.histogram}
        if self.num_saved_candidates is not None:
            saved_data = {**saved_data, "saved_candidates": self.saved_candidates}
        if tarball is None:
            tarball = self.compress_restart
        dump2pkl(saved_data, restart_file, compress=tarball)

    def restart_from(self, restart_file: Union[str, None] = None):
        """
        Recover all data from
        restart_file : name of the file from which the data is recovered; if None self.restart_file is used
        """
        if restart_file is None:
            restart_file = self.restart_file
        recovered_data = loadpkl(restart_file, compress=self.compress_restart)
        self.cur_tps = recovered_data["cur_tps"]
        self.MC_step_counter = recovered_data["MC_step_counter"]
        self.global_MC_step_counter = recovered_data["global_MC_step_counter"]
        self.num_attempted_crossovers = recovered_data["num_attempted_crossovers"]
        self.num_valid_crossovers = recovered_data["num_valid_crossovers"]
        self.num_accepted_crossovers = recovered_data["num_accepted_crossovers"]

        self.num_attempted_simple_moves = recovered_data["num_attempted_simple_moves"]
        self.num_valid_simple_moves = recovered_data["num_valid_simple_moves"]
        self.num_accepted_simple_moves = recovered_data["num_accepted_simple_moves"]

        self.num_attempted_tempering_swaps = recovered_data["num_attempted_tempering_swaps"]
        self.num_accepted_tempering_swaps = recovered_data["num_accepted_tempering_swaps"]

        self.moves_since_changed = recovered_data["moves_since_changed"]
        self.global_steps_since_last = recovered_data["global_steps_since_last"]
        if self.keep_histogram:
            self.histogram = recovered_data["histogram"]
        if self.num_saved_candidates is not None:
            self.saved_candidates = recovered_data["saved_candidates"]
        self.min_function = recovered_data["min_function"]
        np.random.set_state(recovered_data["numpy_rng_state"])
        random.setstate(recovered_data["random_rng_state"])

    def dump_extra_histogram(self):
        """
        Dump a histogram to a dump file with a yet unoccupied name.
        """
        last_tot_pot_call_cut = self.histogram_last_tot_pot_call_cut()
        dumped_tps = SortedList()
        tp_id = 0
        while tp_id != len(self.histogram):
            if self.histogram[tp_id].last_tot_pot_call_global_MC_step < last_tot_pot_call_cut:
                dumped_tps.add(deepcopy(self.histogram[tp_id]))
                del self.histogram[tp_id]
            else:
                tp_id += 1

        dump2pkl(dumped_tps, self.histogram_file_dump(), compress=self.compress_restart)

    def histogram_last_tot_pot_call_cut(self):
        """
        At which last_tot_pot_call_global_MC_step cut the histogram in order to dump only self.histogram_dump_portion of it
        """
        last_call_hist = SortedDict()
        for tp in self.histogram:
            cur_last_call = tp.last_tot_pot_call_global_MC_step
            if cur_last_call not in last_call_hist:
                last_call_hist[cur_last_call] = 0
            last_call_hist[cur_last_call] += 1
        # Decide where to cut
        cut_num = 0
        target_cut_num = int(len(self.histogram) * self.histogram_dump_portion)
        for last_call, num_tps in last_call_hist.items():
            cut_num += num_tps
            if cut_num > target_cut_num:
                return last_call
        return last_call_hist.keys()[-1]

    def histogram_file_dump(self):
        """
        Returns name of the histogram dump file for a given dump_id.
        dump_id : int id of the dump
        """
        dump_id = 1
        while True:
            output = (
                self.histogram_dump_file_prefix
                + str(dump_id)
                + pkl_compress_ending[self.compress_restart]
            )
            if not os.path.isfile(output):
                return output
            dump_id += 1

    def move_statistics(self):
        """
        Return dictionnary containing information necessary to determine data such as acceptance rate, validity ratio, etc.
        """
        return {
            "num_attempted_crossovers": self.num_attempted_crossovers,
            "num_valid_crossovers": self.num_valid_crossovers,
            "num_accepted_crossovers": self.num_accepted_crossovers,
            "num_attempted_simple_moves": self.num_attempted_simple_moves,
            "num_valid_simple_moves": self.num_valid_simple_moves,
            "num_accepted_simple_moves": self.num_accepted_simple_moves,
            "num_attempted_tempering_swaps": self.num_attempted_tempering_swaps,
            "num_accepted_tempering_swaps": self.num_accepted_tempering_swaps,
        }
