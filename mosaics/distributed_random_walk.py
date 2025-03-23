# Implementation of distributed version of MOSAiCS.
# Inspiration: https://doi.org/10.1016/j.csda.2008.10.025
# TODO: For now only implemented loky parallelization which is confined to one machine. If someone needs many-machine parallelization
# it should be possible to implement mpi4py-based object treating several RandomWalkEnsemble objects the same way RandomWalkEnsemble treats RandomWalk objects.
import os
import random
from copy import deepcopy
from itertools import repeat
from subprocess import run as terminal_run

import numpy as np
from loky import get_reusable_executor
from loky.process_executor import TerminatedWorkerError
from sortedcontainers import SortedList

from .chem_graph import misc_global_variables_current_kwargs, set_misc_global_variables
from .random_walk import (
    CandidateCompound,
    Metropolis_acceptance_probability,
    RandomWalk,
    TrajectoryPoint,
    default_minfunc_name,
    maintain_sorted_CandidateCompound_list,
)


class SubpopulationPropagationIntermediateResults:
    def __init__(
        self,
        rw,
        num_exploration_replicas=None,
        num_greedy_replicas=None,
        cloned_betas=True,
        num_beta_subpopulation_clones=2,
        track_worst_accepted_candidates=False,
        num_minfunc_saved_steps=None,
    ):
        """
        Update/store intermediate data.
        Being expanded to include everything required in .optimization_protocol.
        """
        self.nsteps = 0
        self.num_replicas = rw.num_replicas
        self.num_exploration_replicas = num_exploration_replicas
        self.num_greedy_replicas = num_greedy_replicas
        self.cloned_betas = cloned_betas
        if self.cloned_betas:
            self.num_beta_subpopulation_clones = num_beta_subpopulation_clones
        else:
            self.num_beta_subpopulation_clones = 1
        if self.num_exploration_replicas is not None:
            self.sum_tempering_neighbors_acceptance_probability = np.zeros(
                (self.num_exploration_replicas - 1,)
            )

        self.track_worst_accepted_candidates = track_worst_accepted_candidates
        self.worst_accepted_candidates = None

        self.num_minfunc_saved_steps = num_minfunc_saved_steps
        if self.num_minfunc_saved_steps is None:
            self.minfunc_val_log = None
        else:
            self.minfunc_val_log = np.empty((self.num_minfunc_saved_steps, len(rw.cur_tps)))

    def update(self, rw):
        """
        Update information stored in the object.
        """
        self.nsteps += 1
        if self.minfunc_val_log is not None:
            self.update_minfunc_val_log(rw)
        if self.track_worst_accepted_candidates:
            self.update_worst_accepted_candidates(rw.cur_tps, rw.min_function_name)
        if self.num_exploration_replicas is not None:
            self.update_sum_tempering_neighbors_acceptance_probability(rw)

    def update_minfunc_val_log(self, rw):
        for rep_id, cur_tp in enumerate(rw.cur_tps):
            cur_minfunc = cur_tp.calculated_data[rw.min_function_name]
            self.minfunc_val_log[self.nsteps - 1, rep_id] = cur_minfunc

    def update_sum_tempering_neighbors_acceptance_probability(self, rw):
        for exploration_replica_id in range(self.num_exploration_replicas - 1):
            beta_interval_start = (
                self.num_greedy_replicas + exploration_replica_id
            ) * self.num_beta_subpopulation_clones
            for cloned_id in range(self.num_beta_subpopulation_clones):
                beta_id1 = beta_interval_start + cloned_id
                beta_id2 = beta_id1 + self.num_beta_subpopulation_clones
                beta1 = rw.betas[beta_id1]
                beta2 = rw.betas[beta_id2]
                minfunc_1 = rw.cur_tps[beta_id1].calculated_data[rw.min_function_name]
                minfunc_2 = rw.cur_tps[beta_id2].calculated_data[rw.min_function_name]
                # acceptance probability of swapping the two trajectory points
                acc_prob = Metropolis_acceptance_probability(
                    (beta1 - beta2) * (minfunc_1 - minfunc_2)
                )
                self.sum_tempering_neighbors_acceptance_probability[exploration_replica_id] += (
                    acc_prob / self.num_beta_subpopulation_clones
                )

    def update_worst_accepted_candidates(self, current_trajectory_points, minimized_function_name):
        if self.worst_accepted_candidates is None:
            self.worst_accepted_candidates = [
                CandidateCompound(
                    deepcopy(cur_tp), cur_tp.calculated_data[minimized_function_name]
                )
                for cur_tp in current_trajectory_points
            ]
            return
        for rep_id, (cur_tp, worst_candidate) in enumerate(
            zip(current_trajectory_points, self.worst_accepted_candidates)
        ):
            if cur_tp == worst_candidate.tp:
                continue
            minfunc_val = cur_tp.calculated_data[minimized_function_name]
            if minfunc_val < worst_candidate.func_val:
                continue
            self.worst_accepted_candidates[rep_id] = CandidateCompound(
                deepcopy(cur_tp), minfunc_val
            )


def gen_subpopulation_propagation_result(
    init_tps,
    betas,
    random_rng_state,
    numpy_rng_state,
    num_global_steps=1,
    misc_random_walk_kwargs={},
    global_step_kwargs={},
    synchronization_signal_file=None,
    synchronization_check_frequency=None,
    extra_intermediate_results_kwargs={},
    misc_global_variables_needed_kwargs={},
    num_extra_rng_calls=0,
):
    set_misc_global_variables(**misc_global_variables_needed_kwargs)
    # Create the random walk for propagation.
    rw = RandomWalk(init_tps=init_tps, betas=betas, **misc_random_walk_kwargs)
    # Initialize the two random number generators.
    if random_rng_state is not None:
        if isinstance(random_rng_state, int):
            random.seed(random_rng_state)
            np.random.seed(numpy_rng_state)
        else:
            random.setstate(random_rng_state)
            np.random.set_state(numpy_rng_state)
    # Do some extra RNG calls if we need to prevent the trajectory from re-encountering a problematic chemical graph.
    if num_extra_rng_calls != 0:
        for _ in range(num_extra_rng_calls):
            _ = random.random()
            _ = np.random.random()

    intermediate_results = SubpopulationPropagationIntermediateResults(
        rw, **extra_intermediate_results_kwargs
    )

    check_other_finished = (synchronization_signal_file is not None) and (
        synchronization_check_frequency is not None
    )
    # Propagate.
    for step_id in range(num_global_steps):
        rw.global_random_change(**global_step_kwargs)
        intermediate_results.update(rw)
        if check_other_finished:
            if step_id % synchronization_check_frequency == 0:
                if os.path.isfile(synchronization_signal_file):
                    break
    if check_other_finished:
        terminal_run(["touch", synchronization_signal_file])
    return rw, intermediate_results, random.getstate(), np.random.get_state()


class DistributedRandomWalk:
    def __init__(
        self,
        num_processes=1,
        num_subpopulations=1,
        num_internal_global_steps=1,
        betas=None,
        init_egc=None,
        init_egcs=None,
        init_tps=None,
        min_function=None,
        min_function_name=default_minfunc_name,
        randomized_change_params={},
        greedy_delete_checked_paths=False,
        global_step_params={},
        save_logs=False,
        saved_candidates_max_difference=None,
        num_saved_candidates=None,
        previous_saved_candidates=None,
        synchronization_signal_file=None,
        synchronization_check_frequency=None,
        subpopulation_propagation_seed=None,
        subpopulation_random_rng_states=None,
        subpopulation_numpy_rng_states=None,
        cloned_betas=True,
        num_beta_subpopulation_clones=2,
        extra_intermediate_results_kwargs={},
        terminated_worker_max_restart_number=0,
        terminated_worker_num_extra_rng_calls=1,
        debug=False,
    ):
        # Subpopulation parameters
        self.num_subpopulations = num_subpopulations
        self.cloned_betas = cloned_betas
        self.num_beta_subpopulation_clones = num_beta_subpopulation_clones
        # Simulation parameters.
        self.init_betas(betas)
        self.num_internal_global_steps = num_internal_global_steps
        self.min_function = min_function
        self.min_function_name = min_function_name
        self.greedy_delete_checked_paths = greedy_delete_checked_paths
        self.random_walk_kwargs = {
            "randomized_change_params": deepcopy(randomized_change_params),
            "debug": debug,
            "min_function": self.min_function,
            "min_function_name": self.min_function_name,
            "greedy_delete_checked_paths": self.greedy_delete_checked_paths,
        }
        self.global_step_params = global_step_params
        self.subpopulation_indices_list = None
        self.num_internal_global_steps = num_internal_global_steps
        # Related to synchronization between different walkers.
        self.synchronization_signal_file = synchronization_signal_file
        self.synchronization_check_frequency = synchronization_check_frequency
        # Random number generator states used inside subpopulations.
        self.subpopulation_propagation_seed = subpopulation_propagation_seed
        self.subpopulation_random_rng_states = subpopulation_random_rng_states
        self.subpopulation_numpy_rng_states = subpopulation_numpy_rng_states
        if self.subpopulation_random_rng_states is None:
            self.subpopulation_random_rng_states = self.default_init_rng_states()
        if self.subpopulation_numpy_rng_states is None:
            self.subpopulation_numpy_rng_states = self.default_init_rng_states()
        # Parallelization.
        self.num_processes = num_processes
        self.executor_needs_reset = True
        self.executor = None
        # Logs of relevant information.
        self.save_logs = save_logs
        if self.save_logs:
            self.propagation_logs = []
        else:
            self.propagation_logs = None

        # For storing most important temporary data.
        # Saving best candidate molecules.
        self.saved_candidates = previous_saved_candidates
        if self.saved_candidates is None:
            self.saved_candidates = SortedList()
        self.num_saved_candidates = num_saved_candidates
        self.saved_candidates_max_difference = saved_candidates_max_difference
        # Related to distribution of the optimized quantity.
        self.extra_intermediate_results_kwargs = extra_intermediate_results_kwargs
        self.init_saved_temp_data()

        # Correct random walk keyword arguments.
        self.random_walk_kwargs["num_saved_candidates"] = self.num_saved_candidates
        self.random_walk_kwargs[
            "saved_candidates_max_difference"
        ] = self.saved_candidates_max_difference
        # Make sure betas and initial conditions are not accidentally defined twice for the random walk.
        for del_key in ["betas", "init_tps", "init_egcs"]:
            if del_key in self.random_walk_kwargs:
                del self.random_walk_kwargs[del_key]
        # For storing statistics on move success.
        self.num_attempted_crossovers = 0
        self.num_valid_crossovers = 0
        self.num_accepted_crossovers = 0

        self.num_attempted_simple_moves = 0
        self.num_valid_simple_moves = 0
        self.num_accepted_simple_moves = 0

        self.num_attempted_tempering_swaps = 0
        self.num_accepted_tempering_swaps = 0

        # Initial conditions.
        self.initialize_current_trajectory_points(
            init_tps=init_tps, init_egcs=init_egcs, init_egc=init_egc
        )

        self.terminated_worker_max_restart_number = terminated_worker_max_restart_number
        self.terminated_worker_num_extra_rng_calls = terminated_worker_num_extra_rng_calls
        self.subpopulation_num_extra_rng_calls = np.zeros((self.num_subpopulations,), dtype=int)
        self.subpopulation_num_attempted_restarts = np.zeros((self.num_subpopulations,), dtype=int)
        self.subpopulation_propagation_completed = np.zeros((self.num_subpopulations,), dtype=int)
        self.subpopulation_propagation_results_list = [
            None for _ in range(self.num_subpopulations)
        ]

    def init_saved_temp_data(self):
        self.worst_accepted_candidates = None
        self.minfunc_val_log = None
        num_expl_repl_label = "num_exploration_replicas"
        self.track_av_tempering_neighbors_acceptance_probability = (
            num_expl_repl_label in self.extra_intermediate_results_kwargs
        )
        if self.track_av_tempering_neighbors_acceptance_probability:
            self.true_num_exploration_replicas = self.extra_intermediate_results_kwargs[
                num_expl_repl_label
            ]
            assert self.true_num_exploration_replicas is not None
            self.av_tempering_neighbors_acceptance_probability = np.empty(
                (self.true_num_exploration_replicas - 1,)
            )

    def initialized_candidate(self, init_tp):
        minfunc_eval_dict = {self.min_function_name: self.min_function}
        minfunc_val = init_tp.calc_or_lookup(minfunc_eval_dict)[self.min_function_name]
        return CandidateCompound(init_tp, minfunc_val)

    def initialize_current_trajectory_points(self, init_tps=None, init_egcs=None, init_egc=None):
        if init_tps is None:
            if init_egcs is None:
                assert init_egc is not None
                new_tp = TrajectoryPoint(egc=init_egc)
                self.update_saved_candidates(
                    self.initialized_candidate(new_tp)
                )  # to make sure that optimized function is evaluated just once
                init_tps = [deepcopy(new_tp) for _ in self.betas]
            else:
                init_tps = [TrajectoryPoint(egc=init_egc) for init_egc in init_egcs]
        for init_tp in init_tps:
            self.update_saved_candidates(self.initialized_candidate(init_tp))
        self.current_trajectory_points = init_tps

    def init_cloned_betas(self, beta_input):
        self.tot_num_clones = self.num_subpopulations * self.num_beta_subpopulation_clones
        self.original_betas = np.array(beta_input)
        betas = []
        for beta in beta_input:
            betas += [beta for _ in range(self.tot_num_clones)]
        self.betas = np.array(betas)

    def init_betas(self, beta_input):
        if self.cloned_betas:
            self.init_cloned_betas(beta_input)
        else:
            self.betas = beta_input
        self.num_replicas = len(self.betas)

    def true_beta_id2beta_ids(self, true_beta_id):
        if self.cloned_betas:
            return np.array(
                range(
                    self.tot_num_clones * true_beta_id,
                    self.tot_num_clones * (true_beta_id + 1),
                )
            )
        else:
            return np.array([true_beta_id])

    def true_beta_val_ids2beta_ids(self, true_beta_val, true_beta_id):
        all_true_beta_ids = self.complete_with_equal_betas(true_beta_val, true_beta_id)
        output = self.true_beta_id2beta_ids(all_true_beta_ids[0])
        for other_beta_id in all_true_beta_ids[1:]:
            output = np.append(output, self.true_beta_id2beta_ids(other_beta_id))
        return output

    def true_betas(self):
        if self.cloned_betas:
            return self.original_betas
        else:
            return self.betas

    def complete_with_equal_betas(self, beta_val, beta_id):
        output = [beta_id]
        for other_beta_id, other_beta in enumerate(self.true_betas()):
            if (other_beta_id != beta_id) and (other_beta == beta_val):
                output.append(other_beta_id)
        return output

    def largest_beta_ids(self):
        max_beta = None
        max_beta_id = None
        for beta_id, beta in enumerate(self.true_betas()):
            if beta is None:
                continue
            if (max_beta is None) or (max_beta < beta):
                max_beta = beta
                max_beta_id = beta_id

        return self.true_beta_val_ids2beta_ids(max_beta, max_beta_id)

    def smallest_beta_ids(self):
        min_beta = None
        min_beta_id = None
        for beta_id, beta in enumerate(self.true_betas()):
            if beta is None:
                continue
            if (min_beta is None) or (min_beta > beta):
                min_beta = beta
                min_beta_id = beta_id
        return self.true_beta_val_ids2beta_ids(min_beta, min_beta_id)

    def add_to_move_statistics(self, random_walk_instance):
        self.num_attempted_crossovers += random_walk_instance.num_attempted_crossovers
        self.num_valid_crossovers += random_walk_instance.num_valid_crossovers
        self.num_accepted_crossovers += random_walk_instance.num_accepted_crossovers

        self.num_attempted_simple_moves += random_walk_instance.num_attempted_simple_moves
        self.num_valid_simple_moves += random_walk_instance.num_valid_simple_moves
        self.num_accepted_simple_moves += random_walk_instance.num_accepted_simple_moves

        self.num_attempted_tempering_swaps += random_walk_instance.num_attempted_tempering_swaps
        self.num_accepted_tempering_swaps += random_walk_instance.num_accepted_tempering_swaps

    def default_init_rng_states(self):
        if self.subpopulation_propagation_seed is None:
            return list(repeat(None, self.num_subpopulations))
        else:
            return [
                i + int(self.subpopulation_propagation_seed)
                for i in range(self.num_subpopulations)
            ]

    def generate_cloned_subpopulation_indices(self):
        shuffled_indices = []
        for _ in self.original_betas:
            clone_indices = []
            for subpop_id in range(self.num_subpopulations):
                clone_indices += [subpop_id for _ in range(self.num_beta_subpopulation_clones)]
            random.shuffle(clone_indices)
            shuffled_indices += clone_indices
        self.subpopulation_indices_list = [[] for _ in range(self.num_subpopulations)]
        for replica_id, subpopulation_id in enumerate(shuffled_indices):
            self.subpopulation_indices_list[subpopulation_id].append(replica_id)
        self.subpopulation_indices_list = [
            np.array(subpopulation_indices)
            for subpopulation_indices in self.subpopulation_indices_list
        ]

    def divide_into_subpopulations(self, indices_list):
        shuffled_indices_list = deepcopy(indices_list)
        random.shuffle(shuffled_indices_list)
        subpopulation_indices = [[] for _ in range(self.num_subpopulations)]
        for shuffled_order, id in enumerate(shuffled_indices_list):
            subpopulation_indices[shuffled_order % self.num_subpopulations].append(id)
        return subpopulation_indices

    def generate_subpopulation_indices(self):
        """
        Randomly generate indices of replicas in different subpopulations.
        Both greedy and exploration replicas are divided equally among replicas.
        """
        if self.cloned_betas:
            self.generate_cloned_subpopulation_indices()
            return
        all_greedy_indices = []
        all_exploration_indices = []
        for replica_id, beta in enumerate(self.betas):
            if beta is None:
                all_greedy_indices.append(replica_id)
            else:
                all_exploration_indices.append(replica_id)
        subpopulation_greedy_indices_list = self.divide_into_subpopulations(all_greedy_indices)
        subpopulation_exploration_indices_list = self.divide_into_subpopulations(
            all_exploration_indices
        )
        self.subpopulation_indices_list = [
            np.array(greedy_indices + exploration_indices)
            for greedy_indices, exploration_indices in zip(
                subpopulation_greedy_indices_list,
                subpopulation_exploration_indices_list,
            )
        ]

    def update_saved_candidates(self, new_candidate):
        """
        Include Candidate object into saved_candidates list.
        """
        maintain_sorted_CandidateCompound_list(self, candidate=new_candidate)

    def update_temporary_data(
        self, subpopulation_indices, cur_random_walk, cur_intermediate_results
    ):
        # Update saved candidates list.
        for candidate in cur_random_walk.saved_candidates:
            self.update_saved_candidates(candidate)
        if self.track_av_tempering_neighbors_acceptance_probability:
            self.av_tempering_neighbors_acceptance_probability[:] += (
                cur_intermediate_results.sum_tempering_neighbors_acceptance_probability[:]
                / self.num_subpopulations
                / cur_intermediate_results.nsteps
            )
        if cur_intermediate_results.track_worst_accepted_candidates:
            if self.worst_accepted_candidates is None:
                self.worst_accepted_candidates = [None for _ in self.betas]
            self.update_worst_accepted_candidates(subpopulation_indices, cur_intermediate_results)
        if cur_intermediate_results.minfunc_val_log is not None:
            if self.minfunc_val_log is None:
                self.minfunc_val_log = np.empty(
                    (
                        cur_intermediate_results.minfunc_val_log.shape[0],
                        self.num_replicas,
                    )
                )
            for replica_index, replica_minfunc_logs in zip(
                subpopulation_indices, cur_intermediate_results.minfunc_val_log.T
            ):
                self.minfunc_val_log[:, replica_index] = replica_minfunc_logs[:]

    def update_worst_accepted_candidates(self, subpopulation_indices, cur_intermediate_results):
        for true_id, worst_cand in zip(
            subpopulation_indices, cur_intermediate_results.worst_accepted_candidates
        ):
            old_candidate = self.worst_accepted_candidates[true_id]
            if (old_candidate is None) or (old_candidate.func_val < worst_cand.func_val):
                self.worst_accepted_candidates[true_id] = worst_cand

    def completed_subpopulation_sublist(self, full_list):
        sublist = []
        for completed, element in zip(self.subpopulation_propagation_completed, full_list):
            if not completed:
                sublist.append(element)
        return sublist

    def num_incomplete_calculations(self):
        return sum(np.logical_not(self.subpopulation_propagation_completed))

    def subpopulation_propagation_inputs(self):
        """
        Arguments used in loky's map.
        """
        all_init_tps = []
        all_betas = []
        for subpopulation_id, subpopulation_indices in enumerate(self.subpopulation_indices_list):
            if self.subpopulation_propagation_completed[subpopulation_id]:
                continue
            # The initial trajectory points and the beta values.
            cur_init_tps = []
            cur_betas = []
            for i in subpopulation_indices:
                cur_init_tps.append(deepcopy(self.current_trajectory_points[i]))
                cur_betas.append(self.betas[i])
            all_init_tps.append(cur_init_tps)
            all_betas.append(cur_betas)

        input_list = [
            all_init_tps,
            all_betas,
            self.completed_subpopulation_sublist(self.subpopulation_random_rng_states),
            self.completed_subpopulation_sublist(self.subpopulation_numpy_rng_states),
        ]
        for other_arg in [
            self.num_internal_global_steps,
            self.random_walk_kwargs,
            self.global_step_params,
            self.synchronization_signal_file,
            self.synchronization_signal_file,
            self.extra_intermediate_results_kwargs,
            misc_global_variables_current_kwargs(),
        ]:
            input_list.append(repeat(other_arg, self.num_incomplete_calculations()))
        input_list.append(
            self.completed_subpopulation_sublist(self.subpopulation_num_extra_rng_calls)
        )
        return input_list

    def update_current_trajectory_points(self, subpopulation_indices, random_walk):
        for internal_replica_id, true_replica_id in enumerate(subpopulation_indices):
            self.current_trajectory_points[true_replica_id] = deepcopy(
                random_walk.cur_tps[internal_replica_id]
            )

    def attempt_subpopulation_propagation(
        self,
    ):
        if self.executor_needs_reset:
            self.executor = get_reusable_executor(max_workers=self.num_processes)
            self.executor_needs_reset = False

        all_propagated_subpopulation_ids = np.where(
            np.logical_not(self.subpopulation_propagation_completed)
        )[0]
        results_iterator = self.executor.map(
            gen_subpopulation_propagation_result, *self.subpopulation_propagation_inputs()
        )
        if self.synchronization_signal_file is not None:
            terminal_run(["rm", "-f", self.synchronization_signal_file])
        propagated_subpopulation_id = 0
        while True:
            if propagated_subpopulation_id == len(all_propagated_subpopulation_ids):
                break
            subpopulation_id = all_propagated_subpopulation_ids[propagated_subpopulation_id]
            try:
                cur_res = results_iterator.__next__()
                self.subpopulation_propagation_completed[subpopulation_id] = True
                self.subpopulation_propagation_results_list[subpopulation_id] = cur_res
            except StopIteration:
                break
            except TerminatedWorkerError:
                self.subpopulation_num_attempted_restarts[subpopulation_id] += 1
                if (
                    self.subpopulation_num_attempted_restarts[subpopulation_id]
                    > self.terminated_worker_max_restart_number
                ):
                    raise TerminatedWorkerError
                self.subpopulation_num_extra_rng_calls[subpopulation_id] += 1
                self.executor_needs_reset = True
            propagated_subpopulation_id += 1

    def generate_all_subpopulation_propagation_results(self):
        self.subpopulation_propagation_completed[:] = False
        self.subpopulation_num_extra_rng_calls[:] = 0
        self.subpopulation_num_attempted_restarts[:] = 0
        while False in self.subpopulation_propagation_completed:
            self.attempt_subpopulation_propagation()

    def propagate_subpopulations(self):
        if self.save_logs:
            subpopulation_propagation_latest_logs = []
        if self.track_av_tempering_neighbors_acceptance_probability:
            self.av_tempering_neighbors_acceptance_probability[:] = 0.0

        self.generate_all_subpopulation_propagation_results()

        for subpopulation_index, propagation_results in enumerate(
            self.subpopulation_propagation_results_list
        ):
            random_walk = propagation_results[0]
            intermediate_results = propagation_results[1]
            subpopulation_indices = self.subpopulation_indices_list[subpopulation_index]
            if self.save_logs:
                # Save subpopulation indices, RandomWalk object, and
                subpopulation_propagation_latest_logs.append(
                    (subpopulation_indices, random_walk, intermediate_results)
                )
            self.update_temporary_data(subpopulation_indices, random_walk, intermediate_results)
            self.add_to_move_statistics(random_walk)
            self.subpopulation_random_rng_states[subpopulation_index] = propagation_results[2]
            self.subpopulation_numpy_rng_states[subpopulation_index] = propagation_results[3]
            self.update_current_trajectory_points(subpopulation_indices, random_walk)
        if self.save_logs:
            self.propagation_logs.append(subpopulation_propagation_latest_logs)

    def propagate(self):
        self.generate_subpopulation_indices()
        self.propagate_subpopulations()
