# Algorithm for using MOSAiCS for (ideally) completely blind optimization with automatic self-adjustment of all relevant simulation parameters.
# Inspired by this paper: 10.1093/mnras/stv2422
# I tried making the algorithm as barebones as possible, hopefully making something that would break less.
# TODO: Double-check whether it is necessary to account for block-averaging?
# TODO: Compare change of the average to RMSE of the average and not variable STDDEV?
from .random_walk import default_minfunc_name
from .beta_choice import gen_exp_beta_array
from .distributed_random_walk import DistributedRandomWalk
from copy import deepcopy
import numpy as np


class OptimizationProtocol:
    def __init__(
        self,
        minimized_function,
        minimized_function_name=default_minfunc_name,
        num_exploration_replicas=1,
        num_greedy_replicas=1,
        use_distributed_random_walk=True,
        num_processes=1,
        num_subpopulations=1,
        cloned_betas=True,
        num_beta_subpopulation_clones=2,
        num_internal_global_steps=1,
        subpopulation_propagation_seed=None,
        num_intermediate_propagations=1,
        randomized_change_params={},
        global_step_params={},
        greedy_delete_checked_paths=False,
        max_num_stagnating_iterations=1,
        max_num_iterations=None,
        save_random_walk_logs=False,
        beta_change_multiplier_bounds=(1.0, 2.0),
        init_beta_guess=1.0,
        init_lower_beta_guess=None,
        init_upper_beta_guess=None,
        target_largest_beta_minfunc_eff_std=None,
        target_tempering_acceptance_probability_interval=None,
        target_extrema_smallest_beta_log_prob_interval=None,
        significant_average_minfunc_change_rel_stddev=None,
        signif_av_minfunc_change_rel_stddev_largest_beta=None,
        signif_av_minfunc_change_rel_stddev_smallest_beta=None,
        init_egcs=None,
        init_egc=None,
        saved_candidates_max_difference=None,
        num_saved_candidates=1,
        terminated_worker_max_restart_number=0,
        terminated_worker_num_extra_rng_calls=1,
    ):
        """
        A very basic protocol for optimizing beta parameters during the simulation.
        Lowest temperature is adjusted for the standard deviation of minimized function in the corresponding ensemble to match a user-defined interval
        around a value of the order of the optimized model's error.
        Highest temperature is adjusted to match a user-defined interval in acceptance probability efficiency across the temperature ladder.
        target_largest_beta_minfunc_eff_std - target range for effective standard derivation of minimized function over largest beta in trajectory.
            Should approximately correspond to how accurately we would care to adjust the minimized function.
        target_extrema_smallest_beta_log_prob_interval - target range for logarithm of ratio of Boltzmann probability of the largest minimized function value accepted
            by the code and minimized function average over the smallest beta replica.
        significant_average_minfunc_change_rel_stddev - if normalized largest deviation of minimized function average over largest beta replicas
            deviates from the iteration mean by more than significant_average_minfunc_change_rel_stddev*(effective standard deviation) then
            iteration is considered non-equilibrated and unfit to judge whether beta values should be changed.
        """
        # What is minimized.
        self.minimized_function = minimized_function
        self.minimized_function_name = minimized_function_name
        # Beta parameters.
        self.num_exploration_replicas = num_exploration_replicas
        self.num_greedy_replicas = num_greedy_replicas
        if init_lower_beta_guess is None:
            init_lower_beta_guess = init_beta_guess
        if init_upper_beta_guess is None:
            init_upper_beta_guess = init_beta_guess
        self.lower_beta_value = init_lower_beta_guess
        self.upper_beta_value = init_upper_beta_guess
        # Replica propagation parameters.
        self.randomized_change_params = randomized_change_params
        self.global_step_params = global_step_params
        self.greedy_delete_checked_paths = greedy_delete_checked_paths
        # How long simulations are run between beta updates and how many of the latter are performed.
        self.num_internal_global_steps = num_internal_global_steps
        self.num_intermediate_propagations = num_intermediate_propagations
        self.max_num_stagnating_iterations = max_num_stagnating_iterations
        self.max_num_iterations = max_num_iterations
        # Parameters associated with parallelization.
        self.num_processes = num_processes
        self.num_subpopulations = num_subpopulations
        self.num_beta_subpopulation_clones = num_beta_subpopulation_clones
        self.use_distributed_random_walk = use_distributed_random_walk
        self.cloned_betas = cloned_betas
        # How much data is saved.
        self.saved_candidates_max_difference = saved_candidates_max_difference
        self.num_saved_candidates = num_saved_candidates
        self.save_random_walk_logs = save_random_walk_logs
        self.terminated_worker_max_restart_number = terminated_worker_max_restart_number
        self.terminated_worker_num_extra_rng_calls = (
            terminated_worker_num_extra_rng_calls
        )
        self.init_random_walk(
            init_egc=init_egc,
            init_egcs=init_egcs,
            subpopulation_propagation_seed=subpopulation_propagation_seed,
        )

        # How beta is updated.
        self.beta_change_multiplier_bounds = beta_change_multiplier_bounds
        self.target_largest_beta_minfunc_eff_std = target_largest_beta_minfunc_eff_std
        self.target_tempering_acceptance_probability_interval = (
            target_tempering_acceptance_probability_interval
        )
        self.target_extrema_smallest_beta_log_prob_interval = (
            target_extrema_smallest_beta_log_prob_interval
        )
        assert (
            self.target_largest_beta_minfunc_eff_std[1]
            > self.target_largest_beta_minfunc_eff_std[0]
        )
        assert self.target_largest_beta_minfunc_eff_std[0] >= 0.0
        self.significant_average_minfunc_change_rel_stddev = (
            significant_average_minfunc_change_rel_stddev
        )
        if signif_av_minfunc_change_rel_stddev_largest_beta is None:
            self.signif_av_minfunc_change_rel_stddev_largest_beta = (
                significant_average_minfunc_change_rel_stddev
            )
        else:
            self.signif_av_minfunc_change_rel_stddev_largest_beta = (
                signif_av_minfunc_change_rel_stddev_largest_beta
            )
        if signif_av_minfunc_change_rel_stddev_smallest_beta is None:
            self.signif_av_minfunc_change_rel_stddev_smallest_beta = (
                significant_average_minfunc_change_rel_stddev
            )
        else:
            self.signif_av_minfunc_change_rel_stddev_smallest_beta = (
                signif_av_minfunc_change_rel_stddev_smallest_beta
            )
        # Current status.
        self.largest_beta_equilibrated = False
        self.smallest_beta_equilibrated = False
        self.new_best_candidate_obtained = False

        # Attributes used to track optimization progress and update beta values.
        self.init_iteration_logs()

    def get_initial_average(self, beta_ids):
        init_av_minfunc = 0.0
        for beta_id in beta_ids:
            considered_tp = self.distributed_random_walk.current_trajectory_points[
                beta_id
            ]
            init_av_minfunc += considered_tp.calculated_data[
                self.minimized_function_name
            ]
        init_av_minfunc /= len(beta_ids)
        return init_av_minfunc

    def init_iteration_logs(self):
        """
        Initialize arrays where temporary data is saved.
        """
        self.best_candidate_log = [deepcopy(self.current_best_candidate())]
        self.num_stagnating_iterations = 0
        self.iteration_counter = 0
        drw = self.distributed_random_walk
        # Average of minimized function over largest real beta replicas over an interation with the same beta.
        # TODO if I implement restart make sure this is overridden.
        self.largest_beta_iteration_av_minfunc_log = [
            self.get_initial_average(drw.largest_beta_ids())
        ]
        self.smallest_beta_iteration_av_minfunc_log = [
            self.get_initial_average(drw.smallest_beta_ids())
        ]
        # Saves averages of minimized function over largest and smallest real beta replicas for a given Monte Carlo step.
        self.iter_tot_num_global_steps = (
            self.num_internal_global_steps * self.num_intermediate_propagations
        )
        self.smallest_beta_iteration_running_av_minfunc = np.empty(
            (self.iter_tot_num_global_steps,)
        )
        self.largest_beta_iteration_running_av_minfunc = np.empty(
            (self.iter_tot_num_global_steps,)
        )

    def init_random_walk(
        self, init_egc=None, init_egcs=None, subpopulation_propagation_seed=None
    ):
        """
        Initialize RandomWalk or DistributedRandomWalk object used for population propagation.
        """
        if self.use_distributed_random_walk:
            extra_intermediate_results_kwargs = {
                "num_exploration_replicas": self.num_exploration_replicas,
                "num_greedy_replicas": self.num_greedy_replicas,
                "cloned_betas": self.cloned_betas,
                "num_beta_subpopulation_clones": self.num_beta_subpopulation_clones,
                "track_worst_accepted_candidates": True,
                "num_minfunc_saved_steps": self.num_internal_global_steps,
            }
            self.distributed_random_walk = DistributedRandomWalk(
                min_function=self.minimized_function,
                min_function_name=self.minimized_function_name,
                num_processes=self.num_processes,
                num_subpopulations=self.num_subpopulations,
                subpopulation_propagation_seed=subpopulation_propagation_seed,
                num_internal_global_steps=self.num_internal_global_steps,
                betas=self.get_beta_values(),
                init_egcs=init_egcs,
                init_egc=init_egc,
                num_saved_candidates=self.num_saved_candidates,
                saved_candidates_max_difference=self.saved_candidates_max_difference,
                randomized_change_params=self.randomized_change_params,
                global_step_params=self.global_step_params,
                greedy_delete_checked_paths=self.greedy_delete_checked_paths,
                extra_intermediate_results_kwargs=extra_intermediate_results_kwargs,
                save_logs=self.save_random_walk_logs,
                terminated_worker_max_restart_number=self.terminated_worker_max_restart_number,
                terminated_worker_num_extra_rng_calls=self.terminated_worker_num_extra_rng_calls,
            )
        else:
            raise Exception("Using non-distributed random walks not implemented yet.")

    def get_beta_values(self):
        """
        beta values satisfying user-defined upper and lower bounds of the temperature ladder.
        """
        return gen_exp_beta_array(
            self.num_greedy_replicas,
            self.lower_beta_value,
            self.num_exploration_replicas,
            max_real_beta=self.upper_beta_value,
        )

    def update_beta_values(self):
        self.distributed_random_walk.init_betas(self.get_beta_values())

    def update_best_candidate_log(self):
        self.best_candidate_log.append(deepcopy(self.current_best_candidate()))

    def current_best_candidate(self):
        return self.saved_candidates()[0]

    def current_best_minfunc_value(self):
        return self.current_best_candidate().func_val

    def saved_candidates(self):
        return self.distributed_random_walk.saved_candidates

    def save_temp_propagation_data(self, propagation_counter):
        """
        Save temporary data associated with propagation.
        (For now only saves the average over largest beta replicas.)
        """
        smallest_beta_ids = self.distributed_random_walk.smallest_beta_ids()
        largest_beta_ids = self.distributed_random_walk.largest_beta_ids()
        true_step_id = propagation_counter * self.num_internal_global_steps
        for minfunc_vals in self.distributed_random_walk.minfunc_val_log:
            self.smallest_beta_iteration_running_av_minfunc[true_step_id] = np.mean(
                minfunc_vals[smallest_beta_ids]
            )
            self.largest_beta_iteration_running_av_minfunc[true_step_id] = np.mean(
                minfunc_vals[largest_beta_ids]
            )
            true_step_id += 1

    def reequilibration(self):
        for propagation_counter in range(self.num_intermediate_propagations):
            self.distributed_random_walk.propagate()
            self.save_temp_propagation_data(propagation_counter)

    def average_tempering_neighbor_acceptance_probability(self):
        return np.mean(
            self.distributed_random_walk.av_tempering_neighbors_acceptance_probability
        )

    def replica_eff_std(self, running_av_minfunc, beta_ids):
        # Calculate effective std of average over a beta value, then readjust to std of individual beta values.
        std_average_multiplier = np.sqrt(float(len(beta_ids)))
        return np.std(running_av_minfunc) * std_average_multiplier

    def largest_real_beta_eff_std(self):
        return self.replica_eff_std(
            self.largest_beta_iteration_running_av_minfunc,
            self.distributed_random_walk.largest_beta_ids(),
        )

    def smallest_real_beta_eff_std(self):
        return self.replica_eff_std(
            self.smallest_beta_iteration_running_av_minfunc,
            self.distributed_random_walk.smallest_beta_ids(),
        )

    def randomized_beta_change_multiplier(self):
        return np.random.uniform(*self.beta_change_multiplier_bounds)

    def check_largest_real_beta(self):
        largest_beta_eff_std = self.largest_real_beta_eff_std()
        largest_beta_eff_std_too_large = (
            largest_beta_eff_std > self.target_largest_beta_minfunc_eff_std[1]
        )
        largest_beta_eff_std_too_small = (
            largest_beta_eff_std < self.target_largest_beta_minfunc_eff_std[0]
        )
        beta_changed = largest_beta_eff_std_too_small or largest_beta_eff_std_too_large
        if beta_changed:
            beta_change_multiplier = self.randomized_beta_change_multiplier()
            if largest_beta_eff_std_too_small:
                beta_change_multiplier **= -1
            # Move the entire temperature ladder to adjust the largest beta stddev.
            self.upper_beta_value *= beta_change_multiplier
        return beta_changed

    def smallest_beta_extrema_rel_prob_log(self):
        drw = self.distributed_random_walk
        largest_extremum = max(cand.func_val for cand in drw.worst_accepted_candidates)
        smallest_beta_av_minfunc = self.smallest_beta_iteration_av_minfunc_log[-1]
        log_prob = drw.true_betas()[-1] * (
            largest_extremum - smallest_beta_av_minfunc
        )  # Should be positive.
        return log_prob

    def largest_beta_iteration_av_minfunc(self):
        return np.mean(self.largest_beta_iteration_running_av_minfunc)

    def smallest_beta_iteration_av_minfunc(self):
        return np.mean(self.smallest_beta_iteration_running_av_minfunc)

    def check_extrema_availability(self):
        """
        Check that the smallest beta replica can access the extrema of chemical space with ease regulated by user via target_extrema_smallest_beta_log_prob_interval
        attribute.
        """
        log_prob = self.smallest_beta_extrema_rel_prob_log()
        smallest_beta_too_small = (
            log_prob < self.target_extrema_smallest_beta_log_prob_interval[0]
        )
        smallest_beta_too_large = (
            log_prob > self.target_extrema_smallest_beta_log_prob_interval[1]
        )
        beta_changed = smallest_beta_too_small or smallest_beta_too_large
        if beta_changed:
            beta_change_multiplier = self.randomized_beta_change_multiplier()
            if smallest_beta_too_large:
                self.lower_beta_value /= beta_change_multiplier
            else:
                if self.lower_beta_value == self.upper_beta_value:
                    return False
                self.lower_beta_value *= beta_change_multiplier
        if self.lower_beta_value > self.upper_beta_value:
            # Note that we are counting on check_extrema_availability to be
            # called after check_largest_real_beta
            self.lower_beta_value = self.upper_beta_value
            beta_changed = True
        return beta_changed

    def check_smallest_beta(self):
        if self.target_extrema_smallest_beta_log_prob_interval is not None:
            return self.check_extrema_availability()
        raise Exception(
            "For now smallest beta is only adjusted based on target_extrema_smallest_beta_log_prob_interval."
        )

    def clear_random_walk_temp_data(self):
        drw = self.distributed_random_walk
        drw.worst_accepted_candidates = None
        drw.minfunc_val_log = None

    def replicas_equilibrated(
        self, av_minfunc_log, std_minfunc, signif_prop_coeff, beta_ids
    ):
        latest_av_minfunc = av_minfunc_log[-1]
        previous_av_minfunc = av_minfunc_log[-2]
        step_num_clones_normalization = np.sqrt(
            float(self.iter_tot_num_global_steps * len(beta_ids))
        )
        return (
            np.abs(previous_av_minfunc - latest_av_minfunc)
            * step_num_clones_normalization
            < signif_prop_coeff * std_minfunc
        )

    def check_largest_beta_equilibrated(self):
        return self.replicas_equilibrated(
            self.largest_beta_iteration_av_minfunc_log,
            self.largest_real_beta_eff_std(),
            self.signif_av_minfunc_change_rel_stddev_largest_beta,
            self.distributed_random_walk.largest_beta_ids(),
        )

    def check_smallest_beta_equilibrated(self):
        return self.replicas_equilibrated(
            self.smallest_beta_iteration_av_minfunc_log,
            self.smallest_real_beta_eff_std(),
            self.signif_av_minfunc_change_rel_stddev_smallest_beta,
            self.distributed_random_walk.smallest_beta_ids(),
        )

    def update_logs(self):
        self.largest_beta_iteration_av_minfunc_log.append(
            self.largest_beta_iteration_av_minfunc()
        )
        self.smallest_beta_iteration_av_minfunc_log.append(
            self.smallest_beta_iteration_av_minfunc()
        )

    def optimization_iteration(self):
        """
        Run an MC simulation and then re-adjust the upper and lower bounds of the beta ladder based on the simulation's results.
        """
        self.clear_random_walk_temp_data()
        self.reequilibration()
        self.update_logs()
        self.iteration_counter += 1
        self.largest_beta_equilibrated = self.check_largest_beta_equilibrated()
        if self.largest_beta_equilibrated:
            largest_beta_changed = self.check_largest_real_beta()
        else:
            largest_beta_changed = False
        self.smallest_beta_equilibrated = self.check_smallest_beta_equilibrated()
        if self.smallest_beta_equilibrated:
            smallest_beta_changed = self.check_smallest_beta()
        else:
            smallest_beta_changed = False
        beta_changed = largest_beta_changed or smallest_beta_changed
        self.equilibrated = (
            self.largest_beta_equilibrated and self.smallest_beta_equilibrated
        )
        self.best_candidate_log.append(deepcopy(self.current_best_candidate()))
        self.new_best_candidate_obtained = (
            self.best_candidate_log[-1] != self.best_candidate_log[-2]
        )

        self.update_beta_values()
        if self.new_best_candidate_obtained or beta_changed:
            self.num_stagnating_iterations = 0
        else:
            self.num_stagnating_iterations += 1
        if (self.num_stagnating_iterations >= self.max_num_stagnating_iterations) or (
            (self.max_num_iterations is not None)
            and (self.max_num_iterations <= self.iteration_counter)
        ):
            raise StopIteration

    # Treating the protocol as an iterator should be convenient.
    def __iter__(self):
        return self

    def __next__(self):
        self.optimization_iteration()
        return self.iteration_counter
