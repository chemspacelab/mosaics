# Perform optimization in QM9* in a way that SHOULDN'T depend much on the initial choice of beta.
import random

import numpy as np

from mosaics.minimized_functions import OrderSlide
from mosaics.optimization_protocol import OptimizationProtocol
from mosaics.rdkit_draw_utils import draw_chemgraph_to_file
from mosaics.rdkit_utils import SMILES_to_egc
from mosaics.test_utils import SimulationLogIO

max_nhatoms = 9  # Not 15 to cut down on the CPU time.

init_SMILES = "C"

possible_elements = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br"]

not_protonated = [5, 8, 9, 14, 15, 16, 17, 35]
forbidden_bonds = [
    (7, 7),
    (7, 8),
    (8, 8),
    (7, 9),
    (8, 9),
    (9, 9),
    (7, 17),
    (8, 17),
    (9, 17),
    (17, 17),
    (7, 35),
    (8, 35),
    (9, 35),
    (17, 35),
    (35, 35),
    (15, 15),
    (16, 16),
]
nhatoms_range = [1, max_nhatoms]

# Define minimized function using parameters discussed in the MOSAiCS paper for EGP*.
minimized_function = OrderSlide(possible_elements=possible_elements)

target_largest_beta_minfunc_eff_std = (0.5, 2.0)


randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": nhatoms_range,
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
    "not_protonated": not_protonated,
    "crossover_max_num_affected_bonds": 3,
    "linear_scaling_elementary_mutations": False,
    "linear_scaling_crossover_moves": False,
}
global_step_params = {
    "num_parallel_tempering_attempts": 64,
    "num_crossover_attempts": 16,
    "prob_dict": {"simple": 0.5, "crossover": 0.25, "tempering": 0.25},
}

num_processes = 16  # 32
num_subpopulations = 32
num_exploration_replicas = 16
num_greedy_replicas = 1

# Simulation thoroughness.
max_num_stagnating_iterations = 16
max_num_iterations = 80
num_internal_global_steps = 32
num_intermediate_propagations = 8

seed = 1
random.seed(seed)
np.random.seed(seed)

opt_protocol = OptimizationProtocol(
    minimized_function,
    num_exploration_replicas=num_exploration_replicas,
    num_greedy_replicas=num_greedy_replicas,
    num_processes=num_processes,
    num_subpopulations=num_subpopulations,
    num_internal_global_steps=num_internal_global_steps,
    num_intermediate_propagations=num_intermediate_propagations,
    randomized_change_params=randomized_change_params,
    global_step_params=global_step_params,
    max_num_stagnating_iterations=max_num_stagnating_iterations,
    max_num_iterations=max_num_iterations,
    beta_change_multiplier_bounds=(1.0, 4.0),
    init_beta_guess=1.0,
    target_largest_beta_minfunc_eff_std=target_largest_beta_minfunc_eff_std,
    target_extrema_smallest_beta_log_prob_interval=(
        0.5,
        2.0,
    ),  # target_tempering_acceptance_probability_interval=(0.25, 0.5),
    significant_average_minfunc_change_rel_stddev=16.0,
    subpopulation_propagation_seed=seed,
    greedy_delete_checked_paths=True,
    init_egc=SMILES_to_egc(init_SMILES),  # saved_candidates_max_difference=None,
    saved_candidates_max_difference=0.5,
    debug=True,
)

sim_log = SimulationLogIO(filename="toy_opt.log", benchmark_filename="toy_opt_benchmark.log")
sim_log.print_timestamp(comment="SIM_START")

for iteration_id in opt_protocol:
    cur_best_cand = opt_protocol.current_best_candidate()
    sim_log.print_list(
        opt_protocol.saved_candidates(),
        comment="BEST_CANDS_ITER_" + str(iteration_id),
        sorted_comparison=True,
    )
    sim_log.print(
        opt_protocol.lower_beta_value,
        opt_protocol.upper_beta_value,
        comment="BETA_VALUES_" + str(iteration_id),
    )
    sim_log.print(
        opt_protocol.largest_beta_iteration_av_minfunc(),
        opt_protocol.largest_real_beta_eff_std(),
        comment="largest_beta_minfunc_mean_and_effective_std" + str(iteration_id),
    )
    sim_log.print(
        opt_protocol.largest_beta_equilibrated,
        comment="largest_beta_equilibrated_" + str(iteration_id),
    )
    sim_log.print(
        opt_protocol.smallest_beta_iteration_av_minfunc(),
        opt_protocol.smallest_real_beta_eff_std(),
        comment="smallest_beta_minfunc_mean_and_effective_std" + str(iteration_id),
    )
    sim_log.print(
        opt_protocol.smallest_beta_equilibrated,
        comment="smallest_beta_equilibrated_" + str(iteration_id),
    )
    sim_log.print(
        opt_protocol.average_tempering_neighbor_acceptance_probability(),
        comment="average_tempering_neighbor_acceptance_probability",
    )
    sim_log.print(
        opt_protocol.smallest_beta_extrema_rel_prob_log(),
        comment="extrema/average_prob_density_log",
    )
    drw = opt_protocol.distributed_random_walk
    sim_log.print_list(drw.largest_beta_ids(), comment="largest_beta_ids")
    sim_log.print_list(drw.smallest_beta_ids(), comment="smallest_beta_ids")

sim_log.print_list(
    opt_protocol.saved_candidates(),
    comment="FINAL_BEST_CANDIDATES",
    sorted_comparison=True,
)
sim_log.print_timestamp(comment="SIM_FINISH")

print("Final best candidates:")
for cand_id, candidate in enumerate(opt_protocol.saved_candidates()):
    print("Candidate", cand_id, ":", candidate)
    png_filename = "best_candidate_" + str(cand_id) + ".png"
    draw_chemgraph_to_file(candidate.tp.chemgraph(), png_filename, file_format="PNG")
