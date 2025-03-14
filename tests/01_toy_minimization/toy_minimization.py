# Script with an example of optimizing a toy problem in chemical graph space.
import random

import numpy as np

from mosaics import ExtGraphCompound, RandomWalk
from mosaics.beta_choice import gen_exp_beta_array
from mosaics.minimized_functions import OrderSlide
from mosaics.test_utils import SimulationLogIO
from mosaics.valence_treatment import set_color_defining_neighborhood_radius

set_color_defining_neighborhood_radius(1)
random.seed(1)
np.random.seed(1)

possible_elements = ["C", "N", "O", "F", "P", "S"]

forbidden_bonds = [
    (7, 7),
    (8, 8),
    (9, 9),
    (7, 8),
    (7, 9),
    (8, 9),
    (15, 15),
    (16, 16),
    (15, 16),
]

betas = gen_exp_beta_array(4, 8.0, 16, max_real_beta=0.5)

num_MC_steps = 2000

bias_coeff = 0.1

randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [1, 16],
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
    "not_protonated": [16],  # S not protonated
}
global_change_params = {
    "num_parallel_tempering_attempts": 5,
    "num_crossover_attempts": 5,
    "prob_dict": {"simple": 0.5, "crossover": 0.25, "tempering": 0.25},
}

# All replicas are initialized in methane.
init_ncharges = [6]
init_adj_matrix = [[0]]

init_egcs = [
    ExtGraphCompound(
        nuclear_charges=init_ncharges,
        adjacency_matrix=init_adj_matrix,
        hydrogen_autofill=True,
    )
    for _ in betas
]

min_func = OrderSlide(possible_elements=possible_elements)

rw = RandomWalk(
    bias_coeff=bias_coeff,
    randomized_change_params=randomized_change_params,
    betas=betas,
    min_function=min_func,
    init_egcs=init_egcs,
    keep_histogram=True,
    keep_full_trajectory=True,
    restart_file="larger_mols_restart.pkl",
    saved_candidates_max_difference=0.5,
    debug=True,
)

sim_log = SimulationLogIO(
    filename="toy_minimization.log", benchmark_filename="toy_minimization_benchmark.log"
)
sim_log.print_timestamp(comment="SIM_START")

for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)
    if MC_step % 200 == 0:
        step_label = "STEP" + str(MC_step)
        sim_log.print_list(rw.cur_tps, comment=step_label)

rw.make_restart()

for k, val in rw.move_statistics().items():
    sim_log.print(val, comment=k)

sim_log.print(
    rw.min_function.call_counter,
    len(rw.histogram),
    comment="number_of_calls_vs_histogram_size",
)

sim_log.print_list(rw.saved_candidates, comment="SAVED_CANDIDATES", sorted_comparison=True)

sim_log.print_timestamp(comment="SIM_FINISH")

print("BENCHMARK AGREEMENT:", (not sim_log.difference_encountered))
