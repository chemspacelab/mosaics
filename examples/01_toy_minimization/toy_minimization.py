import random

import numpy as np

from mosaics import ExtGraphCompound, RandomWalk
from mosaics.beta_choice import gen_exp_beta_array
from mosaics.minimized_functions import OrderSlide

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
    num_saved_candidates=100,
    debug=True,
)

for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)
    print(MC_step, rw.cur_tps)
    print("BEST VAL:", rw.saved_candidates[0].func_val)

rw.make_restart()

print()

print("Move statistics:")
for k, val in rw.move_statistics().items():
    print(k, ":", val)

print(
    "Number of calls vs histogram size:",
    rw.min_function.call_counter,
    len(rw.histogram),
)

num_printed_saved_candidates = 20

for i, cur_cand in enumerate(rw.saved_candidates[:num_printed_saved_candidates]):
    print("Best molecule", i, ":", cur_cand.tp)
    print("Value of minimized function:", cur_cand.func_val)
    print("Found on step:", cur_cand.tp.first_global_MC_step_encounter)
