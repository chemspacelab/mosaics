# The same example script as in 02_xTB_property_optimization, but capitalizing on distributed parallelism.
# WARNING: It consumes a lot of CPU time. AND its occasional crash was the reason for the "10^7" crash remark in MOSAiCS paper SI.
import random

import numpy as np

from mosaics.beta_choice import gen_exp_beta_array
from mosaics.distributed_random_walk import DistributedRandomWalk
from mosaics.minimized_functions.morfeus_quantity_estimates import LinComb_Morfeus_xTB_code
from mosaics.rdkit_draw_utils import draw_chemgraph_to_file
from mosaics.rdkit_utils import SMILES_to_egc, canonical_SMILES_from_tp

random.seed(1)
np.random.seed(1)

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
nhatoms_range = [1, 15]

betas = gen_exp_beta_array(1, 8.0, 32, max_real_beta=0.125)

num_propagations = 20

NCPUs = 32

randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": nhatoms_range,
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
    "not_protonated": not_protonated,
}
global_step_params = {
    "num_parallel_tempering_attempts": 64,
    "num_crossover_attempts": 16,
    "prob_dict": {"simple": 0.5, "crossover": 0.25, "tempering": 0.25},
}


# Define minimized function using parameters discussed in the MOSAiCS paper for EGP*.
min_HOMO_LUMO_gap = 0.08895587351640835
quant_prop_coeff = 281.71189055703087
quantity_of_interest = "solvation_energy"
solvent = "water"
minimized_function = LinComb_Morfeus_xTB_code(
    [quantity_of_interest],
    [quant_prop_coeff],
    constr_quants=["HOMO_LUMO_gap"],
    cq_lower_bounds=[min_HOMO_LUMO_gap],
    num_attempts=1,
    num_conformers=8,
    remaining_rho=0.9,
    ff_type="MMFF94",
    solvent=solvent,
)

num_saved_candidates = 40
drw = DistributedRandomWalk(
    betas=betas,
    init_egc=SMILES_to_egc(init_SMILES),
    min_function=minimized_function,
    num_processes=NCPUs,
    num_subpopulations=NCPUs,
    num_internal_global_steps=100,
    global_step_params=global_step_params,
    greedy_delete_checked_paths=True,
    num_saved_candidates=num_saved_candidates,
    debug=True,
    randomized_change_params=randomized_change_params,
    save_logs=True,
)


for propagation_step in range(num_propagations):
    drw.propagate()
    print(
        "___BEST___", propagation_step, drw.saved_candidates[0]
    )  # The "___BEST___" is added to easily grep out all the rdkit error messages.


for i, cur_cand in enumerate(drw.saved_candidates):
    print("Best molecule", i, ":", canonical_SMILES_from_tp(cur_cand.tp))
    print("Value of minimized function:", cur_cand.func_val)
    png_filename = "best_candidate_" + str(i) + ".png"
    draw_chemgraph_to_file(cur_cand.tp.chemgraph(), png_filename, file_format="PNG")

total_call_counter = 0
for prop_outputs in drw.propagation_logs:
    for prop_output in prop_outputs:
        random_walk = prop_output[1]
        total_call_counter += random_walk.min_function.call_counter

print("Number of minimized function calls:", total_call_counter)
