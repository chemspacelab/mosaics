# This script makes an example optimization run similar to the ones whose results are presented in the original MOSAiCS paper.
from mosaics.random_walk import RandomWalk
from mosaics.beta_choice import gen_exp_beta_array
from mosaics.rdkit_utils import SMILES_to_egc, canonical_SMILES_from_tp
from mosaics.rdkit_draw_utils import draw_chemgraph_to_file
from mosaics.minimized_functions.morfeus_quantity_estimates import (
    LinComb_Morfeus_xTB_code,
)

# SMILES of the molecule from which we will start optimization.
init_SMILES = "C"
# Parameter of the QM9* chemical space over which the property is optimized.
possible_elements = ["C", "N", "O", "F"]
forbidden_bonds = [(7, 7), (7, 8), (8, 8), (7, 9), (8, 9), (9, 9)]
not_protonated = [8, 9]
nhatoms_range = [1, 9]
# Define minimized function using parameters discussed in the MOSAiCS paper.
min_HOMO_LUMO_gap = 0.08895587351640835
quant_prop_coeff = 263.14266129033456
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

# 'xTB_res' denotes intermediate xTB data stored by default for each chemical graph visited by the trajectory.
# Including it into 'delete_temp_data' makes the code delete all 'xTB_res', leaving only the value of the minimized function.
delete_temp_data = ["xTB_res"]


# None corresponds to greedy optimization, other betas are used in a Metropolis scheme.
betas = gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0)
print("Chosen betas:", betas)
# Each replica starts at methane.
init_egcs = [SMILES_to_egc(init_SMILES) for _ in betas]

# On average, 300-400 global MC steps is enough to find the minimum over QM9*.
num_MC_steps = 500
make_restart_frequency = 100
# Soft exit is triggered by creating a file called "EXIT".
soft_exit_check_frequency = 100
# We do not use the history-dependent vias here.
bias_coeff = None

randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": nhatoms_range,
    "final_nhatoms_range": nhatoms_range,
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
    "not_protonated": not_protonated,
    "added_bond_orders": [1, 2, 3],
}
global_change_params = {
    "num_parallel_tempering_tries": 128,
    "num_genetic_tries": 32,
    "prob_dict": {"simple": 0.6, "genetic": 0.2, "tempering": 0.2},
}

rw = RandomWalk(
    init_egcs=init_egcs,
    bias_coeff=bias_coeff,
    randomized_change_params=randomized_change_params,
    betas=betas,
    min_function=minimized_function,
    min_function_name="constrained_solvation_energy",
    make_restart_frequency=make_restart_frequency,
    soft_exit_check_frequency=make_restart_frequency,
    restart_file="restart_file.pkl",
    num_saved_candidates=100,
    keep_histogram=True,
    delete_temp_data=delete_temp_data,
    greedy_delete_checked_paths=True,
)

print("Started candidate search.")
print("# MC_step # Minimum value found # Minimum SMILES")
for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)
    # Print information about the current known molecule with best minimized function value.
    cur_best_candidate = rw.saved_candidates[0]
    cur_min_func_val = cur_best_candidate.func_val
    cur_min_SMILES = canonical_SMILES_from_tp(cur_best_candidate.tp)
    print(MC_step, cur_min_func_val, cur_min_SMILES)

rw.make_restart()

print("Number of minimized function calls:", minimized_function.call_counter)

num_plotted_candidates = 5
print("Plotting", num_plotted_candidates, "best candidates saved.")
for cand_id in range(num_plotted_candidates):
    png_filename = "best_candidate_" + str(cand_id + 1) + ".png"
    draw_chemgraph_to_file(
        rw.saved_candidates[cand_id].tp.chemgraph(), png_filename, file_format="PNG"
    )
