# Perform optimization in QM9* in a way that SHOULDN'T depend much on the initial choice of beta.
from mosaics.optimization_protocol import OptimizationProtocol
from mosaics.minimized_functions import OrderSlide
from mosaics import ExtGraphCompound
import random
import numpy as np
#from mosaics.rdkit_utils import canonical_SMILES_from_tp

max_nhatoms = 9  # Not 15 to cut down on the CPU time.

# Define initial molecule.
# All replicas are initialized in methane.
init_ncharges = [6]
init_adj_matrix = [[0]]
init_egc = ExtGraphCompound(
        nuclear_charges=init_ncharges,
        adjacency_matrix=init_adj_matrix,
        hydrogen_autofill=True,
    )
# With RdKit installed can also be done with:
#from mosaics.rdkit_utils import SMILES_to_egc
#init_egc=SMILES_to_egc("C")

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
}
global_step_params = {
    "num_parallel_tempering_tries": 64,
    "num_genetic_tries": 16,
    "prob_dict": {"simple": 0.5, "genetic": 0.25, "tempering": 0.25},
}

num_processes = 16  # 32
num_subpopulations = 32
num_exploration_replicas = 16
num_greedy_replicas = 1

seed = 1
random.seed(seed)
np.random.seed(seed)

opt_protocol = OptimizationProtocol(
    minimized_function,
    num_exploration_replicas=num_exploration_replicas,
    num_greedy_replicas=num_greedy_replicas,
    num_processes=num_processes,
    num_subpopulations=num_subpopulations,
    num_internal_global_steps=32,
    num_intermediate_propagations=8,
    randomized_change_params=randomized_change_params,
    global_step_params=global_step_params,
    max_num_stagnating_iterations=16,
    max_num_iterations=80,
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
    init_egc=init_egc,  # saved_candidates_max_difference=None,
    num_saved_candidates=40,
)

for iteration_id in opt_protocol:
    cur_best_cand = opt_protocol.current_best_candidate()
    print("___")
    print("___Best candidate at iteration", iteration_id, ":", cur_best_cand)
#    print("___Best candidate SMILES:", canonical_SMILES_from_tp(cur_best_cand.tp))
    print(
        "___Beta bounds:", opt_protocol.lower_beta_value, opt_protocol.upper_beta_value
    )
    print(
        "___Largest real beta minimized function mean, effective std, and equilibration:",
        opt_protocol.largest_beta_iteration_av_minfunc(),
        opt_protocol.largest_real_beta_eff_std(),
        opt_protocol.largest_beta_equilibrated,
    )
    print(
        "___Smallest real beta minimized function mean, effective std, and equilibration:",
        opt_protocol.smallest_beta_iteration_av_minfunc(),
        opt_protocol.smallest_real_beta_eff_std(),
        opt_protocol.smallest_beta_equilibrated,
    )
    print(
        "___Average tempering neighbor acceptance probability:",
        opt_protocol.average_tempering_neighbor_acceptance_probability(),
    )
    print(
        "___Extrema/average probability density log:",
        opt_protocol.smallest_beta_extrema_rel_prob_log(),
    )
    print("___Sanity check:")
    drw = opt_protocol.distributed_random_walk
    print("___Largest beta ids:", drw.largest_beta_ids())
    print("___Smallest beta ids:", drw.smallest_beta_ids())

print("Final best candidates:")
for cand_id, candidate in enumerate(opt_protocol.saved_candidates()):
    print("Candidate", cand_id, ":", candidate)

# If you have RdKit installed it draw final candidates.
try:
    from mosaics.rdkit_draw_utils import draw_chemgraph_to_file
except ModuleNotFoundError:
    quit()

for cand_id, candidate in enumerate(opt_protocol.saved_candidates()):
    png_filename = "best_candidate_" + str(cand_id) + ".png"
    draw_chemgraph_to_file(candidate.tp.chemgraph(), png_filename, file_format="PNG")
