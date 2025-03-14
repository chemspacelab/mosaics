from mosaics.beta_choice import gen_exp_beta_array
from mosaics.minimized_functions import chemspace_potentials

params = {
    "min_d": 10,
    "max_d": 11,
    "strictly_in": True,
    "V_0_pot": 0.05,
    "V_0_synth": 0.05,
    "NPAR": 2,
    "Nsteps": 100,
    "bias_strength": "none",
    "possible_elements": ["C", "O", "N", "F"],
    "not_protonated": None,
    "forbidden_bonds": [(8, 9), (8, 8), (9, 9), (7, 7)],
    "nhatoms_range": [13, 16],
    "betas": gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
    "make_restart_frequency": None,
    "rep_type": "MolDescriptors",
    "synth_cut_soft": 3,
    "synth_cut_hard": 5,
    "rep_name": "MolDescriptors",
    "mmff_check": True,
    "verbose": True,
}
if __name__ == "__main__":
    MOLS, D = chemspace_potentials.chemspacesampler_MolDescriptors(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O", params=params
    )
    print(MOLS)
    print(D)
