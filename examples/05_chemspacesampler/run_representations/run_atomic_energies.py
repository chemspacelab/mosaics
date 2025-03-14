from mosaics.beta_choice import gen_exp_beta_array
from mosaics.minimized_functions import chemspace_potentials


def main():
    params = {
        "min_d": 0.0,
        "max_d": 0.0,
        "strictly_in": False,
        "V_0_pot": 0.05,
        "V_0_synth": 0.05,
        "NPAR": 2,
        "Nsteps": 50,
        "bias_strength": "none",
        "possible_elements": ["C", "O", "N"],
        "not_protonated": None,
        "forbidden_bonds": [(8, 9), (8, 8), (9, 9), (7, 7)],
        "nhatoms_range": [13, 13],
        "betas": gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        "make_restart_frequency": None,
        "rep_type": "3d",
        "rep_name": "atomic_energy",
        "rep_dict": "michael",
        "synth_cut_soft": 8,
        "synth_cut_hard": 9,
        "ensemble": False,
        "verbose": True,
    }
    MOLS, D = chemspace_potentials.chemspacesampler_atomization_rep(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O", params=params
    )
    print(MOLS)
    print(D)


if __name__ == "__main__":
    main()
