from mosaics.beta_choice import gen_exp_beta_array
from mosaics.minimized_functions import chemspace_potentials


def main():
    params = {
        "min_d": 2.0,
        "max_d": 5.0,
        "strictly_in": True,
        "V_0_pot": 0.05,
        "V_0_synth": 0.05,
        "NPAR": 1,
        "Nsteps": 15,
        "bias_strength": "none",
        "possible_elements": ["C", "O", "N", "F"],
        "not_protonated": None,
        "forbidden_bonds": [(8, 9), (8, 8), (9, 9), (7, 7)],
        "nhatoms_range": [13, 16],
        "betas": gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        "make_restart_frequency": None,
        "rep_type": "2d",
        "nBits": 2048,
        "synth_cut_soft": 2,
        "synth_cut_hard": 5,
        "rep_name": "ECFP",
        "mmff_check": True,
        "verbose": True,
    }

    MOLS, D = chemspace_potentials.chemspacesampler_ECFP(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O", params=params
    )
    print(MOLS)
    print(D)


if __name__ == "__main__":
    main()
