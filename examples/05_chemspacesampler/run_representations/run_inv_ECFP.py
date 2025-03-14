from mosaics.beta_choice import gen_exp_beta_array
from mosaics.minimized_functions import chemspace_potentials, inversion_potentials


def main():
    params = {
        "V_0_pot": 1,
        "NPAR": 2,
        "max_d": 0.1,
        "strictly_in": False,
        "Nsteps": 100,  # 200
        "bias_strength": "weak",
        "pot_type": "parabola",
        "possible_elements": ["C", "O", "N"],
        "not_protonated": None,
        "forbidden_bonds": None,
        "nhatoms_range": [1, 27],
        "betas": gen_exp_beta_array(4, 1.0, 32, max_real_beta=32.0),
        "make_restart_frequency": None,
        "rep_type": "2d",
        "nBits": 1024,
        "rep_name": "inv_ECFP",
        "strategy": "default",
        "d_threshold": 0.1,
        "verbose": True,
    }

    smiles_init, smiles_target = "C", "CCCO"
    X_target, _, _ = chemspace_potentials.initialize_from_smiles(
        smiles_target, nBits=params["nBits"]
    )
    MOLS, D = inversion_potentials.chemspacesampler_inv_ECFP(smiles_init, X_target, params=params)
    print("MOLS", MOLS[:10])
    print("D", D[:10])


if __name__ == "__main__":
    main()
