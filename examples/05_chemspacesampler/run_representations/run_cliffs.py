from mosaics.minimized_functions import chemspace_potentials
from mosaics.beta_choice import gen_exp_beta_array

def main():
    params = {
        'min_d': 0.0,
        'max_d': 120.0,
        'strictly_in': True,
        'V_0_pot': 0.05,
        'V_0_synth': 0.05,
        'NPAR': 8,
        'Nsteps': 10,
        'bias_strength': "none",
        'possible_elements': ["C", "O", "N", "F"],
        'not_protonated': None, 
        'forbidden_bonds': [(8, 9), (8,8), (9,9), (7,7)],
        'nhatoms_range': [4, 8],
        'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        'make_restart_frequency': None,
        'rep_type': '3d',
        'rep_name':"BoB_cliffs",
        'synth_cut_soft':3,
        'synth_cut_hard':5,
        'ensemble': False,
        'property': 'MolLogP',
        'jump': 0.9,
        "verbose": True,
    }

    MOLS, D, P = chemspace_potentials.chemspacesampler_find_cliffs(smiles="CCCC", params=params)
    print(MOLS)
    print(D)
    print(P)

if __name__ == "__main__":
    main()