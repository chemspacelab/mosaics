from mosaics.minimized_functions import chemspace_potentials
from mosaics.beta_choice import gen_exp_beta_array
import pdb  
def main():
    params = {
        'V_0_pot': 0.5,
        'NPAR': 24,
        'max_d': 0.1,
        'strictly_in': False,
        'Nsteps': 10000, # 200
        'bias_strength': "stronger",
        'pot_type': 'parabola',
        'possible_elements': ["C", "O", "N", "F"],
        'not_protonated': None, 
        'forbidden_bonds': None,
        'nhatoms_range': [1, 20],
        'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        'make_restart_frequency': None,
        "rep_type": "2d",
        "nBits": 2048,
        'rep_name': 'inv_ECFP',
        'strategy': 'modify_pot',
        'd_threshold': 0.1,
        'Nparts': 100, # 12
        'growth_factor': 1.5,
        "verbose": False
    }

    smiles_init, smiles_target ="C", "CCCC"
    X_target, _, _ = chemspace_potentials.initialize_from_smiles(smiles_target,nBits=params['nBits'])
    #pdb.set_trace()
    MOLS, D = chemspace_potentials.chemspacesampler_inv_ECFP(smiles_init,X_target, params=params)    
    print("MOLS", MOLS)
    print("D", D)
if __name__ == "__main__":
    main()