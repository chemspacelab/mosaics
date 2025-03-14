from mosaics.beta_choice import gen_exp_beta_array
from mosaics.minimized_functions import chemspace_potentials
import matplotlib.pyplot as plt

def main():
    params = {
        "min_d": 0.0,
        "max_d": 130.0,
        "strictly_in": False,
        "V_0_pot": 0.05,
        "V_0_synth": 0.05,
        "NPAR": 16,
        "Nsteps": 20,
        "bias_strength": "none",
        "possible_elements": ["C", "O", "N", "F"],
        "not_protonated": None,
        "forbidden_bonds": [(8, 9), (8, 8), (9, 9), (7, 7)],
        "nhatoms_range": [6, 6],
        "betas": gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        "make_restart_frequency": None,
        "rep_type": "3d",
        "rep_name": "BoB_cliffs",
        "synth_cut_soft": 3,
        "synth_cut_hard": 9,
        "ensemble": False,
        "property": "gap",
        "jump": 1.5,
        "verbose": True,
    }

    MOLS, D, P, P0 = chemspace_potentials.chemspacesampler_find_cliffs(
        smiles="C1=CC=CC=C1", params=params
    )
    print(MOLS[:10])
    print(D[:10])
    print(P[:10])
    print(P0)

    #create a new plot 
    fig, ax = plt.subplots()
    ax.plot(D, P, label="Property")
    #horizontal line at P0
    ax.axhline(y=P0, color='r', linestyle='--', label="P0")
    plt.show()
if __name__ == "__main__":
    main()
