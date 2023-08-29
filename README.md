# MOSAiCS Codebase

[![License Badge](https://img.shields.io/github/license/chemspacelab/mosaics)](https://github.com/chemspacelab/mosaics/blob/main/LICENSE)
[![Issues Badge](https://img.shields.io/github/issues/chemspacelab/mosaics)](https://github.com/chemspacelab/mosaics/issues)
[![Pull Requests Badge](https://img.shields.io/github/issues-pr/chemspacelab/mosaics)](https://github.com/chemspacelab/mosaics/pulls)
[![Contributors Badge](https://img.shields.io/github/contributors/chemspacelab/mosaics)](https://github.com/chemspacelab/mosaics/graphs/contributors)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)

Welcome to the MOSAiCS codebase! This repository contains a collection of scripts and modules that enable the optimization of molecular structures using the MOSAiCS algorithm. MOSAiCS stands for "Molecular Optimization by Simulated Annealing and Combinatorial Search" and is a method for exploring the chemical space to find optimized molecular structures.

## :microscope: Purpose and Functionalities

The MOSAiCS codebase provides the necessary tools to perform molecular optimization runs using the MOSAiCS algorithm. The main functionalities of this codebase include:

- **Random Walk**: The code uses the `RandomWalk` module from the `mosaics` package to perform a random walk in the chemical space.
- **Beta Choice**: The `gen_exp_beta_array` function from the `beta_choice` module is used to generate an array of beta values.
- **RDKit Utilities**: The `rdkit_utils` module provides utilities for converting SMILES strings to EGCs and generating canonical SMILES from tp.
- **RDKit Draw Utilities**: The `rdkit_draw_utils` module provides a function to draw a chemical graph to a file.
- **Minimized Functions**: The `morfeus_quantity_estimates` module provides a function to define a minimized function using parameters discussed in the MOSAiCS paper for EGP*.

## :wrench: Installation

To install the MOSAiCS codebase, follow these steps:

1. Clone the repository:

   ```bash
   git clone [https://github.com/chemspacelab/mosaics.git](https://github.com/chemspacelab/mosaics.git)
   ```

2. Change into the `mosaics` directory:

   ```bash
   cd mosaics
   ```

3. Install the package using `setuptools`:

   ```bash
   python setup.py install
   ```

## :package: Dependencies

The MOSAiCS codebase has the following dependencies:

- RDKit
- numpy
- igraph
- sortedcontainers

## :computer: Examples of Usage

Here is an example of how to use the MOSAiCS codebase to perform a molecular optimization run:

```python
from mosaics.random_walk import RandomWalk
from mosaics.beta_choice import gen_exp_beta_array
from mosaics.rdkit_utils import SMILES_to_egc, canonical_SMILES_from_tp
from mosaics.rdkit_draw_utils import draw_chemgraph_to_file
from mosaics.minimized_functions.morfeus_quantity_estimates import LinComb_Morfeus_xTB_code

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
    num_internal_global_steps=500,
    global_step_params=global_step_params,
    greedy_delete_checked_paths=True,
    num_saved_candidates=num_saved_candidates,
    debug=True,
    randomized_change_params=randomized_change_params,
    save_logs=True,
)
```

More can be found in the [examples](https://github.com/chemspacelab/mosaics/tree/main/examples) folder.

## :busts_in_silhouette: Authors

The MOSAiCS codebase was developed by Anders Steen Christensen and Konstantin Karandashev.

## :handshake: Contributing

Contributions to the MOSAiCS codebase are welcome! If you encounter any issues or have suggestions for improvements, please [open an issue](https://github.com/chemspacelab/mosaics/issues) on GitHub. 

When making a pull request, please ensure that your code adheres to the existing structure and conventions of the codebase. 

## :email: Support

For support or questions related to the MOSAiCS codebase, please [contact the authors](mailto:authors@example.com).

## :scroll: License

The MOSAiCS codebase is licensed under the [MIT License](https://github.com/chemspacelab/mosaics/blob/main/LICENSE).