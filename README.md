## mosaics
Evolutionary Monte Carlo algorithm for optimization in chemical space.

[![DOI](https://img.shields.io/badge/arXiv-2307.15563-b31b1b.svg)](https://arxiv.org/abs/2307.15563)

<img src="cover.png" width="50%" height="50%" />

## :clipboard: Description

MOSAiCS is an Evolutionary Monte Carlo algorithm designed for optimizing target functions over the space of organic molecules. The algorithm combines the benefits of both genetic algorithms and Monte Carlo techniques, providing a powerful tool for complex optimization tasks in applied science.

For more information, please read our paper ["Evolutionary Monte Carlo of QM properties in chemical space: Electrolyte design"](https://arxiv.org/abs/2307.15563) published on arXiv.
  
## :package: Dependencies

MOSAiCS has the following dependencies:

- numpy
- igraph
- sortedcontainers
- scipy
- loky

Additional packages are required to run some example scripts.

## :wrench: Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/chemspacelab/mosaics
   ```

2. Change into the `mosaics` directory:

   ```bash
   cd mosaics
   ```

3. Install the package using `setuptools`
   ```bash
   python setup.py install
   ```
   or `pip`
   ```bash
   pip install .
   ```

## :toolbox: Usage
To use the package in your Python script import it with:
```python
import mosaics
```

## :computer: Examples
We have prepared several examples to help you get started with MOSAiCS. Each example is located in its own directory under the `examples/` folder.

### [Toy Minimization](examples/01_toy_minimization/)
This example shows the basics of the our algorithm by using it to minimize a function of chemical graph's nuclear charges over chemical space.

### [xTB Property Optimization](examples/02_xTB_property_optimization/)
Optimize solvation energy using extended Tight-Binding (xTB) calculations with MMFF94 conformers. (Largely a reproduction of the numerical experiments performed in [arXiv:2307.15563](https://arxiv.org/abs/2307.15563).) NOTE: Requires installation of [`morfeus`](https://pypi.org/project/morfeus-ml/) and [`xtb-python`](https://xtb-python.readthedocs.io/en/latest/#) packages.

### [Distributed Random Walk](examples/03_distributed_random_walk/)
Learn to distribute computational work across multiple nodes or processors.

### [Blind Optimization Protocol](examples/04_blind_optimization_protocol/)
Learn to use protocols for tuning beta parameters during optimization.

### [ChemSpaceSampler](examples/05_chemspacesampler/)
Showcases the algorithm's ability to explore various regions of the chemical space. For further reading see ["Understanding Representations by Exploring Galaxies in Chemical Space"](https://arxiv.org/abs/2309.09194) published on arXiv.

## :straight_ruler: Tests
Unfortunately, due to uncertain nature of Monte Carlo trajectories the only way to completely verify correctness of installation is by running a relatively long Monte Carlo calculation with a certain random number generator seed and compare it to the benchmark trajectory. Hence each example script in `examples` with a toy problem function also contains a benchmark `*.log` file to which the output can be compared; such calculations take significantly more time than a typical test. A reference environment for which the benchmarks could be reproduced is found in `examples/benchmark_env.yml` (in case, for example, an environment update changes the way random number generation works). Reproducing output of the example in `examples/01_toy_minimization` should be enough to verify installation.

## :handshake: Contributing
We welcome contributions and feedback from the community. If you encounter any issues or have suggestions for improvements, please [open an issue](https://github.com/chemspacelab/mosaics/issues) on GitHub.

## :scroll: License
This project is licensed under the MIT License

## :email: Support

For support or questions related to the MOSAiCS codebase, please [contact the authors](mailto:kvkarandashev@gmail.com).