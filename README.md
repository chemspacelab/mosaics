# mosaics
Evolutionary Monte Carlo algorithm for optimization in chemical space.

[![DOI](https://img.shields.io/badge/arXiv-2307.15563-b31b1b.svg)](https://arxiv.org/abs/2307.15563)

## Description

MOSAiCS is an Evolutionary Monte Carlo algorithm designed for optimizing target functions over the space of organic molecules. The algorithm combines the benefits of both genetic algorithms and Monte Carlo techniques, providing a powerful tool for complex optimization tasks in applied science.

For more information, please read our paper ["Evolutionary Monte Carlo of QM properties in chemical space: Electrolyte design"](https://arxiv.org/abs/2307.15563) published on arXiv.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
  - [Toy Minimization](#toy-minimization)
  - [xTB Property Optimization](#xtb-property-optimization)
  - [Distributed Random Walk](#distributed-random-walk)
  - [Blind Optimization Protocol](#blind-optimization-protocol)
  - [ChemSpaceSampler](#chemspacesampler)
- [Contributing](#contributing)
- [License](#license)
  
## Installation
```bash
# Clone the repository
git clone https://github.com/yourgithubusername/MOSAiCS.git

# Navigate to the directory
cd MOSAiCS

# Install the package
pip install .
```

## Usage
To use the package in your Python script, import it like so:
```python
import mosaics
```

## Examples
We have prepared several examples to help you get started with MOSAiCS. Each example is located in its own directory under the `examples/` folder.

### [Toy Minimization](examples/01_toy_minimization/)
This example shows the basics of the minimization algorithm.

### [xTB Property Optimization](examples/02_xTB_property_optimization/)
Optimize specific properties using extended Tight-Binding (xTB) calculations.

### [Distributed Random Walk](examples/03_distributed_random_walk/)
Learn to distribute computational work across multiple nodes or processors.

### [Blind Optimization Protocol](examples/04_blind_optimization_protocol/)
Demonstrates advanced optimization methods without relying on all typical variables.

### [ChemSpaceSampler](examples/05_chemspacesampler/)
Showcases the algorithm's ability to explore various regions of the chemical space.

## Contributing
We welcome contributions from the community.

## License
This project is licensed under the MIT License
