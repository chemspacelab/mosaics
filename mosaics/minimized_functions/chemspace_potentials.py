# Standard library imports
import copy
import glob
import os
import random
import shutil

# from rdkit.Contrib.SA_Score import sascorer
import sys
import tempfile
import time

# Third party imports
import numpy as np
import rdkit.Chem.Crippen as Crippen
from numpy.linalg import norm
from rdkit import Chem
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import pandas as pd
import sascorer
from joblib import Parallel, delayed
from rdkit.Chem import AllChem
from tqdm import tqdm

# Local application/library specific imports
from mosaics import RandomWalk
from mosaics.beta_choice import gen_exp_beta_array
from mosaics.data import *
from mosaics.minimized_functions import chemspace_sampler_default_params
from mosaics.minimized_functions.morfeus_quantity_estimates import (
    morfeus_coord_info_from_tp,
    morfeus_FF_xTB_code_quants,
)
from mosaics.minimized_functions.representations import *
from mosaics.misc_procedures import str_atom_corr
from mosaics.random_walk import TrajectoryPoint
from mosaics.trajectory_analysis import ordered_trajectory_from_restart
from mosaics.rdkit_utils import RdKitFailure, SMILES_to_egc, chemgraph_to_canonical_rdkit
from mosaics.utils import loadpkl

import pdb

# /home/jan/executables/mosaics/examples/03_chemspacesampler/representations.py


def trajectory_point_to_canonical_rdkit(tp_in, SMILES_only=False):
    """
    Converts a trajectory point to a canonical RDKit molecule.

    Args:
        tp_in: A trajectory point (typically an instance of the TrajectoryPoint class)
        SMILES_only: If True, only the SMILES string of the molecule is returned. Otherwise,
                     a RDKit molecule object is returned. Default is False.

    Returns:
        RDKit molecule object or SMILES string, based on the value of SMILES_only.
    """
    return chemgraph_to_canonical_rdkit(tp_in.egc.chemgraph, SMILES_only=SMILES_only)


def max_element_counts(elements):
    # Initialize an empty dictionary to store the maximum counts
    max_counts = {}

    # Iterate over each sub-array in the main array
    for sub_array in elements:
        # Count the occurrences of each element in the sub-array
        unique, counts = np.unique(sub_array, return_counts=True)
        count_dict = dict(zip(unique, counts))

        # Compare the counts with the current maximum counts
        for element, count in count_dict.items():
            if element not in max_counts or count > max_counts[element]:
                max_counts[element] = count

    # find the maximum number of each element in the list of lists
    max_n = max([len(x) for x in elements])

    return max_counts, max_n


def get_boltzmann_weights(energies, T=300):
    """
    Calculate the Boltzmann weights for a set of energies at a given temperature.

    Parameters
    ----------
    energies : np.array of shape (n,)
        1-D array containing the energies of 'n' states.
    T : float, optional
        Temperature in Kelvin. Default is 300 K.

    Returns
    -------
    boltzmann_weights : np.array of shape (n,)
        1-D array containing the Boltzmann weights corresponding to the input energies.
    """
    beta = 1 / (K_B_KCAL_PER_MOL_PER_K * T)
    boltzmann_weights = np.exp(-energies * beta)
    boltzmann_weights /= np.sum(boltzmann_weights)
    return boltzmann_weights


def fml_rep_SOAP(COORDINATES, NUC_CHARGES, WEIGHTS, possible_elements=["C", "O", "N", "F"]):
    """
    Calculate the FML (Free Energy Machine Learning) representation, which is the Boltzmann-weighted SOAP representation.

    Parameters
    ----------
    COORDINATES : np.array of shape (n, m, 3)
        3-D array containing the coordinates of 'm' atoms for 'n' states.
    NUC_CHARGES : np.array of shape (m,)
        1-D array containing the nuclear charges of 'm' atoms.
    WEIGHTS : np.array of shape (n,)
        1-D array containing the weights for 'n' states.
    possible_elements : list of strings, optional
        List of possible elements present in the molecules. Default is ["C", "O", "N", "F"].

    Returns
    -------
    fml_rep_SOAP : np.array of shape (N,) where 'N' is the dimension of the SOAP vector
        FML representation of the input molecular system.
    """
    X = []

    for i in range(len(COORDINATES)):
        X.append(gen_soap(COORDINATES[i], NUC_CHARGES, possible_elements))
    X = np.array(X)
    X = np.average(X, axis=0, weights=WEIGHTS)
    return X


def fml_rep_bob(coords, symbols, WEIGHTS, params):
    X = []
    for i in range(len(coords)):
        X.append(generate_bob(symbols, coords[i], asize=params["asize"]))
    X = np.array(X)
    X = np.average(X, axis=0, weights=WEIGHTS)
    return X


def fml_rep_CM(coords, charges, WEIGHTS, pad):
    X = []
    for i in range(len(coords)):
        X.append(generate_CM(coords[i], charges, pad))
    X = np.array(X)
    X = np.average(X, axis=0, weights=WEIGHTS)
    return X


def fml_rep_MBDF(coords, charges, WEIGHTS, params):
    X = []
    for i in range(len(coords)):
        X.append(global_MBDF_bagged_wrapper(charges, coords[i], params))
    X = np.array(X)
    X = np.average(X, axis=0, weights=WEIGHTS)
    return X


class potential_SOAP:
    """
    Class to represent a potential using Smooth Overlap of Atomic Positions (SOAP).
    """

    def __init__(self, X_init, params):
        """
        Initializes the potential_SOAP class.

        Parameters:
        X_init (np.array): Initial representation.
        gamma (float): A parameter for flat_parabola_potential function.
        sigma (float): A parameter for flat_parabola_potential function.
        possible_elements (list): List of possible atomic elements.
        verbose (bool): Verbosity flag.
        """
        self.params = params
        self.X_init = X_init
        self.gamma = params["min_d"]
        self.sigma = params["max_d"]
        self.V_0_pot = params["V_0_pot"]
        self.V_0_synth = params["V_0_synth"]
        self.possible_elements = params["possible_elements"] + ["H"]
        self.verbose = params["verbose"]
        self.synth_cut_soft = params["synth_cut_soft"]
        self.synth_cut_hard = params["synth_cut_hard"]
        self.ensemble = params["ensemble"]
        self.morfeus_output = {"morfeus": morfeus_coord_info_from_tp}

        if self.ensemble:
            self.morfeus_args = chemspace_sampler_default_params.morfeus_args_confs
        else:
            self.morfeus_args = chemspace_sampler_default_params.morfeus_args_single

        self.potential = self.flat_parabola_potential

    def flat_parabola_potential(self, d):
        """
        A function to describe the potential as a flat parabola.

        Parameters:
        d (float): Distance.

        Returns:
        float: Potential value.
        """

        if d < self.gamma:
            return self.V_0_pot * (d - self.gamma) ** 2
        if self.gamma <= d <= self.sigma:
            return 0
        if d > self.sigma:
            return self.V_0_pot * (d - self.sigma) ** 2

    def synth_potential(self, score):
        if score > self.synth_cut_hard:
            return None
        if score > self.synth_cut_soft:
            return self.V_0_synth * (score - self.synth_cut_soft) ** 2
        else:
            return 0

    def __call__(self, trajectory_point_in):
        """
        Calculates the potential energy of a trajectory point.

        Parameters:
        trajectory_point_in (TrajectoryPoint): Trajectory point.

        Returns:
        float: Potential energy of the trajectory point.
        """

        try:
            output = trajectory_point_in.calc_or_lookup(
                self.morfeus_output,
                kwargs_dict=self.morfeus_args,
            )["morfeus"]

            coords = output["coordinates"]
            charges = output["nuclear_charges"]
            SMILES = output["canon_rdkit_SMILES"]

            rdkit_mol_no_H = Chem.RemoveHs(Chem.MolFromSmiles(SMILES))
            score = sascorer.calculateScore(rdkit_mol_no_H)

            V_synth = self.synth_potential(score)
            if V_synth == None:
                return None

            if self.ensemble:
                if coords.shape[1] == charges.shape[0]:
                    X_test = fml_rep_SOAP(
                        coords,
                        charges,
                        output["rdkit_Boltzmann"],
                        possible_elements=self.possible_elements,
                    )
                else:
                    return None
            else:
                if coords.shape[0] == charges.shape[0]:
                    X_test = gen_soap(coords, charges, species=self.possible_elements)
                else:
                    return None

        except Exception as e:
            print(e)
            print("Error in 3d conformer sampling")
            return None

        distance = norm(X_test - self.X_init)
        V = self.potential(distance) + V_synth

        if self.verbose:
            print(SMILES, distance, V)
        return V


class potential_MolDescriptors:
    def __init__(self, X_init, params):
        self.X_init = X_init
        self.gamma = params["min_d"]
        self.sigma = params["max_d"]
        self.V_0_pot = params["V_0_pot"]
        self.V_0_synth = params["V_0_synth"]
        self.synth_cut_soft = params["synth_cut_soft"]
        self.synth_cut_hard = params["synth_cut_hard"]
        self.mmff_check = params["mmff_check"]
        self.verbose = params["verbose"]

        self.canonical_rdkit_output = {"canonical_rdkit": trajectory_point_to_canonical_rdkit}

        self.potential = self.flat_parabola_potential
        # self.norm_init = norm(X_init)

    def flat_parabola_potential(self, d):
        """
        Flat parabola potential. Allows sampling within a distance basin
        interval of I in [gamma, sigma]. The potential is given by:
        """

        if d < self.gamma:
            return self.V_0_pot * (d - self.gamma) ** 2
        if self.gamma <= d <= self.sigma:
            return 0
        if d > self.sigma:
            return self.V_0_pot * (d - self.sigma) ** 2

    def synth_potential(self, score):
        if score > self.synth_cut_hard:
            return None
        if score > self.synth_cut_soft:
            return self.V_0_synth * (score - self.synth_cut_soft) ** 2
        else:
            return 0

    def __call__(self, trajectory_point_in):
        rdkit_mol, canon_SMILES = trajectory_point_in.calc_or_lookup(self.canonical_rdkit_output)[
            "canonical_rdkit"
        ]

        if rdkit_mol is None:
            raise RdKitFailure

        rdkit_mol_no_H = Chem.RemoveHs(rdkit_mol)
        score = sascorer.calculateScore(rdkit_mol_no_H)

        V_synth = self.synth_potential(score)
        if V_synth == None:
            return None

        if self.mmff_check:
            if not AllChem.MMFFHasAllMoleculeParams(rdkit_mol):
                return None

        X_test = calc_all_descriptors(rdkit_mol)
        d = norm(X_test - self.X_init)  # /self.norm_init
        V = self.potential(d) + V_synth

        if self.verbose:
            print(canon_SMILES, d, V, score)

        return V


class potential_BoB:
    """
    Class to represent a potential using Bag of Bonds (BoB).
    """

    def __init__(self, X_init, params):
        """
        Initializes the potential_BoB class.

        Parameters:
        X_init (np.array): Initial positions of particles.
        gamma (float): A parameter for flat_parabola_potential function.
        sigma (float): A parameter for flat_parabola_potential function.
        possible_elements (list): List of possible atomic elements.
        verbose (bool): Verbosity flag.
        """
        self.params = params
        self.X_init = X_init
        self.gamma = params["min_d"]
        self.sigma = params["max_d"]
        self.V_0_pot = params["V_0_pot"]
        self.V_0_synth = params["V_0_synth"]
        self.possible_elements = params["possible_elements"] + ["H"]
        self.verbose = params["verbose"]
        self.synth_cut_soft = params["synth_cut_soft"]
        self.synth_cut_hard = params["synth_cut_hard"]
        self.ensemble = params["ensemble"]
        self.morfeus_output = {"morfeus": morfeus_coord_info_from_tp}

        if self.ensemble:
            self.morfeus_args = chemspace_sampler_default_params.morfeus_args_confs
        else:
            self.morfeus_args = chemspace_sampler_default_params.morfeus_args_single

        self.potential = self.flat_parabola_potential

    def flat_parabola_potential(self, d):
        """
        A function to describe the potential as a flat parabola.

        Parameters:
        d (float): Distance.

        Returns:
        float: Potential value.
        """

        if d < self.gamma:
            return self.V_0_pot * (d - self.gamma) ** 2
        if self.gamma <= d <= self.sigma:
            return 0
        if d > self.sigma:
            return self.V_0_pot * (d - self.sigma) ** 2

    def synth_potential(self, score):
        if score > self.synth_cut_hard:
            return None
        if score > self.synth_cut_soft:
            return self.V_0_synth * (score - self.synth_cut_soft) ** 2
        else:
            return 0

    def __call__(self, trajectory_point_in):
        """
        Calculates the potential energy of a trajectory point.

        Parameters:
        trajectory_point_in (TrajectoryPoint): Trajectory point.

        Returns:
        float: Potential energy of the trajectory point.
        """

        try:
            output = trajectory_point_in.calc_or_lookup(
                self.morfeus_output,
                kwargs_dict=self.morfeus_args,
            )["morfeus"]

            coords = output["coordinates"]
            charges = output["nuclear_charges"]
            symbols = [str_atom_corr(charge) for charge in charges]
            SMILES = output["canon_rdkit_SMILES"]

            rdkit_mol_no_H = Chem.RemoveHs(Chem.MolFromSmiles(SMILES))
            score = sascorer.calculateScore(rdkit_mol_no_H)

            V_synth = self.synth_potential(score)
            if V_synth == None:
                return None

            if self.ensemble:
                if coords.shape[1] == charges.shape[0]:
                    X_test = fml_rep_bob(coords, symbols, output["rdkit_Boltzmann"], self.params)
                else:
                    return None

            else:
                if coords.shape[0] == charges.shape[0]:
                    X_test = generate_bob(symbols, coords, asize=self.params["asize"])

                else:
                    return None

        except Exception as e:
            print(e)
            print("Error in 3d conformer sampling")
            return None

        distance = np.linalg.norm(X_test - self.X_init)
        V = self.potential(distance) + V_synth

        if self.verbose:
            print(SMILES, distance, V)
        return V


class potential_CM:
    """
    Class to represent a potential using Coulomb Matrix (CM).
    """

    def __init__(self, X_init, params):
        """
        Initializes the potential_CM class.

        Parameters:
        X_init (np.array): Initial representation vector
        gamma (float): A parameter for flat_parabola_potential function.
        sigma (float): A parameter for flat_parabola_potential function.
        possible_elements (list): List of possible atomic elements.
        verbose (bool): Verbosity flag.
        """
        self.params = params
        self.X_init = X_init
        self.gamma = params["min_d"]
        self.sigma = params["max_d"]
        self.V_0_pot = params["V_0_pot"]
        self.V_0_synth = params["V_0_synth"]
        self.verbose = params["verbose"]
        self.synth_cut_soft = params["synth_cut_soft"]
        self.synth_cut_hard = params["synth_cut_hard"]
        self.ensemble = params["ensemble"]
        self.morfeus_output = {"morfeus": morfeus_coord_info_from_tp}

        if self.ensemble:
            self.morfeus_args = chemspace_sampler_default_params.morfeus_args_confs
        else:
            self.morfeus_args = chemspace_sampler_default_params.morfeus_args_single

        self.potential = self.flat_parabola_potential

    def flat_parabola_potential(self, d):
        """
        A function to describe the potential as a flat parabola.

        Parameters:
        d (float): Distance.

        Returns:
        float: Potential value.
        """

        if d < self.gamma:
            return self.V_0_pot * (d - self.gamma) ** 2
        if self.gamma <= d <= self.sigma:
            return 0
        if d > self.sigma:
            return self.V_0_pot * (d - self.sigma) ** 2

    def synth_potential(self, score):
        if score > self.synth_cut_hard:
            return None
        if score > self.synth_cut_soft:
            return self.V_0_synth * (score - self.synth_cut_soft) ** 2
        else:
            return 0

    def __call__(self, trajectory_point_in):
        try:
            output = trajectory_point_in.calc_or_lookup(
                self.morfeus_output,
                kwargs_dict=self.morfeus_args,
            )["morfeus"]

            coords = output["coordinates"]
            charges = output["nuclear_charges"]
            SMILES = output["canon_rdkit_SMILES"]

            rdkit_mol_no_H = Chem.RemoveHs(Chem.MolFromSmiles(SMILES))
            score = sascorer.calculateScore(rdkit_mol_no_H)

            V_synth = self.synth_potential(score)
            if V_synth == None:
                return None

            if self.ensemble:
                if coords.shape[1] == charges.shape[0]:
                    X_test = fml_rep_CM(
                        coords,
                        charges,
                        output["rdkit_Boltzmann"],
                        pad=self.params["max_n"],
                    )

                else:
                    return None
            else:
                if coords.shape[0] == charges.shape[0]:
                    X_test = generate_CM(coords, charges, pad=self.params["max_n"])
                else:
                    return None

        except Exception as e:
            print(e)
            print("Error in 3d conformer sampling")
            return None

        distance = norm(X_test - self.X_init)
        V = self.potential(distance) + V_synth

        if self.verbose:
            print(SMILES, distance, V)

        return V


class potential_MBDF:
    """
    Class to represent a potential using MBDF.
    """

    def __init__(self, X_init, params):
        self.params = params
        self.X_init = X_init
        self.gamma = params["min_d"]
        self.sigma = params["max_d"]
        self.V_0_pot = params["V_0_pot"]
        self.V_0_synth = params["V_0_synth"]
        self.verbose = params["verbose"]
        self.synth_cut_soft = params["synth_cut_soft"]
        self.synth_cut_hard = params["synth_cut_hard"]
        self.ensemble = params["ensemble"]
        self.morfeus_output = {"morfeus": morfeus_coord_info_from_tp}

        if self.ensemble:
            self.morfeus_args = chemspace_sampler_default_params.morfeus_args_confs
        else:
            self.morfeus_args = chemspace_sampler_default_params.morfeus_args_single

        self.potential = self.flat_parabola_potential

    def flat_parabola_potential(self, d):
        """
        A function to describe the potential as a flat parabola.

        Parameters:
        d (float): Distance.

        Returns:
        float: Potential value.
        """

        if d < self.gamma:
            return self.V_0_pot * (d - self.gamma) ** 2
        if self.gamma <= d <= self.sigma:
            return 0
        if d > self.sigma:
            return self.V_0_pot * (d - self.sigma) ** 2

    def synth_potential(self, score):
        if score > self.synth_cut_hard:
            return None
        if score > self.synth_cut_soft:
            return self.V_0_synth * (score - self.synth_cut_soft) ** 2
        else:
            return 0

    def __call__(self, trajectory_point_in):
        try:
            output = trajectory_point_in.calc_or_lookup(
                self.morfeus_output,
                kwargs_dict=self.morfeus_args,
            )["morfeus"]

            coords = output["coordinates"]
            charges = output["nuclear_charges"]
            SMILES = output["canon_rdkit_SMILES"]
            rdkit_mol_no_H = Chem.RemoveHs(Chem.MolFromSmiles(SMILES))
            score = sascorer.calculateScore(rdkit_mol_no_H)

            V_synth = self.synth_potential(score)
            if V_synth == None:
                return None

            if self.ensemble:
                if coords.shape[1] == charges.shape[0]:
                    X_test = fml_rep_MBDF(coords, charges, output["rdkit_Boltzmann"], self.params)
                else:
                    return None
            else:
                if coords.shape[0] == charges.shape[0]:
                    X_test = global_MBDF_bagged_wrapper(charges, coords, self.params)
                else:
                    return None

        except Exception as e:
            print(e)
            print("Error in 3d conformer sampling")
            return None

        distance = norm(X_test - self.X_init)
        V = self.potential(distance) + V_synth

        if self.verbose:
            print(SMILES, distance, V)

        return V


class potential_atomic_energy:
    def __init__(self, X_init, params):
        self.params = params
        self.X_init = X_init
        self.gamma = params["min_d"]
        self.sigma = params["max_d"]
        self.V_0_pot = params["V_0_pot"]
        self.V_0_synth = params["V_0_synth"]
        self.verbose = params["verbose"]
        self.synth_cut_soft = params["synth_cut_soft"]
        self.synth_cut_hard = params["synth_cut_hard"]
        self.ensemble = params["ensemble"]
        self.morfeus_output = {"morfeus": morfeus_coord_info_from_tp}

        self.morfeus_args = chemspace_sampler_default_params.morfeus_args_single

        self.potential = self.flat_parabola_potential

    def flat_parabola_potential(self, d):
        """
        A function to describe the potential as a flat parabola.

        Parameters:
        d (float): Distance.

        Returns:
        float: Potential value.
        """

        if d < self.gamma:
            return self.V_0_pot * (d - self.gamma) ** 2
        if self.gamma <= d <= self.sigma:
            return 0
        if d > self.sigma:
            return self.V_0_pot * (d - self.sigma) ** 2

    def synth_potential(self, score):
        if score > self.synth_cut_hard:
            return None
        if score > self.synth_cut_soft:
            return self.V_0_synth * (score - self.synth_cut_soft) ** 2
        else:
            return 0

    def __call__(self, trajectory_point_in):
        try:
            output = trajectory_point_in.calc_or_lookup(
                self.morfeus_output,
                kwargs_dict=self.morfeus_args,
            )["morfeus"]

            coords = output["coordinates"]
            charges = output["nuclear_charges"]
            SMILES = output["canon_rdkit_SMILES"]
            rdkit_mol_no_H = Chem.RemoveHs(Chem.MolFromSmiles(SMILES))
            score = sascorer.calculateScore(rdkit_mol_no_H)

            V_synth = self.synth_potential(score)
            if V_synth == None:
                return None

            if coords.shape[0] == charges.shape[0]:
                X_test = gen_atomic_energy_rep(
                    charges, coords, self.params["asize2"], rep_dict="atomic_energy"
                )

                if X_test is None:
                    return None

            else:
                return None

        except Exception as e:
            print(e)
            print("Error in 3d conformer sampling")
            return None

        distance = norm(X_test - self.X_init)
        V = self.potential(distance) + V_synth

        if self.verbose:
            print(SMILES, distance, V)

        return V


class potential_BoB_cliffs:
    """
    Class to represent a potential using Bag of Bonds (BoB).
    """

    def __init__(self, X_init, params):
        """
        Initializes the potential_BoB class.

        Parameters:
        X_init (np.array): Initial representations
        gamma (float): A parameter for flat_parabola_potential function.
        sigma (float): A parameter for flat_parabola_potential function.
        possible_elements (list): List of possible atomic elements.
        verbose (bool): Verbosity flag.
        """
        self.params = params
        self.X_init = X_init
        self.gamma = params["min_d"]
        self.sigma = params["max_d"]
        self.V_0_pot = params["V_0_pot"]
        self.V_0_synth = params["V_0_synth"]
        self.possible_elements = params["possible_elements"] + ["H"]
        self.verbose = params["verbose"]
        self.synth_cut_soft = params["synth_cut_soft"]
        self.synth_cut_hard = params["synth_cut_hard"]
        self.ensemble = params["ensemble"]
        self.property = params["property"]
        self.jump = params["jump"]
        self.prop_0 = params["prop_0"]
        self.morfeus_output = {"morfeus": morfeus_coord_info_from_tp}

        if self.ensemble:
            self.morfeus_args = chemspace_sampler_default_params.morfeus_args_confs
        else:
            self.morfeus_args = chemspace_sampler_default_params.morfeus_args_single

        self.potential = self.flat_parabola_potential

    def flat_parabola_potential(self, d):
        """
        A function to describe the potential as a flat parabola.

        Parameters:
        d (float): Distance.

        Returns:
        float: Potential value.
        """

        if d < self.gamma:
            return self.V_0_pot * (d - self.gamma) ** 2
        if self.gamma <= d <= self.sigma:
            return 0
        if d > self.sigma:
            return self.V_0_pot * (d - self.sigma) ** 2

    def synth_potential(self, score):
        if score > self.synth_cut_hard:
            return None
        if score > self.synth_cut_soft:
            return self.V_0_synth * (score - self.synth_cut_soft) ** 2
        else:
            return 0

    def __call__(self, trajectory_point_in):
        """
        Calculates the potential energy of a trajectory point.

        Parameters:
        trajectory_point_in (TrajectoryPoint): Trajectory point.

        Returns:
        float: Potential energy of the trajectory point.
        """

        try:
            output = trajectory_point_in.calc_or_lookup(
                self.morfeus_output,
                kwargs_dict=self.morfeus_args,
            )["morfeus"]

            coords = output["coordinates"]
            charges = output["nuclear_charges"]
            symbols = [str_atom_corr(charge) for charge in charges]
            SMILES = output["canon_rdkit_SMILES"]

            rdkit_mol_no_H = Chem.RemoveHs(Chem.MolFromSmiles(SMILES))
            score = sascorer.calculateScore(rdkit_mol_no_H)
            V_synth = self.synth_potential(score)
            if V_synth == None:
                return None

            if self.property == "gap":
                prop_t = compute_values(SMILES)[2]
            elif self.property == "MolLogP":
                prop_t = Crippen.MolLogP(rdkit_mol_no_H, True)
            else:
                print("Property not implemented")

            if abs(prop_t) > self.jump*abs(self.prop_0):
                pass
            else:
                return 1000

            if self.ensemble:
                if coords.shape[1] == charges.shape[0]:
                    X_test = fml_rep_bob(coords, symbols, output["rdkit_Boltzmann"], self.params)
                else:
                    return None

            else:
                if coords.shape[0] == charges.shape[0]:
                    X_test = generate_bob(symbols, coords, asize=self.params["asize"])

        except Exception as e:
            print(e)
            print("Error in 3d conformer sampling")
            return None

        distance = np.linalg.norm(X_test - self.X_init)
        V = self.potential(distance) + V_synth

        if self.verbose:
            print(SMILES, distance, V, self.prop_0, prop_t)
        return V


class potential_ECFP:

    """
    Sample local chemical space of the inital compound
    with input representation X_init.
    """

    def __init__(self, X_init, params):
        self.X_init = X_init
        self.gamma = params["min_d"]
        self.sigma = params["max_d"]
        self.V_0_pot = params["V_0_pot"]
        self.V_0_synth = params["V_0_synth"]
        self.nBits = params["nBits"]
        self.mmff_check = params["mmff_check"]
        self.synth_cut_soft = params["synth_cut_soft"]
        self.synth_cut_hard = params["synth_cut_hard"]
        self.verbose = params["verbose"]

        self.canonical_rdkit_output = {"canonical_rdkit": trajectory_point_to_canonical_rdkit}

        self.potential = self.flat_parabola_potential

    def flat_parabola_potential(self, d):
        """
        Flat parabola potential. Allows sampling within a distance basin
        interval of I in [gamma, sigma]. The potential is given by:
        """

        if d < self.gamma:
            return self.V_0_pot * (d - self.gamma) ** 2
        if self.gamma <= d <= self.sigma:
            return 0
        if d > self.sigma:
            return self.V_0_pot * (d - self.sigma) ** 2

    def synth_potential(self, score):
        if score > self.synth_cut_hard:
            return None
        if score > self.synth_cut_soft:
            return self.V_0_synth * (score - self.synth_cut_soft) ** 2
        else:
            return 0

    def __call__(self, trajectory_point_in):
        rdkit_mol, canon_SMILES = trajectory_point_in.calc_or_lookup(self.canonical_rdkit_output)[
            "canonical_rdkit"
        ]

        if rdkit_mol is None:
            raise RdKitFailure

        rdkit_mol_no_H = Chem.RemoveHs(rdkit_mol)
        score = sascorer.calculateScore(rdkit_mol_no_H)

        V_synth = self.synth_potential(score)
        if V_synth == None:
            return None

        if self.mmff_check:
            try:
                if not AllChem.MMFFHasAllMoleculeParams(rdkit_mol):
                    return None
            except:
                return None

        X_test = extended_get_single_FP(rdkit_mol, nBits=self.nBits)
        distance = norm(X_test - self.X_init)
        V = self.potential(distance) + V_synth

        if self.verbose:
            print(canon_SMILES, distance, V)

        return V

    def evaluate_point(self, trajectory_point_in):
        """
        Evaluate the function on a list of trajectory points
        """

        rdkit_mol, canon_SMILES = trajectory_point_in.calc_or_lookup(self.canonical_rdkit_output)[
            "canonical_rdkit"
        ]

        X_test = extended_get_single_FP(rdkit_mol, nBits=self.nBits)
        d = norm(X_test - self.X_init)
        V = self.potential(d)

        return V, d

    def evaluate_trajectory(self, trajectory_points):
        """
        Evaluate the function on a list of trajectory points
        """

        values = []
        for trajectory_point in trajectory_points:
            values.append(self.evaluate_point(trajectory_point))

        return np.array(values)


def mc_run(init_egc, min_func, min_func_name, respath, label, params):
    seed = int(str(hash(label))[1:8])
    np.random.seed(1337 + seed)
    random.seed(1337 + seed)

    num_replicas = len(params["betas"])
    init_egcs = [copy.deepcopy(init_egc) for _ in range(num_replicas)]

    bias_coeffs = {"none": None, "weak": 0.2, "stronger": 2.0}
    bias_coeff = bias_coeffs[params["bias_strength"]]
    vbeta_bias_coeff = bias_coeffs[params["bias_strength"]]

    randomized_change_params = {
        "max_fragment_num": 1,
        "nhatoms_range": params["nhatoms_range"],
        "possible_elements": params["possible_elements"],
        "bond_order_changes": [-1, 1],
        "forbidden_bonds": params["forbidden_bonds"],
        "not_protonated": params["not_protonated"],
        "added_bond_orders": [1, 2, 3],
    }
    global_change_params = {
        "num_parallel_tempering_attempts": 128,
        "num_crossover_attempts": 32,
        "prob_dict": {"simple": 0.6, "crossover": 0.2, "tempering": 0.2},
    }

    rw = RandomWalk(
        init_egcs=init_egcs,
        bias_coeff=bias_coeff,
        vbeta_bias_coeff=vbeta_bias_coeff,
        randomized_change_params=randomized_change_params,
        betas=params["betas"],
        min_function=min_func,
        min_function_name=min_func_name,
        keep_histogram=True,
        keep_full_trajectory=False,
        make_restart_frequency=params["make_restart_frequency"],
        soft_exit_check_frequency=params["make_restart_frequency"],
        restart_file=respath + f"/{label}.pkl",
        max_histogram_size=None,
        linear_storage=True,
        greedy_delete_checked_paths=True,
        debug=True,
    )
    for MC_step in range(params["Nsteps"]):
        rw.global_random_change(**global_change_params)

    rw.ordered_trajectory()
    time.sleep(0.1)
    unique_filename = f"{label}_{os.getpid()}.pkl"
    rw.make_restart(tarball=True, restart_file=respath + "/" + unique_filename)


def mc_run_QM9(init_egc, Nsteps, min_func, min_func_name, respath, label):
    bias_strength = "none"
    possible_elements = ["C", "O", "N", "F"]
    not_protonated = None

    forbidden_bonds = [(8, 9), (8, 8), (9, 9), (7, 7)]

    nhatoms_range = [1, 9]
    betas = gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0)
    num_replicas = len(betas)
    init_egcs = [copy.deepcopy(init_egc) for _ in range(num_replicas)]

    make_restart_frequency = None
    num_MC_steps = Nsteps

    bias_coeffs = {"none": None, "weak": 0.2, "stronger": 0.4}
    bias_coeff = bias_coeffs[bias_strength]
    vbeta_bias_coeff = bias_coeffs[bias_strength]

    randomized_change_params = {
        "max_fragment_num": 1,
        "nhatoms_range": nhatoms_range,
        "possible_elements": possible_elements,
        "bond_order_changes": [-1, 1],
        "forbidden_bonds": forbidden_bonds,
        "not_protonated": not_protonated,
        "added_bond_orders": [1, 2, 3],
    }
    global_change_params = {
        "num_parallel_tempering_tries": 128,
        "num_genetic_tries": 32,
        "prob_dict": {"simple": 0.6, "genetic": 0.2, "tempering": 0.2},
    }

    rw = RandomWalk(
        init_egcs=init_egcs,
        bias_coeff=bias_coeff,
        vbeta_bias_coeff=vbeta_bias_coeff,
        randomized_change_params=randomized_change_params,
        betas=betas,
        min_function=min_func,
        min_function_name=min_func_name,
        keep_histogram=True,
        keep_full_trajectory=False,
        make_restart_frequency=make_restart_frequency,
        soft_exit_check_frequency=make_restart_frequency,
        restart_file=respath + f"/{label}.pkl",
        max_histogram_size=None,
        linear_storage=True,
        greedy_delete_checked_paths=True,
        debug=True,
    )
    for MC_step in range(num_MC_steps):
        rw.global_random_change(**global_change_params)

    rw.ordered_trajectory()
    rw.make_restart(tarball=True)


def initialize_fml_from_smiles(smiles, ensemble=True):
    rdkit_H = Chem.AddHs(Chem.MolFromSmiles(smiles))
    smiles_with_H = Chem.MolToSmiles(rdkit_H)
    init_egc = SMILES_to_egc(smiles_with_H)
    trajectory_point = TrajectoryPoint(init_egc)
    morfeus_output = {"morfeus": morfeus_coord_info_from_tp}

    if ensemble:
        morfeus_args = chemspace_sampler_default_params.morfeus_args_confs
    else:
        morfeus_args = chemspace_sampler_default_params.morfeus_args_single

    output = trajectory_point.calc_or_lookup(
        morfeus_output,
        kwargs_dict=morfeus_args,
    )["morfeus"]

    return init_egc, output, rdkit_H


def compute_values(smi, **kwargs):
    quantities = [
        "solvation_energy",
        "HOMO_LUMO_gap",
    ]

    kwargs = {
        "ff_type": "MMFF94",
        "remaining_rho": 0.9,
        "num_conformers": 4,  # 32
        "num_attempts": 2,  # 16
        "solvent": "water",
        "quantities": quantities,
    }

    egc = SMILES_to_egc(smi)
    tp = TrajectoryPoint(egc=egc)
    results = morfeus_FF_xTB_code_quants(tp, **kwargs)

    val = results["mean"]["solvation_energy"]
    std = results["std"]["solvation_energy"]

    val_gap = results["mean"]["HOMO_LUMO_gap"]
    std_gap = results["std"]["HOMO_LUMO_gap"]

    try:
        properties = [float(val), float(std), float(val_gap), float(std_gap)]
    except:
        properties = [np.nan, np.nan, np.nan, np.nan]
    return properties


def compute_values_parallel(SMILES, njobs=10):
    PROPERTIES = [Parallel(n_jobs=njobs)(delayed(compute_values)(smi) for smi in SMILES)]
    return PROPERTIES[0]


def initialize_from_smiles(SMILES, nBits=2048):
    smiles_with_H = Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(SMILES)))
    output = SMILES_to_egc(smiles_with_H)
    rdkit_init = Chem.AddHs(Chem.MolFromSmiles(smiles_with_H))
    X_init = get_all_FP([rdkit_init], nBits=nBits)
    return X_init, rdkit_init, output


class Analyze_Chemspace:
    def __init__(self, path, rep_type="2d", full_traj=False, verbose=False):
        """
        mode : either optimization of dipole and gap = "optimization" or
               sampling locally in chemical space = "sampling"
        """

        self.path = path
        self.rep_type = rep_type
        self.results = glob.glob(path)
        self.verbose = verbose
        self.full_traj = full_traj

    def parse_results(self):
        if self.verbose:
            print("Parsing results...")
            Nsim = len(self.results)
            print("Number of simulations: {}".format(Nsim))

        ALL_HISTOGRAMS = []
        ALL_TRAJECTORIES = []

        if self.full_traj:
            for run in tqdm(self.results, disable=not self.verbose):
                restart_data = loadpkl(run, compress=True)

                HISTOGRAM = self.to_dataframe(restart_data["histogram"])
                HISTOGRAM = HISTOGRAM.sample(frac=1).reset_index(drop=True)
                ALL_HISTOGRAMS.append(HISTOGRAM)
                if self.full_traj:
                    traj = np.array(ordered_trajectory_from_restart(restart_data))
                    CURR_TRAJECTORIES = []
                    for T in range(traj.shape[1]):
                        sel_temp = traj[:, T]
                        TRAJECTORY = self.to_dataframe(sel_temp)
                        CURR_TRAJECTORIES.append(TRAJECTORY)
                    ALL_TRAJECTORIES.append(CURR_TRAJECTORIES)
        else:
            ALL_HISTOGRAMS = Parallel(n_jobs=8)(
                delayed(self.process_run)(run) for run in self.results
            )

        self.ALL_HISTOGRAMS, self.ALL_TRAJECTORIES = ALL_HISTOGRAMS, ALL_TRAJECTORIES
        self.GLOBAL_HISTOGRAM = pd.concat(ALL_HISTOGRAMS)
        self.GLOBAL_HISTOGRAM = self.GLOBAL_HISTOGRAM.drop_duplicates(subset=["SMILES"])

        return self.ALL_HISTOGRAMS, self.GLOBAL_HISTOGRAM, self.ALL_TRAJECTORIES

    def process_run(self, run):
        restart_data = loadpkl(run, compress=True)

        HISTOGRAM = self.to_dataframe(restart_data["histogram"])
        HISTOGRAM = HISTOGRAM.sample(frac=1).reset_index(drop=True)
        return HISTOGRAM

    def convert_from_tps(self, mols):
        """
        Convert the list of trajectory points molecules to SMILES strings.
        tp_list: list of molecudfles as trajectory points
        smiles_mol: list of rdkit molecules
        """

        if self.rep_type == "MolDescriptors":
            SMILES = []
            VALUES = []

            for tp in mols:
                curr_data = tp.calculated_data
                SMILES.append(curr_data["canonical_rdkit"][-1])
                VALUES.append(curr_data["chemspacesampler"])

            VALUES = np.array(VALUES)

            return SMILES, VALUES

        if self.rep_type == "2d":
            SMILES = []
            VALUES = []

            for tp in mols:
                curr_data = tp.calculated_data
                SMILES.append(curr_data["canonical_rdkit"][-1])
                VALUES.append(curr_data["chemspacesampler"])

            VALUES = np.array(VALUES)

            return SMILES, VALUES
        if self.rep_type == "3d":
            SMILES = []
            VALUES = []

            for tp in mols:
                curr_data = tp.calculated_data
                SMILES.append(curr_data["morfeus"]["canon_rdkit_SMILES"])
                VALUES.append(curr_data["chemspacesampler"])

            VALUES = np.array(VALUES)

            return SMILES, VALUES

    def to_dataframe(self, obj):
        """
        Convert the trajectory point object to a dataframe
        and extract xTB values if available.
        """

        df = pd.DataFrame()

        SMILES, VALUES = self.convert_from_tps(obj)
        df["SMILES"] = SMILES
        df["VALUES"] = VALUES
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def process_smiles_to_3d(self, smi, params):
        init_egc, output, curr_rdkit = initialize_fml_from_smiles(smi, ensemble=params["ensemble"])
        return init_egc, output, curr_rdkit

    def reevaluate_3d(self, TP_ALL, params):
        if params["rep_name"] == "BoB":
            X = []
            if params["ensemble"]:
                for TP in TP_ALL:
                    try:
                        X.append(
                            fml_rep_bob(
                                TP["coordinates"],
                                [str_atom_corr(charge) for charge in TP["nuclear_charges"]],
                                TP["rdkit_Boltzmann"],
                                params,
                            )
                        )
                    except:
                        X.append(np.array([np.nan]))

                X = np.array(X)
                return X

            else:
                for TP in TP_ALL:
                    try:
                        X.append(
                            generate_bob(
                                [str_atom_corr(charge) for charge in TP["nuclear_charges"]],
                                TP["coordinates"],
                                asize=params["asize"],
                            )
                        )
                    except:
                        X.append(np.array([np.nan]))

                X = np.array(X)
                return X

        elif params["rep_name"] == "CM":
            X = []
            if params["ensemble"]:
                for TP in TP_ALL:
                    try:
                        X.append(
                            fml_rep_CM(
                                TP["coordinates"],
                                TP["nuclear_charges"],
                                TP["rdkit_Boltzmann"],
                                pad=params["max_n"],
                            )
                        )
                    except:
                        X.append(np.array([np.nan]))

                X = np.array(X)
                return X

            else:
                for TP in TP_ALL:
                    try:
                        X.append(
                            generate_CM(
                                TP["coordinates"],
                                TP["nuclear_charges"],
                                pad=params["max_n"],
                            )
                        )
                    except:
                        X.append(np.array([np.nan]))

                X = np.array(X)
                return X

        elif params["rep_name"] == "SOAP":
            X = []
            if params["ensemble"]:
                for TP in TP_ALL:
                    try:
                        X.append(
                            fml_rep_SOAP(
                                TP["coordinates"],
                                TP["nuclear_charges"],
                                TP["rdkit_Boltzmann"],
                                possible_elements=params["possible_elements"] + ["H"],
                            )
                        )
                    except:
                        X.append(np.array([np.nan]))

                X = np.array(X)
                return X

            else:
                for TP in TP_ALL:
                    try:
                        X.append(
                            gen_soap(
                                TP["coordinates"],
                                TP["nuclear_charges"],
                                species=params["possible_elements"] + ["H"],
                            )
                        )
                    except:
                        X.append(np.array([np.nan]))

                X = np.array(X)
                return X

        elif params["rep_name"] == "MBDF":
            X = []
            if params["ensemble"]:
                for TP in TP_ALL:
                    try:
                        X.append(
                            fml_rep_MBDF(
                                TP["coordinates"],
                                TP["nuclear_charges"],
                                TP["rdkit_Boltzmann"],
                                params,
                            )
                        )
                    except:
                        X.append(np.array([np.nan]))

                X = np.array(X)
                return X

            else:
                for TP in TP_ALL:
                    try:
                        X.append(
                            global_MBDF_bagged_wrapper(
                                TP["nuclear_charges"], TP["coordinates"], params
                            )
                        )
                    except:
                        X.append(np.array([np.nan]))

                X = np.array(X)
                return X

        elif params["rep_name"] == "atomic_energy":
            X = []

            for TP in TP_ALL:
                try:
                    X.append(
                        gen_atomic_energy_rep(
                            TP["nuclear_charges"],
                            TP["coordinates"],
                            params["asize2"],
                            rep_dict="atomic_energy",
                        )
                    )
                except:
                    X.append(np.array([np.nan]))

            X = np.array(X)
            return X

        else:
            print("Representation not implemented")
            return None

    def post_process(self, curr_h, X_I, params):
        if params["strictly_in"]:
            in_interval = curr_h["VALUES"] == 0.0
            SMILES = curr_h["SMILES"][in_interval].values
        else:
            SMILES = curr_h["SMILES"].values

        if len(SMILES) == 0:
            return [], []

        if params["rep_type"] == "3d":
            results = Parallel(n_jobs=params["NPAR"])(
                delayed(self.process_smiles_to_3d)(smi, params) for smi in SMILES
            )
            _, TP_ALL, _ = zip(*results)

            if params["rep_name"] == "SOAP":
                X_ALL = self.reevaluate_3d(TP_ALL, params)

            if params["rep_name"] == "BoB":
                X_ALL = self.reevaluate_3d(TP_ALL, params)

            if params["rep_name"] == "CM":
                X_ALL = self.reevaluate_3d(TP_ALL, params)

            if params["rep_name"] == "MBDF":
                X_ALL = self.reevaluate_3d(TP_ALL, params)

            if params["rep_name"] == "atomic_energy":
                X_ALL = self.reevaluate_3d(TP_ALL, params)

            if params["rep_name"] == "BoB_cliffs":
                if params["ensemble"]:
                    X_ALL = np.asarray(
                        [
                            fml_rep_bob(
                                TP["coordinates"],
                                [str_atom_corr(charge) for charge in TP["nuclear_charges"]],
                                TP["rdkit_Boltzmann"],
                                params,
                            )
                            for TP in TP_ALL
                        ]
                    )
                else:
                    X_ALL =  [
                            generate_bob(
                                [str_atom_corr(charge) for charge in TP["nuclear_charges"]],
                                TP["coordinates"],
                                asize=params["asize"], 
                            )
                            if TP["coordinates"] is not None else np.nan
                            for TP in TP_ALL
                        ]

            D = np.array([norm(X_I - X) for X in X_ALL])
            # pdb.set_trace()
            non_nan_indices = np.where(~np.isnan(D))
            D_filtered = D[non_nan_indices]
            SMILES_filtered = SMILES[non_nan_indices]

            SMILES_filtered = SMILES_filtered[np.argsort(D_filtered)]
            D_filtered = D_filtered[np.argsort(D_filtered)]
            # pdb.set_trace()
            if params["rep_name"] == "BoB_cliffs":
                if params["property"] == "gap":

                    # pdb.set_trace()
                    # compute_values(SMILES[0])
                    p_values = compute_values_parallel(
                        SMILES_filtered, njobs=params["NPAR"]
                    )
                    p_values = np.array(p_values)[:, 2]
                elif params["property"] == "MolLogP":
                    p_values = Parallel(n_jobs=params["NPAR"])(
                        delayed(Crippen.MolLogP)(Chem.MolFromSmiles(smi), True)
                        for smi in SMILES_filtered
                    )
                    p_values = np.array(p_values)
                else:
                    print("Property not implemented")
                    p_values = None
                # p_values = p_values[non_nan_indices]
                # p_values = p_values[np.argsort(D_filtered)]

            if params["rep_name"] != "BoB_cliffs":
                return SMILES_filtered, D_filtered

            else:
                return SMILES_filtered, D_filtered, p_values

        else:
            if params["rep_type"] == "MolDescriptors":
                SMILES = np.array(
                    [Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(smi))) for smi in SMILES]
                )
                explored_rdkit = np.array([Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in SMILES])
                X_ALL = np.array([calc_all_descriptors(rdkit_mol) for rdkit_mol in explored_rdkit])
                D = np.array([norm(X_I - X) for X in X_ALL])
                SMILES = SMILES[np.argsort(D)]
                SMILES = np.array(
                    [Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(smi))) for smi in SMILES]
                )
                D = D[np.argsort(D)]

            if params["rep_type"] == "2d":
                if params["rep_name"] == "inv_ECFP":
                    SMILES = np.array(
                        [Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(smi))) for smi in SMILES]
                    )
                    explored_rdkit = np.array(
                        [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in SMILES]
                    )
                    X_ALL = get_all_FP(explored_rdkit, nBits=params["nBits"])
                    D = np.array([tanimoto_distance(X_I, X) for X in X_ALL])
                    SMILES = SMILES[np.argsort(D)]
                    SMILES = np.array(
                        [
                            Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(smi)))
                            for smi in SMILES
                        ]
                    )
                    D = D[np.argsort(D)]

                elif params["rep_name"] == "inv_props":
                    from mosaics.minimized_functions import inversion_potentials

                    SMILES = np.array(
                        [Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(smi))) for smi in SMILES]
                    )
                    explored_rdkit = np.array(
                        [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in SMILES]
                    )
                    X_ALL = np.array(
                        [
                            inversion_potentials.compute_molecule_properties(rdkit_mol)
                            for rdkit_mol in explored_rdkit
                        ]
                    )
                    # get_all_FP(explored_rdkit, nBits=params["nBits"])
                    D = np.array([norm(X_I - X) for X in X_ALL])
                    # np.array([tanimoto_distance(X_I, X) for X in X_ALL])
                    SMILES = SMILES[np.argsort(D)]
                    SMILES = np.array(
                        [
                            Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(smi)))
                            for smi in SMILES
                        ]
                    )
                    D = D[np.argsort(D)]

                elif params["rep_name"] == "ECFP":
                    SMILES = np.array(
                        [Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(smi))) for smi in SMILES]
                    )
                    explored_rdkit = np.array(
                        [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in SMILES]
                    )
                    X_ALL = get_all_FP(explored_rdkit, nBits=params["nBits"])
                    D = np.array([norm(X_I - X) for X in X_ALL])
                    SMILES = SMILES[np.argsort(D)]
                    SMILES = np.array(
                        [
                            Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(smi)))
                            for smi in SMILES
                        ]
                    )
                    D = D[np.argsort(D)]

                elif params["rep_name"] == "inv_latent":
                    model_transformer, scalar_features = (
                        params["model"],
                        params["scalar_features"],
                    )
                    SMILES = np.array(
                        [Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(smi))) for smi in SMILES]
                    )
                    explored_rdkit = np.array(
                        [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in SMILES]
                    )
                    X_ALL = scalar_features.transform(
                        get_all_FP(explored_rdkit, nBits=params["nBits"])
                    )

                    T_ALL = model_transformer.transform(X_ALL)
                    # np.array([model_transformer.transform( x.reshape(1,-1) ) for x in X_ALL])
                    # Parallel(n_jobs=8)(delayed(model_transformer.transform)(x.reshape(-1,1)) for x in X_ALL)
                    #

                    D = np.array([norm(X_I.flatten() - T) for T in T_ALL])
                    SMILES = SMILES[np.argsort(D)]
                    SMILES = np.array(
                        [
                            Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(smi)))
                            for smi in SMILES
                        ]
                    )
                    D = D[np.argsort(D)]

                else:
                    print("Representation not implemented")
                    return None

            return SMILES, D


def chemspacesampler_ECFP(smiles, params=None):
    """
    Run the chemspacesampler with ECFP fingerprints.
    """
    X, rdkit_init, egc = initialize_from_smiles(smiles)

    if params is None:
        num_heavy_atoms = rdkit_init.GetNumHeavyAtoms()
        elements = list({atom.GetSymbol() for atom in rdkit_init.GetAtoms()})
        params = {
            "min_d": 0.0,
            "max_d": 6.0,
            "NPAR": 1,
            "Nsteps": 100,
            "bias_strength": "none",
            "possible_elements": elements,
            "not_protonated": None,
            "forbidden_bonds": [(8, 9), (8, 8), (9, 9), (7, 7)],
            "nhatoms_range": [num_heavy_atoms, num_heavy_atoms],
            "betas": gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
            "make_restart_frequency": None,
            "rep_type": "2d",
            "nBits": 2048,
            "mmff_check": True,
            "synth_cut_soft": 3,
            "synth_cut_hard": 4,
            "verbose": False,
        }

    min_func = potential_ECFP(X, params=params)

    respath = tempfile.mkdtemp()
    if params["NPAR"] == 1:
        mc_run(egc, min_func, "chemspacesampler", respath, f"results_{0}", params)
    else:
        Parallel(n_jobs=params["NPAR"])(
            delayed(mc_run)(egc, min_func, "chemspacesampler", respath, f"results_{i}", params)
            for i in range(params["NPAR"])
        )
    ana = Analyze_Chemspace(respath + f"/*.pkl", rep_type="2d", full_traj=False, verbose=False)
    _, GLOBAL_HISTOGRAM, _ = ana.parse_results()
    MOLS, D = ana.post_process(GLOBAL_HISTOGRAM, X, params)
    shutil.rmtree(respath)

    return MOLS, D


def chemspacesampler_MolDescriptors(smiles, params=None):
    """
    Run the chemspacesampler with MolDescriptors.
    """

    _, rdkit_init, egc = initialize_from_smiles(smiles)
    X = calc_all_descriptors(rdkit_init)

    if params is None:
        params = {
            "min_d": 0.0,
            "max_d": 4.0,
            "NPAR": 4,
            "Nsteps": 100,
            "bias_strength": "none",
            "possible_elements": ["C", "O", "N", "F"],
            "not_protonated": None,
            "forbidden_bonds": [(8, 9), (8, 8), (9, 9), (7, 7)],
            "nhatoms_range": [6, 6],
            "betas": gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
            "make_restart_frequency": None,
            "rep_type": "MolDescriptors",
            "synth_cut_soft": 3,
            "synth_cut_hard": 4,
            "mmff_check": True,
            "verbose": False,
        }

    min_func = potential_MolDescriptors(X, params)

    respath = tempfile.mkdtemp()
    if params["NPAR"] == 1:
        mc_run(egc, min_func, "chemspacesampler", respath, f"results_{0}", params)
    else:
        Parallel(n_jobs=params["NPAR"])(
            delayed(mc_run)(egc, min_func, "chemspacesampler", respath, f"results_{i}", params)
            for i in range(params["NPAR"])
        )
    ana = Analyze_Chemspace(
        respath + f"/*.pkl", rep_type="MolDescriptors", full_traj=False, verbose=False
    )
    _, GLOBAL_HISTOGRAM, _ = ana.parse_results()
    MOLS, D = ana.post_process(GLOBAL_HISTOGRAM, X, params)
    shutil.rmtree(respath)

    return MOLS, D


def chemspacesampler_SOAP(smiles, params=None):
    """
    Runs the chemspacesampler with SOAP Ensemble representations.

    Parameters:
    smiles (str): SMILES string of the molecule.
    params (dict, optional): Parameters for SOAP representation. If None, default parameters will be used.

    Returns:
    tuple: Number of structures (N) and molecular SMILES (MOLS).
    """

    if params is None:
        num_heavy_atoms = rdkit_init.GetNumHeavyAtoms()
        elements = list({atom.GetSymbol() for atom in rdkit_init.GetAtoms()})

        params = {
            "min_d": 0.0,
            "max_d": 150.0,
            "V_0_pot": 0.05,
            "V_0_synth": 0.05,
            "NPAR": 1,
            "Nsteps": 100,
            "bias_strength": "none",
            "possible_elements": elements,
            "not_protonated": None,
            "forbidden_bonds": [(8, 9), (8, 8), (9, 9), (7, 7)],
            "nhatoms_range": [num_heavy_atoms, num_heavy_atoms],
            "betas": gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
            "make_restart_frequency": None,
            "rep_type": "3d",
            "rep_name": "SOAP",
            "synth_cut_soft": 3,
            "synth_cut_hard": 4,
            "ensemble": True,
            "verbose": False,
        }

    init_egc, tp, rdkit_init = initialize_fml_from_smiles(smiles, ensemble=params["ensemble"])
    coords, charges = tp["coordinates"], tp["nuclear_charges"]

    if params["ensemble"]:
        X = fml_rep_SOAP(
            coords, charges, tp["rdkit_Boltzmann"], params["possible_elements"] + ["H"]
        )
    else:
        X = gen_soap(coords, charges, species=params["possible_elements"] + ["H"])

    min_func = potential_SOAP(X, params)
    respath = tempfile.mkdtemp()
    if params["NPAR"] > 1:
        Parallel(n_jobs=params["NPAR"])(
            delayed(mc_run)(
                init_egc, min_func, "chemspacesampler", respath, f"results_{i}", params
            )
            for i in range(params["NPAR"])
        )
    else:
        mc_run(init_egc, min_func, "chemspacesampler", respath, f"results_{0}", params)
    ana = Analyze_Chemspace(respath + f"/*.pkl", rep_type="3d", full_traj=False, verbose=False)
    _, GLOBAL_HISTOGRAM, _ = ana.parse_results()
    MOLS, D = ana.post_process(GLOBAL_HISTOGRAM, X, params)
    shutil.rmtree(respath)

    return MOLS, D


def chemspacesampler_BoB(smiles, params=None):
    if params is None:
        num_heavy_atoms = rdkit_init.GetNumHeavyAtoms()
        elements = list({atom.GetSymbol() for atom in rdkit_init.GetAtoms()})

        params = {
            "min_d": 0.0,
            "max_d": 150.0,
            "NPAR": 1,
            "Nsteps": 100,
            "bias_strength": "none",
            "possible_elements": elements,
            "not_protonated": None,
            "forbidden_bonds": [(8, 9), (8, 8), (9, 9), (7, 7)],
            "nhatoms_range": [num_heavy_atoms, num_heavy_atoms],
            "betas": gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
            "make_restart_frequency": None,
            "rep_type": "3d",
            "rep_name": "BoB",
            "synth_cut_soft": 3,
            "synth_cut_hard": 5,
            "ensemble": True,
            "verbose": False,
        }
    init_egc, tp, rdkit_init = initialize_fml_from_smiles(smiles, ensemble=params["ensemble"])
    coords, charges = tp["coordinates"], tp["nuclear_charges"]
    symbols = [str_atom_corr(charge) for charge in charges]

    asize, max_n = max_element_counts([symbols * 3])
    asize_copy = asize.copy()
    elements_to_exclude = ["H"]
    elements_not_excluded = [key for key in asize_copy if key not in elements_to_exclude]

    if elements_not_excluded:  # checks if list is not empty
        avg_value = sum(asize_copy[key] for key in elements_not_excluded) / len(
            elements_not_excluded
        )
    else:
        avg_value = 0  # or any other value you consider appropriate when there are no elements

    for element in params["possible_elements"]:
        if element not in asize:
            asize[element] = 2 * int(avg_value)

    params["asize"], params["max_n"], params["unique_elements"] = (
        asize,
        3 * max_n,
        list(asize.keys()),
    )
    if params["ensemble"]:
        X = fml_rep_bob(coords, symbols, tp["rdkit_Boltzmann"], params)
    else:
        X = generate_bob(symbols, coords, asize=params["asize"])

    min_func = potential_BoB(X, params)
    respath = tempfile.mkdtemp()

    if params["NPAR"] > 1:
        Parallel(n_jobs=params["NPAR"])(
            delayed(mc_run)(
                init_egc, min_func, "chemspacesampler", respath, f"results_{i}", params
            )
            for i in range(params["NPAR"])
        )
    else:
        mc_run(init_egc, min_func, "chemspacesampler", respath, f"results_{0}", params)
    ana = Analyze_Chemspace(respath + f"/*.pkl", rep_type="3d", full_traj=False, verbose=False)
    _, GLOBAL_HISTOGRAM, _ = ana.parse_results()
    MOLS, D = ana.post_process(GLOBAL_HISTOGRAM, X, params)
    shutil.rmtree(respath)

    return MOLS, D


def chemspacesampler_CM(smiles, params=None):
    init_egc, tp, rdkit_init = initialize_fml_from_smiles(smiles, ensemble=params["ensemble"])

    if params is None:
        num_heavy_atoms = rdkit_init.GetNumHeavyAtoms()
        elements = list({atom.GetSymbol() for atom in rdkit_init.GetAtoms()})

        params = {
            "min_d": 0.0,
            "max_d": 150.0,
            "NPAR": 1,
            "Nsteps": 100,
            "bias_strength": "none",
            "possible_elements": elements,
            "not_protonated": None,
            "forbidden_bonds": [(8, 9), (8, 8), (9, 9), (7, 7)],
            "nhatoms_range": [num_heavy_atoms, num_heavy_atoms],
            "betas": gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
            "make_restart_frequency": None,
            "rep_type": "3d",
            "rep_name": "CM",
            "synth_cut_soft": 3,
            "synth_cut_hard": 5,
            "ensemble": True,
            "verbose": False,
        }

    coords, charges = tp["coordinates"], tp["nuclear_charges"]
    symbols = [str_atom_corr(charge) for charge in charges]

    _, max_n = max_element_counts([symbols * 3])
    params["max_n"] = 5 * max_n

    if params["ensemble"]:
        X = fml_rep_CM(coords, charges, tp["rdkit_Boltzmann"], pad=params["max_n"])
    else:
        X = generate_CM(coords, charges, pad=params["max_n"])

    min_func = potential_CM(X, params)
    respath = tempfile.mkdtemp()

    if params["NPAR"] > 1:
        Parallel(n_jobs=params["NPAR"])(
            delayed(mc_run)(
                init_egc, min_func, "chemspacesampler", respath, f"results_{i}", params
            )
            for i in range(params["NPAR"])
        )
    else:
        mc_run(init_egc, min_func, "chemspacesampler", respath, f"results_{0}", params)

    ana = Analyze_Chemspace(respath + f"/*.pkl", rep_type="3d", full_traj=False, verbose=False)
    _, GLOBAL_HISTOGRAM, _ = ana.parse_results()
    MOLS, D = ana.post_process(GLOBAL_HISTOGRAM, X, params)
    shutil.rmtree(respath)

    return MOLS, D


def chemspacesampler_atomization_rep(smiles, params=None):
    init_egc, tp, rdkit_init = initialize_fml_from_smiles(smiles, ensemble=params["ensemble"])
    coords, charges = tp["coordinates"], tp["nuclear_charges"]
    symbols = [str_atom_corr(charge) for charge in charges]

    asize, max_n = max_element_counts([symbols * 3])
    asize_copy = asize.copy()
    elements_to_exclude = ["H"]
    elements_not_excluded = [key for key in asize_copy if key not in elements_to_exclude]

    if elements_not_excluded:  # checks if list is not empty
        avg_value = sum(asize_copy[key] for key in elements_not_excluded) / len(
            elements_not_excluded
        )
    else:
        avg_value = 0  # or any other value you consider appropriate when there are no elements

    for element in params["possible_elements"]:
        if element not in asize:
            asize[element] = 2 * int(avg_value)

    params["asize"], params["max_n"], params["unique_elements"] = (
        asize,
        3 * max_n,
        list(asize.keys()),
    )
    params["asize2"] = {NUCLEAR_CHARGE[k]: v for k, v in zip(asize.keys(), asize.values())}

    if params["ensemble"]:
        print("Not implemented")
        return None
    else:
        X = gen_atomic_energy_rep(charges, coords, params["asize2"], "atomic_energy")

    min_func = potential_atomic_energy(X, params)
    respath = tempfile.mkdtemp()

    if params["NPAR"] > 1:
        Parallel(n_jobs=params["NPAR"])(
            delayed(mc_run)(
                init_egc, min_func, "chemspacesampler", respath, f"results_{i}", params
            )
            for i in range(params["NPAR"])
        )
    else:
        mc_run(init_egc, min_func, "chemspacesampler", respath, f"results_{0}", params)

    ana = Analyze_Chemspace(respath + f"/*.pkl", rep_type="3d", full_traj=False, verbose=False)
    _, GLOBAL_HISTOGRAM, _ = ana.parse_results()
    MOLS, D = ana.post_process(GLOBAL_HISTOGRAM, X, params)
    shutil.rmtree(respath)

    return MOLS, D


def chemspacesampler_MBDF(smiles, params=None):
    if params is None:
        params = {
            "min_d": 0.0,
            "max_d": 120.0,
            "strictly_in": True,
            "V_0_pot": 0.05,
            "V_0_synth": 0.05,
            "NPAR": 1,
            "Nsteps": 5,
            "bias_strength": "none",
            "possible_elements": ["C", "O", "N", "F", "Si"],
            "not_protonated": None,
            "forbidden_bonds": [(8, 9), (8, 8), (9, 9), (7, 7)],
            "nhatoms_range": [1, 16],
            "betas": gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
            "make_restart_frequency": None,
            "rep_type": "3d",
            "rep_name": "MBDF",
            "synth_cut_soft": 7,
            "synth_cut_hard": 9,
            "ensemble": True,
            "verbose": True,
        }
    init_egc, tp, rdkit_init = initialize_fml_from_smiles(smiles, ensemble=params["ensemble"])
    coords, charges = tp["coordinates"], tp["nuclear_charges"]
    symbols = [str_atom_corr(charge) for charge in charges]

    asize, max_n = max_element_counts([symbols * 3])
    asize_copy = asize.copy()
    elements_to_exclude = ["H"]
    elements_not_excluded = [key for key in asize_copy if key not in elements_to_exclude]

    if elements_not_excluded:  # checks if list is not empty
        avg_value = sum(asize_copy[key] for key in elements_not_excluded) / len(
            elements_not_excluded
        )
    else:
        avg_value = 0  # or any other value you consider appropriate when there are no elements

    for element in params["possible_elements"]:
        if element not in asize:
            asize[element] = 2 * int(avg_value)

    params["asize"], params["max_n"], params["unique_elements"] = (
        asize,
        3 * max_n,
        list(asize.keys()),
    )
    params["asize2"] = {NUCLEAR_CHARGE[k]: v for k, v in zip(asize.keys(), asize.values())}
    params["grid1"], params["grid2"] = fourier_grid()

    if params["ensemble"]:
        X = fml_rep_MBDF(coords, charges, tp["rdkit_Boltzmann"], params)
    else:
        X = global_MBDF_bagged_wrapper(charges, coords, params)

    min_func = potential_MBDF(X, params)
    respath = tempfile.mkdtemp()
    if params["NPAR"] > 1:
        Parallel(n_jobs=params["NPAR"])(
            delayed(mc_run)(
                init_egc, min_func, "chemspacesampler", respath, f"results_{i}", params
            )
            for i in range(params["NPAR"])
        )
    else:
        mc_run(init_egc, min_func, "chemspacesampler", respath, f"results_{0}", params)

    ana = Analyze_Chemspace(respath + f"/*.pkl", rep_type="3d", full_traj=False, verbose=False)
    _, GLOBAL_HISTOGRAM, _ = ana.parse_results()

    MOLS, D = ana.post_process(GLOBAL_HISTOGRAM, X, params)
    shutil.rmtree(respath)

    return MOLS, D


def chemspacesampler_find_cliffs(smiles, params=None):
    init_egc, tp, rdkit_init = initialize_fml_from_smiles(smiles, ensemble=params["ensemble"])
    if params is None:
        num_heavy_atoms = rdkit_init.GetNumHeavyAtoms()
        elements = list({atom.GetSymbol() for atom in rdkit_init.GetAtoms()})

        params = {
            "min_d": 0.0,
            "max_d": 150.0,
            "V_0_pot": 0.05,
            "V_0_synth": 0.05,
            "NPAR": 1,
            "Nsteps": 100,
            "bias_strength": "none",
            "possible_elements": elements,
            "not_protonated": None,
            "forbidden_bonds": [(8, 9), (8, 8), (9, 9), (7, 7)],
            "nhatoms_range": [num_heavy_atoms, num_heavy_atoms],
            "betas": gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
            "make_restart_frequencV = self.potential(distance) + V_synthy": None,
            "rep_type": "3d",
            "rep_name": "BoB_cliffs",
            "synth_cut_soft": 3,
            "synth_cut_hard": 5,
            "ensemble": False,
            "property": "gap",
            "jump": 0.2,
            "verbose": True,
        }

    if params["property"] == "gap":
        prop_0 = compute_values(smiles)[2]
    elif params["property"] == "MolLogP":
        prop_0 = Crippen.MolLogP(Chem.MolFromSmiles(smiles), True)
    else:
        print("Property not implemented")

    params["prop_0"] = prop_0
    coords, charges = tp["coordinates"], tp["nuclear_charges"]
    symbols = [str_atom_corr(charge) for charge in charges]

    asize, max_n = max_element_counts([symbols * 3])
    asize_copy = asize.copy()
    elements_to_exclude = ["H"]
    elements_not_excluded = [key for key in asize_copy if key not in elements_to_exclude]

    if elements_not_excluded:
        avg_value = sum(asize_copy[key] for key in elements_not_excluded) / len(
            elements_not_excluded
        )
    else:
        avg_value = 0

    for element in params["possible_elements"]:
        if element not in asize:
            asize[element] = 2 * int(avg_value)

    params["asize"], params["max_n"], params["unique_elements"] = (
        asize,
        3 * max_n,
        list(asize.keys()),
    )

    if params["ensemble"]:
        X = fml_rep_bob(coords, symbols, tp["rdkit_Boltzmann"], params)
    else:
        X = generate_bob(symbols, coords, asize=params["asize"])

    min_func = potential_BoB_cliffs(X, params)

    respath = tempfile.mkdtemp()
    if params["NPAR"] > 1:
        Parallel(n_jobs=params["NPAR"])(
            delayed(mc_run)(
                init_egc, min_func, "chemspacesampler", respath, f"results_{i}", params
            )
            for i in range(params["NPAR"])
        )
    else:
        mc_run(init_egc, min_func, "chemspacesampler", respath, f"results_{0}", params)

    print("now analyzing results...")

    ana = Analyze_Chemspace(respath + f"/*.pkl", rep_type="3d", full_traj=False, verbose=False)
    _, GLOBAL_HISTOGRAM, _ = ana.parse_results()

    MOLS, D, P = ana.post_process(GLOBAL_HISTOGRAM, X, params)
    shutil.rmtree(respath)

    return MOLS, D, P, prop_0
