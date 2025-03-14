import os
import sys

import numpy as np
from rdkit.Chem import RDConfig

from mosaics.beta_choice import gen_exp_beta_array
from mosaics.minimized_functions import chemspace_potentials
from mosaics.rdkit_utils import RdKitFailure, SMILES_to_egc

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import copy
import shutil
import tempfile

import numpy as np
import sascorer
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def compute_molecule_properties(mol):
    # List of RDKit descriptor functions to use
    descriptor_functions = [func for name, func in Descriptors.descList]

    # Calculate each descriptor
    properties = [func(mol) for func in descriptor_functions]

    # If there are more custom properties, compute and add them to the properties list
    # ...

    # Convert to a numpy array for consistency
    properties_vector = np.array(properties)

    return properties_vector


class potential_inv_ECFP:
    def __init__(self, X_target, params):
        self.X_target = X_target
        self.V_0_pot = params["V_0_pot"]
        self.sigma = params["max_d"]
        self.nBits = params["nBits"]
        self.pot_type = params["pot_type"]
        self.verbose = params["verbose"]

        self.canonical_rdkit_output = {
            "canonical_rdkit": chemspace_potentials.trajectory_point_to_canonical_rdkit
        }

        if self.pot_type == "flat_parabola":
            self.potential = self.flat_parabola_potential
        elif self.pot_type == "parabola":
            self.potential = self.parabola_potential

    def parabola_potential(self, d):
        return self.V_0_pot * d**2

    def flat_parabola_potential(self, d):
        """
        A function to describe the potential as a flat parabola.

        Parameters:
        d (float): Distance.

        Returns:
        float: Potential value.
        """

        if d <= self.sigma:
            return 0
        if d > self.sigma:
            return self.V_0_pot * (d - self.sigma) ** 2

    def __call__(self, trajectory_point_in):
        rdkit_mol, canon_SMILES = trajectory_point_in.calc_or_lookup(self.canonical_rdkit_output)[
            "canonical_rdkit"
        ]

        if rdkit_mol is None:
            raise RdKitFailure

        X_test = chemspace_potentials.extended_get_single_FP(rdkit_mol, nBits=self.nBits)
        distance = chemspace_potentials.tanimoto_distance(X_test, self.X_target)
        # distance =  np.linalg.norm(compute_molecule_properties(rdkit_mol) - self.X_target)
        V = self.potential(distance)

        if self.verbose:
            print(canon_SMILES, distance, V)

        return V


def chemspacesampler_inv_ECFP(smiles_init, X_target, params=None):
    X, _, egc_0 = chemspace_potentials.initialize_from_smiles(smiles_init, nBits=params["nBits"])
    if params is None:
        params = {
            "V_0_pot": 0.05,
            "max_d": 6.0,
            "NPAR": 4,
            "Nsteps": 100,
            "bias_strength": "none",
            "possible_elements": ["C", "O", "N", "F"],
            "not_protonated": None,
            "forbidden_bonds": None,
            "nhatoms_range": [6, 6],
            "betas": gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
            "make_restart_frequency": None,
            "rep_type": "2d",
            "nBits": 2048,
            "rep_name": "ECFP",
            "strategy": "default",
            "verbose": False,
        }

    if params["strategy"] == "default":
        min_func = potential_inv_ECFP(X_target, params=params)
        respath = tempfile.mkdtemp()
        if params["NPAR"] == 1:
            chemspace_potentials.mc_run(
                egc_0, min_func, "chemspacesampler", respath, f"results_{0}", params
            )
        else:
            Parallel(n_jobs=params["NPAR"])(
                delayed(chemspace_potentials.mc_run)(
                    egc_0, min_func, "chemspacesampler", respath, f"results_{i}", params
                )
                for i in range(params["NPAR"])
            )
        ana = chemspace_potentials.Analyze_Chemspace(
            respath + f"/*.pkl", rep_type="2d", full_traj=False, verbose=False
        )
        _, GLOBAL_HISTOGRAM, _ = ana.parse_results()
        MOLS, D = ana.post_process(GLOBAL_HISTOGRAM, X_target, params)
        shutil.rmtree(respath)

        return MOLS, D

    elif params["strategy"] == "modify_pot":
        d_init = chemspace_potentials.tanimoto_distance(X, X_target)
        egc_best, d_best, V_0_best = egc_0, d_init, params["V_0_pot"]
        series = [1 for i in range(params["Nparts"])]
        N_budget = [int(round(s / sum(series) * params["Nsteps"])) for s in series]
        N_budget = np.array(N_budget)
        for n in N_budget:
            params["V_0_pot"] = V_0_best
            params["Nsteps"] = n
            min_func = potential_inv_ECFP(X_target, params=params)
            respath = tempfile.mkdtemp()
            if params["NPAR"] == 1:
                chemspace_potentials.mc_run(
                    egc_best,
                    min_func,
                    "chemspacesampler",
                    respath,
                    f"results_{0}",
                    params,
                )
            else:
                Parallel(n_jobs=params["NPAR"])(
                    delayed(chemspace_potentials.mc_run)(
                        egc_best,
                        min_func,
                        "chemspacesampler",
                        respath,
                        f"results_{i}",
                        params,
                    )
                    for i in range(params["NPAR"])
                )
            ana = chemspace_potentials.Analyze_Chemspace(
                respath + f"/*.pkl", rep_type="2d", full_traj=False, verbose=False
            )
            _, GLOBAL_HISTOGRAM, _ = ana.parse_results()
            MOLS, D = ana.post_process(GLOBAL_HISTOGRAM, X_target, params)
            shutil.rmtree(respath)
            if D[0] < d_best:
                d_best, mol_best, V_0_best = D[0], MOLS[0], params["V_0_pot"]
                print(mol_best, d_best, V_0_best)
                egc_best = SMILES_to_egc(mol_best)
                if d_best <= params["d_threshold"]:
                    print("Found molecule within threshold")
                    return MOLS, D
            else:
                params["V_0_pot"] = 2 * V_0_best
                min_func = potential_inv_ECFP(X_target, params=params)
                respath = tempfile.mkdtemp()
                if params["NPAR"] == 1:
                    chemspace_potentials.mc_run(
                        egc_best,
                        min_func,
                        "chemspacesampler",
                        respath,
                        f"results_{0}",
                        params,
                    )
                else:
                    Parallel(n_jobs=params["NPAR"])(
                        delayed(chemspace_potentials.mc_run)(
                            egc_best,
                            min_func,
                            "chemspacesampler",
                            respath,
                            f"results_{i}",
                            params,
                        )
                        for i in range(params["NPAR"])
                    )
                ana = chemspace_potentials.Analyze_Chemspace(
                    respath + f"/*.pkl", rep_type="2d", full_traj=False, verbose=False
                )
                _, GLOBAL_HISTOGRAM, _ = ana.parse_results()
                MOLS, D = ana.post_process(GLOBAL_HISTOGRAM, X_target, params)
                shutil.rmtree(respath)

                if D[0] < d_best:
                    d_best, mol_best, V_0_best = D[0], MOLS[0], params["V_0_pot"]
                    print(mol_best, d_best, V_0_best)
                    egc_best = SMILES_to_egc(mol_best)
                    if d_best <= params["d_threshold"]:
                        print("Found molecule within threshold")
                        return MOLS, D
                else:
                    params["V_0_pot"] = V_0_best / 2
                    min_func = potential_inv_ECFP(X_target, params=params)
                    respath = tempfile.mkdtemp()
                    if params["NPAR"] == 1:
                        chemspace_potentials.mc_run(
                            egc_best,
                            min_func,
                            "chemspacesampler",
                            respath,
                            f"results_{0}",
                            params,
                        )
                    else:
                        Parallel(n_jobs=params["NPAR"])(
                            delayed(chemspace_potentials.mc_run)(
                                egc_best,
                                min_func,
                                "chemspacesampler",
                                respath,
                                f"results_{i}",
                                params,
                            )
                            for i in range(params["NPAR"])
                        )
                    ana = chemspace_potentials.Analyze_Chemspace(
                        respath + f"/*.pkl",
                        rep_type="2d",
                        full_traj=False,
                        verbose=False,
                    )
                    _, GLOBAL_HISTOGRAM, _ = ana.parse_results()
                    MOLS, D = ana.post_process(GLOBAL_HISTOGRAM, X_target, params)
                    shutil.rmtree(respath)

                    if D[0] < d_best:
                        d_best, mol_best, V_0_best = D[0], MOLS[0], params["V_0_pot"]
                        print(mol_best, d_best, V_0_best)
                        egc_best = SMILES_to_egc(mol_best)
                        if d_best <= params["d_threshold"]:
                            print("Found molecule within threshold")
                            return MOLS, D

        return MOLS, D

    elif params["strategy"] == "contract":
        d_init = chemspace_potentials.tanimoto_distance(X, X_target)
        d_arr = np.linspace(d_init, params["max_d"], params["Nparts"])
        # Generate exponential growth series
        series = [params["growth_factor"] ** i for i in range(params["Nparts"])]

        # Normalize series to make sum equals Nsteps
        N_budget = [int(round(s / sum(series) * params["Nsteps"])) for s in series]

        # Correct possible rounding errors
        N_budget[-1] += params["Nsteps"] - sum(N_budget)
        N_budget = np.array(N_budget)[::-1]

        egc_rep = [copy.deepcopy(egc_0) for _ in range(params["NPAR"])]
        for d, n in zip(d_arr, N_budget):
            params["max_d"] = d
            params["Nsteps"] = n

            min_func = potential_inv_ECFP(X_target, params=params)
            respath = tempfile.mkdtemp()
            if params["NPAR"] == 1:
                chemspace_potentials.mc_run(
                    egc_rep[0],
                    min_func,
                    "chemspacesampler",
                    respath,
                    f"results_{0}",
                    params,
                )
            else:
                Parallel(n_jobs=params["NPAR"])(
                    delayed(chemspace_potentials.mc_run)(
                        egc_rep[i],
                        min_func,
                        "chemspacesampler",
                        respath,
                        f"results_{i}",
                        params,
                    )
                    for i in range(params["NPAR"])
                )
            ana = chemspace_potentials.Analyze_Chemspace(
                respath + f"/*.pkl", rep_type="2d", full_traj=False, verbose=False
            )
            _, GLOBAL_HISTOGRAM, _ = ana.parse_results()
            MOLS, D = ana.post_process(GLOBAL_HISTOGRAM, X_target, params)
            shutil.rmtree(respath)
            if len(MOLS) > 0 and D[0] < d_init:
                if len(MOLS) > params["NPAR"]:
                    MOLS_CHOICE = MOLS[: params["NPAR"]].tolist()
                else:
                    diff = params["NPAR"] - len(MOLS)
                    MOLS_CHOICE = MOLS.tolist() + [
                        MOLS[i % len(MOLS)].tolist() for i in range(diff)
                    ]
                egc_rep = []
                for mol in MOLS_CHOICE:
                    try:
                        egc_rep.append(SMILES_to_egc(mol))
                    except:
                        egc_rep.append(egc_0)

                smiles_closest, clostest_distance = MOLS[0], D[0]
                print(clostest_distance, smiles_closest)
                if clostest_distance <= params["d_threshold"]:
                    print("Found molecule within threshold")
                    return MOLS, D

        return MOLS, D

    elif params["strategy"] == "modify_pot":
        pass
    else:
        print("Strategy not implemented")


class potential_inv_latent:
    def __init__(self, T_init, T_goal, params=None):
        self.T_init = T_init
        self.T_goal = T_goal
        self.V_0_pot = params["V_0_pot"]
        self.sigma = params["max_d"]
        self.nBits = params["nBits"]
        self.pot_type = params["pot_type"]
        self.verbose = params["verbose"]

        self.model = params["model"]
        self.scalar_features = params["scalar_features"]
        self.scalar_values = params["scalar_values"]

        self.mmff_check = params["mmff_check"]
        self.V_0_synth = params["V_0_synth"]
        self.synth_cut_soft = params["synth_cut_soft"]
        self.synth_cut_hard = params["synth_cut_hard"]

        self.canonical_rdkit_output = {
            "canonical_rdkit": chemspace_potentials.trajectory_point_to_canonical_rdkit
        }

        self.potential = self.parabola_potential

    def parabola_potential(self, d):
        return self.V_0_pot * d**2

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

        V_synth = self.synth_potential(score)
        if V_synth == None:
            return None

        if self.mmff_check:
            try:
                if not AllChem.MMFFHasAllMoleculeParams(rdkit_mol):
                    return None
            except:
                return None

        X_test = chemspace_potentials.extended_get_single_FP(rdkit_mol, nBits=self.nBits).reshape(
            1, -1
        )
        X_test = self.scalar_features.transform(X_test)
        T_test = self.model.transform(X_test)[0]
        distance = np.linalg.norm(T_test - self.T_goal)
        V = self.potential(distance) + V_synth  # + (1/(2+np.linalg.norm(T_test - self.T_init)))

        if self.verbose:
            print(canon_SMILES, distance, score, V)

        return V


def chemspacesampler_inv_latent(SMILES_init, T_init, T_goal, params=None):
    _, _, egc_0 = chemspace_potentials.initialize_from_smiles(SMILES_init, nBits=params["nBits"])
    min_func = potential_inv_latent(T_init, T_goal, params=params)

    respath = tempfile.mkdtemp()
    if params["NPAR"] == 1:
        chemspace_potentials.mc_run(
            egc_0, min_func, "chemspacesampler", respath, f"results_{0}", params
        )
    else:
        Parallel(n_jobs=params["NPAR"])(
            delayed(chemspace_potentials.mc_run)(
                egc_0, min_func, "chemspacesampler", respath, f"results_{i}", params
            )
            for i in range(params["NPAR"])
        )

    ana = chemspace_potentials.Analyze_Chemspace(
        respath + f"/*.pkl", rep_type="2d", full_traj=False, verbose=False
    )

    _, GLOBAL_HISTOGRAM, _ = ana.parse_results()
    MOLS, D = ana.post_process(GLOBAL_HISTOGRAM, T_goal, params)
    shutil.rmtree(respath)

    return MOLS, D
