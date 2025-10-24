import copy

import numpy as np

# Morfeus - used for coordinate generation.
from morfeus.conformer import ConformerEnsemble
from rdkit import RDLogger
from xtb.interface import Calculator

# xtb code - used to get HOMO-LUMO gap, solvation energy, and dipole for Morfeus-generated conformers.
from xtb.libxtb import VERBOSITY_FULL, VERBOSITY_MINIMAL, VERBOSITY_MUTED
from xtb.utils import get_method, get_solvent

from ..data import conversion_coefficient, room_T
from ..misc_procedures import (
    InvalidAdjMat,
    all_None_dict,
    any_element_in_list,
    checked_environ_val,
    repeated_dict,
    weighted_array,
)
from ..rdkit_utils import chemgraph_to_canonical_rdkit
from ..xyz2graph import chemgraph_from_ncharges_coords

RDLogger.DisableLog("rdApp.*")


def morfeus_coord_info_from_tp(
    tp,
    num_attempts=1,
    ff_type="MMFF94",
    return_rdkit_obj=False,
    all_confs=False,
    temperature=room_T,
    coord_validity_check=True,
    **dummy_kwargs,
):
    """
    Coordinates generated for a TrajectoryPoint object using Morfeus.
    tp : TrajectoryPoint object
    num_attempts : number of attempts taken to generate MMFF coordinates (introduced because for QM9 there is a ~10% probability that the coordinate generator won't converge)
    **kwargs : keyword arguments for the egc_with_coords procedure
    """
    output = {"coordinates": None, "nuclear_charges": None, "canon_rdkit_SMILES": None}
    cg = tp.egc.chemgraph
    canon_rdkit_mol, canon_rdkit_SMILES = chemgraph_to_canonical_rdkit(cg)
    output["canon_rdkit_SMILES"] = canon_rdkit_SMILES
    if return_rdkit_obj:
        output["canon_rdkit_mol"] = canon_rdkit_mol
    # TODO: better place to check OMP_NUM_THREADS value?
    try:
        conformers = ConformerEnsemble.from_rdkit(
            canon_rdkit_mol,
            n_confs=num_attempts,
            optimize=ff_type,
            n_threads=checked_environ_val("MORFEUS_NUM_THREADS", default_answer=1),
        )
    except Exception as ex:
        if not isinstance(ex, ValueError):
            print("#PROBLEMATIC_MORFEUS:", tp)
        return output
    if all_confs:
        conformers.prune_rmsd()
    all_coordinates = conformers.get_coordinates()
    energies = conformers.get_energies()

    nuclear_charges = np.array(conformers.elements)
    output["nuclear_charges"] = nuclear_charges

    if len(energies) == 0:
        return output

    min_en_id = np.argmin(energies)

    min_en = energies[min_en_id]
    min_coordinates = all_coordinates[min_en_id]

    if coord_validity_check:
        try:
            coord_based_cg = chemgraph_from_ncharges_coords(nuclear_charges, min_coordinates)
        except InvalidAdjMat:
            return output
        if coord_based_cg != cg:
            return output

    if all_confs:
        output["coordinates"] = all_coordinates
        output["rdkit_energy"] = energies
        output["rdkit_degeneracy"] = conformers.get_degeneracies()
        output["rdkit_Boltzmann"] = conformers.boltzmann_weights(temperature=temperature)
    else:
        output["coordinates"] = all_coordinates[min_en_id]
        output["rdkit_energy"] = min_en
    return output


verbosity_dict = {
    "MUTED": VERBOSITY_MUTED,
    "MINIMAL": VERBOSITY_MINIMAL,
    "FULL": VERBOSITY_FULL,
}


def xTB_singlepoint_res_no_error_check(
    coordinates,
    nuclear_charges,
    accuracy=None,
    verbosity="MUTED",
    parametrization="gfn2-xtb",
    solvent=None,
    max_iterations=None,
    electronic_temperature=None,
):
    calc_obj = Calculator(
        get_method(parametrization),
        nuclear_charges,
        coordinates * conversion_coefficient["Angstrom_Bohr"],
    )
    if accuracy is not None:
        calc_obj.set_accuracy(accuracy)
    if max_iterations is not None:
        calc_obj.set_max_iterations(max_iterations)
    if electronic_temperature is not None:
        calc_obj.set_electronic_temperature(electronic_temperature)
    if solvent is not None:
        calc_obj.set_solvent(get_solvent(solvent))
    calc_obj.set_verbosity(verbosity_dict[verbosity])

    return calc_obj.singlepoint()


def xTB_singlepoint_res(*args, **kwargs):
    try:
        return xTB_singlepoint_res_no_error_check(*args, **kwargs)
    except Exception as ex:
        print("PROBLEM:", ex)
        return None


def xTB_quants(
    coordinates, nuclear_charges, quantities=[], solvent=None, **other_xTB_singlepoint_res
):
    res = xTB_singlepoint_res(
        coordinates, nuclear_charges, solvent=solvent, **other_xTB_singlepoint_res
    )

    if res is None:
        return None

    output = {}
    output["energy"] = res.get_energy()

    if any_element_in_list(quantities, "HOMO_energy", "LUMO_energy", "HOMO_LUMO_gap"):
        # Orbital energies are in Hartree
        orbital_energies = res.get_orbital_eigenvalues()
        orbital_occupations = res.get_orbital_occupations()
        HOMO_energy = max(orbital_energies[np.where(orbital_occupations > 0.1)])
        LUMO_energy = min(orbital_energies[np.where(orbital_occupations < 0.1)])
        output["HOMO_energy"] = HOMO_energy
        output["LUMO_energy"] = LUMO_energy
        output["HOMO_LUMO_gap"] = LUMO_energy - HOMO_energy

    if "dipole" in quantities:
        dipole_vec = res.get_dipole()
        # Dipole is in e*Bohr (a.u.)
        output["dipole"] = np.sqrt(np.sum(dipole_vec**2))

    if any_element_in_list(quantities, "solvation_energy", "energy_no_solvent"):
        res_nosolvent = xTB_singlepoint_res(
            coordinates, nuclear_charges, **other_xTB_singlepoint_res, solvent=None
        )
        if res_nosolvent is None:
            return None
        # energy is in Hartree
        en_nosolvent = res_nosolvent.get_energy()
        output["energy_no_solvent"] = en_nosolvent
        output["solvation_energy"] = output["energy"] - en_nosolvent

    return output


class weighted_index:
    def __init__(self, entry_id, w):
        self.entry_id = entry_id
        self.rho = w


def cut_weights(bweights, remaining_rho=None):
    weights = [None for _ in range(len(bweights))]
    if remaining_rho is None:
        min_en_id = np.argmax(bweights)
        weights[min_en_id] = 1.0
    else:
        bweights_list = weighted_array(
            [weighted_index(w_id, w) for w_id, w in enumerate(bweights)]
        )
        bweights_list.normalize_sort_rhos()
        bweights_list.cutoff_minor_weights(remaining_rho=remaining_rho)
        for wi in bweights_list:
            weights[wi.entry_id] = wi.rho
    return weights


num_eval_name = "num_evals"


def morfeus_FF_xTB_code_quants_weighted(
    tp,
    num_conformers=1,
    ff_type="MMFF94",
    quantities=[],
    remaining_rho=None,
    coord_validity_check=True,
    **xTB_quants_kwargs,
):
    coord_info = morfeus_coord_info_from_tp(
        tp,
        num_attempts=num_conformers,
        ff_type=ff_type,
        all_confs=True,
        coord_validity_check=coord_validity_check,
    )
    coordinates = coord_info["coordinates"]
    if coordinates is None:
        return None
    conf_weights = cut_weights(coord_info["rdkit_Boltzmann"], remaining_rho=remaining_rho)
    ncharges = coord_info["nuclear_charges"]
    output = repeated_dict(quantities, 0.0)
    for conf_coords, weight in zip(coord_info["coordinates"], conf_weights):
        if weight is not None:
            res = xTB_quants(conf_coords, ncharges, quantities=quantities, **xTB_quants_kwargs)
            if res is None:
                print("###PROBLEMATIC MOLECULE for xTB", coord_info["canon_rdkit_SMILES"])
                return None
            for quant in quantities:
                if quant == num_eval_name:
                    output[quant] += 1.0
                else:
                    output[quant] += weight * res[quant]
    return output


def morfeus_FF_xTB_code_quants(
    tp,
    num_conformers=1,
    num_attempts=1,
    ff_type="MMFF94",
    quantities=[],
    remaining_rho=None,
    **xTB_quants_kwargs,
):
    """
    Use morfeus-ml FF coordinates with Grimme lab's xTB code to calculate some quantities.
    """
    quant_arrs = repeated_dict(quantities, np.zeros((num_attempts,)), copy_needed=True)

    for i in range(num_attempts):
        av_res = morfeus_FF_xTB_code_quants_weighted(
            tp,
            num_conformers=num_conformers,
            ff_type=ff_type,
            quantities=quantities,
            remaining_rho=remaining_rho,
            **xTB_quants_kwargs,
        )
        if av_res is None:
            quant_arrs = all_None_dict(quantities)
            break
        for quant in quantities:
            quant_arrs[quant][i] = av_res[quant]

    output = {"arrs": quant_arrs}
    output["mean"] = {}
    output["std"] = {}
    for quant, quant_arr in quant_arrs.items():
        if quant_arr is None:
            output["mean"][quant] = None
            output["std"][quant] = None
        else:
            try:
                mean = np.mean(quant_arr)
            except FloatingPointError:
                mean = 0.0
            output["mean"][quant] = mean
            try:
                stddev = np.std(quant_arr)
            except FloatingPointError:
                stddev = 0.0
            output["std"][quant] = stddev
    return output


class LinComb_Morfeus_xTB_code:
    """
    Calculate linear combination of several quantities obtained from xTB code with morfeus-ml coordinates.
    """

    def __init__(
        self,
        quantities=[],
        coefficients=[],
        constr_quants=[],
        cq_upper_bounds=None,
        cq_lower_bounds=None,
        add_mult_funcs=None,
        add_mult_func_powers=None,
        xTB_res_dict_name="xTB_res",
        **xTB_related_kwargs,
    ):
        self.quantities = quantities
        self.coefficients = coefficients

        self.constr_quants = constr_quants
        self.cq_upper_bounds = cq_upper_bounds
        self.cq_lower_bounds = cq_lower_bounds

        self.needed_quantities = copy.copy(self.quantities)
        for quant in constr_quants:
            if quant not in self.needed_quantities:
                self.needed_quantities.append(quant)

        if add_mult_funcs is None:
            self.add_mult_funcs = [None for _ in range(len(quantities))]
        else:
            self.add_mult_funcs = add_mult_funcs
        if add_mult_func_powers is None:
            self.add_mult_func_powers = [1 for _ in range(len(quantities))]
        else:
            self.add_mult_func_powers = add_mult_func_powers
        self.xTB_res_dict_name = xTB_res_dict_name
        self.xTB_related_kwargs = xTB_related_kwargs
        self.call_counter = 0

    def __call__(self, trajectory_point_in):
        self.call_counter += 1

        xTB_res_dict = trajectory_point_in.calc_or_lookup(
            {self.xTB_res_dict_name: morfeus_FF_xTB_code_quants},
            kwargs_dict={
                self.xTB_res_dict_name: {
                    **self.xTB_related_kwargs,
                    "quantities": self.needed_quantities,
                }
            },
        )[self.xTB_res_dict_name]
        result = 0.0
        for quant_id, quant in enumerate(self.constr_quants):
            cur_val = xTB_res_dict["mean"][quant]
            if cur_val is None:
                return None
            if self.cq_upper_bounds is not None:
                if (self.cq_upper_bounds[quant_id] is not None) and (
                    cur_val > self.cq_upper_bounds[quant_id]
                ):
                    return None
            if self.cq_lower_bounds is not None:
                if (self.cq_lower_bounds[quant_id] is not None) and (
                    cur_val < self.cq_lower_bounds[quant_id]
                ):
                    return None
        for quant, coeff, add_mult, add_mult_power in zip(
            self.quantities,
            self.coefficients,
            self.add_mult_funcs,
            self.add_mult_func_powers,
        ):
            cur_val = xTB_res_dict["mean"][quant]
            if cur_val is None:
                return None
            cur_add = cur_val * coeff
            if add_mult is not None:
                cur_add *= add_mult(trajectory_point_in) ** add_mult_power
            result += cur_add
        return result

    def __str__(self):
        output = (
            "LinComb_Morfeus_xTB_code\nQuants: "
            + str(self.quantities)
            + "\nCoefficients: "
            + str(self.coefficients)
            + "\n"
        )
        if self.constr_quants:
            output += "Constraints:\n"
            for quant_id, quant_name in enumerate(self.constr_quants):
                output += quant_name + " "
                if self.cq_lower_bounds is not None:
                    output += str(self.cq_lower_bounds[quant_id])
                output += ":"
                if self.cq_upper_bounds is not None:
                    output += str(self.cq_upper_bounds[quant_id])
                output += "\n"
        output += "Calculation parameters:\n"
        for k, v in self.xTB_related_kwargs.items():
            output += k + " " + str(v) + "\n"
        return output

    def __repr__(self):
        return str(self)
