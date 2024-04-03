import copy

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from xyz2mol import AC2mol, chiral_stereo_check, xyz2AC

from .ext_graph_compound import ExtGraphCompound
from .misc_procedures import (
    VERBOSITY,
    VERBOSITY_MUTED,
    default_num_procs,
    default_parallel_backend,
    int_atom_checked,
)
from .rdkit_utils import (
    FFInconsistent,
    chemgraph_to_canonical_rdkit,
    chemgraph_to_canonical_rdkit_wcoords_no_check,
)
from .utils import read_xyz_file
from .valence_treatment import ChemGraph, InvalidAdjMat


def xyz2mol_graph(nuclear_charges, charge, coords, get_chirality=False):
    #    try:
    adj_matrix, mol = xyz2AC(nuclear_charges, coords, charge)
    # TODO: K.Karan.: used to be like this to account for all exceptions RdKit would through; realized later it's bad practice.
    # Should we start working with XYZ's again we should uncomment this and add exceptions as we come across them.
    #    except:
    #        raise InvalidAdjMat
    if get_chirality is False:
        return adj_matrix, nuclear_charges, coords
    else:
        chiral_mol = AC2mol(
            mol,
            adj_matrix,
            nuclear_charges,
            charge,
            allow_charged_fragments=True,
            use_graph=True,
        )
        chiral_stereo_check(chiral_mol)
        chiral_centers = Chem.FindMolChiralCenters(chiral_mol)
        return adj_matrix, nuclear_charges, coords, chiral_centers


def xyz_list2mols_extgraph(xyz_file_list, leave_nones=False, xyz_to_add_data=False):
    read_xyzs = [read_xyz_file(xyz_file) for xyz_file in xyz_file_list]
    unfiltered_list = Parallel(n_jobs=default_num_procs(), backend=default_parallel_backend)(
        delayed(xyz2mol_extgraph)(None, read_xyz=read_xyz) for read_xyz in read_xyzs
    )
    output = []
    for egc_id, (egc, xyz_name) in enumerate(zip(unfiltered_list, xyz_file_list)):
        if egc is None:
            if VERBOSITY != VERBOSITY_MUTED:
                print(
                    "WARNING, failed to create EGC for id",
                    egc_id,
                    "xyz name:",
                    xyz_name,
                )
        else:
            if xyz_to_add_data:
                egc.additional_data["xyz"] = xyz_name
        if (egc is not None) or leave_nones:
            output.append(egc)
    return output


def chemgraph_from_ncharges_coords(nuclear_charges, coordinates, charge=0):
    # Convert numpy array to lists as that is the correct input for xyz2mol_graph function.
    converted_ncharges = [int(ncharge) for ncharge in nuclear_charges]
    converted_coordinates = [
        [float(atom_coord) for atom_coord in atom_coords] for atom_coords in coordinates
    ]
    adj_matrix, ncharges, _ = xyz2mol_graph(converted_ncharges, charge, converted_coordinates)
    return ChemGraph(adj_mat=adj_matrix, nuclear_charges=ncharges)


def egc_from_ncharges_coords(nuclear_charges, coordinates, charge=0):
    """
    Generate ExtGraphCompound from coordinates.
    """
    # xyz2mol_graph only accepts lists, not NumPy arrays.
    converted_ncharges = [int(int_atom_checked(ncharge)) for ncharge in nuclear_charges]
    converted_coordinates = [
        [float(atom_coord) for atom_coord in atom_coords] for atom_coords in coordinates
    ]
    bond_order_matrix, _, _ = xyz2mol_graph(converted_ncharges, charge, converted_coordinates)
    return ExtGraphCompound(
        adjacency_matrix=bond_order_matrix,
        nuclear_charges=np.array(converted_ncharges),
        coordinates=coordinates,
    )


def xyz2mol_extgraph(filepath, get_chirality=False, read_xyz=None):
    if read_xyz is None:
        read_xyz = read_xyz_file(filepath)

    nuclear_charges = read_xyz[0]
    coordinates = read_xyz[2]
    add_attr_dict = read_xyz[3]

    charge = None
    if "charge" in add_attr_dict:
        charge = add_attr_dict["charge"]
    if charge is None:
        charge = 0
    try:
        return egc_from_ncharges_coords(nuclear_charges, coordinates, charge=charge)
    except InvalidAdjMat:
        return None


def all_egc_from_tar(tarfile_name):
    import tarfile

    tar_input = tarfile.open(tarfile_name, "r")
    output = []
    for tar_member in tar_input.getmembers():
        extracted_member = tar_input.extractfile(tar_member)
        if extracted_member is not None:
            output.append(xyz2mol_extgraph(extracted_member))
    tar_input.close()
    return output


# Using RdKit to generate ChemGraph and other objects with coordinates, while checking with xyz2mol that the coordinates make sense.
def chemgraph_to_canonical_rdkit_wcoords(cg, **kwargs):
    """
    Creates an rdkit Molecule object whose heavy atoms are canonically ordered.
    cg : ChemGraph input chemgraph object
    ff_type : which forcefield to use; currently MMFF and UFF are available
    num_attempts : how many times the optimization is attempted
    output : RDKit molecule, indices of the heavy atoms, indices of heavy atoms to which a given hydrogen is connected,
    SMILES generated from the canonical RDKit molecules, and the RDKit's coordinates
    """
    (mol, canon_SMILES, rdkit_coords) = chemgraph_to_canonical_rdkit_wcoords_no_check(cg, **kwargs)

    rdkit_nuclear_charges = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    # Additionally check that the coordinates actually correspond to the molecule.
    try:
        coord_based_cg = chemgraph_from_ncharges_coords(rdkit_nuclear_charges, rdkit_coords)
    except InvalidAdjMat:
        raise FFInconsistent
    if coord_based_cg != cg:
        raise FFInconsistent

    return mol, canon_SMILES, rdkit_coords


def egc_with_coords(egc, coords=None, **kwargs):
    """
    Create a copy of an ExtGraphCompound object with coordinates. If coordinates are set to None they are generated with RDKit.
    egc : ExtGraphCompound input object
    coords : None or np.array
    ff_type : str type of force field used; MMFF and UFF are available
    num_attempts : int number of
    """
    output = copy.deepcopy(egc)
    if coords is None:
        (
            _,
            canon_SMILES,
            coords,
        ) = chemgraph_to_canonical_rdkit_wcoords(egc.chemgraph, **kwargs)
    else:
        (
            _,
            canon_SMILES,
        ) = chemgraph_to_canonical_rdkit(egc.chemgraph, **kwargs)

    output.additional_data["canon_rdkit_SMILES"] = canon_SMILES
    output.add_canon_rdkit_coords(coords)
    return output


def coord_info_from_tp(tp, **kwargs):
    """
    Coordinates corresponding to a TrajectoryPoint object
    tp : TrajectoryPoint object
    num_attempts : number of attempts taken to generate MMFF coordinates (introduced because for QM9 there is a ~10% probability that the coordinate generator won't converge)
    **kwargs : keyword arguments for the egc_with_coords procedure
    """
    output = {"coordinates": None, "canon_rdkit_SMILES": None, "nuclear_charges": None}
    try:
        egc_wc = egc_with_coords(tp.egc, **kwargs)
        output["coordinates"] = egc_wc.coordinates
        output["canon_rdkit_SMILES"] = egc_wc.additional_data["canon_rdkit_SMILES"]
        output["nuclear_charges"] = egc_wc.true_ncharges()
    except FFInconsistent:
        pass

    return output
