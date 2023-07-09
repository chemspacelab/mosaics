from xyz2mol import (
    xyz2AC,
    chiral_stereo_check,
    AC2mol,
)
from .misc_procedures import (
    int_atom_checked,
    default_num_procs,
    default_parallel_backend,
)
from rdkit import Chem
from .valence_treatment import InvalidAdjMat, ChemGraph
from joblib import delayed, Parallel
from .utils import read_xyz_file
from .ext_graph_compound import ExtGraphCompound
import numpy as np


def xyz2mol_graph(nuclear_charges, charge, coords, get_chirality=False):
    try:
        adj_matrix, mol = xyz2AC(nuclear_charges, coords, charge)
    except:
        raise InvalidAdjMat
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
    unfiltered_list = Parallel(
        n_jobs=default_num_procs(), backend=default_parallel_backend
    )(delayed(xyz2mol_extgraph)(None, read_xyz=read_xyz) for read_xyz in read_xyzs)
    output = []
    for egc_id, (egc, xyz_name) in enumerate(zip(unfiltered_list, xyz_file_list)):
        if egc is None:
            print("WARNING, failed to create EGC for id", egc_id, "xyz name:", xyz_name)
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
    adj_matrix, ncharges, _ = xyz2mol_graph(
        converted_ncharges, charge, converted_coordinates
    )
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
    bond_order_matrix, _, _ = xyz2mol_graph(
        converted_ncharges, charge, converted_coordinates
    )
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
