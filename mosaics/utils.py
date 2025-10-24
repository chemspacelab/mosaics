import bz2
import pickle
import subprocess

import numpy as np
from igraph import Graph
from sortedcontainers import SortedList

from .chem_graph import ChemGraph
from .chem_graph.heavy_atom import HeavyAtom
from .ext_graph_compound import ExtGraphCompound
from .misc_procedures import int_atom, str_atom_corr


# Routines for xyz import/export.
def check_byte(byte_or_str):
    if isinstance(byte_or_str, str):
        return byte_or_str
    else:
        return byte_or_str.decode("utf-8")


def checked_input_readlines(file_input):
    try:
        lines = [check_byte(l) for l in file_input.readlines()]
    except AttributeError:
        with open(file_input, "r") as input_file:
            lines = input_file.readlines()
    return lines


def write_compound_to_xyz_file(compound, xyz_file_name):
    write_xyz_file(compound.coordinates, xyz_file_name, elements=compound.atomtypes)


def xyz_string(coordinates, elements=None, nuclear_charges=None, extra_string=""):
    """
    Create an xyz-formatted string from coordinates and elements or nuclear charges.
    coordinates : coordinate array
    elements : string array of element symbols
    nuclear_charges : integer array; used to generate element list if elements is set to None
    """
    if elements is None:
        elements = [str_atom_corr(charge) for charge in nuclear_charges]
    output = str(len(coordinates)) + "\n" + extra_string
    for atom_coords, element in zip(coordinates, elements):
        output += "\n" + element + " " + " ".join([str(atom_coord) for atom_coord in atom_coords])
    return output


def write_xyz_file(
    coordinates, xyz_file_name, elements=None, nuclear_charges=None, extra_string=""
):
    xyz_file = open(xyz_file_name, "w")
    xyz_file.write(
        xyz_string(
            coordinates,
            elements=elements,
            nuclear_charges=nuclear_charges,
            extra_string=extra_string,
        )
    )
    xyz_file.close()


def read_xyz_file(xyz_input, additional_attributes=["charge"]):
    lines = checked_input_readlines(xyz_input)

    return read_xyz_lines(lines, additional_attributes=additional_attributes)


def read_xyz_lines(xyz_lines, additional_attributes=["charge"]):
    add_attr_dict = {}
    for add_attr in additional_attributes:
        add_attr_dict = {add_attr: None, **add_attr_dict}

    num_atoms = int(xyz_lines[0])
    xyz_coordinates = np.zeros((num_atoms, 3))
    nuclear_charges = np.zeros((num_atoms,), dtype=int)
    atomic_symbols = []

    lsplit = xyz_lines[1].split()
    for l in lsplit:
        for add_attr in additional_attributes:
            add_attr_eq = add_attr + "="
            if add_attr_eq == l[: len(add_attr_eq)]:
                add_attr_dict[add_attr] = int(l.split("=")[1])

    for atom_id, atom_line in enumerate(xyz_lines[2 : num_atoms + 2]):
        lsplit = atom_line.split()
        atomic_symbol = lsplit[0]
        atomic_symbols.append(atomic_symbol)
        nuclear_charges[atom_id] = int_atom(atomic_symbol)
        for i in range(3):
            xyz_coordinates[atom_id, i] = float(lsplit[i + 1])

    return nuclear_charges, atomic_symbols, xyz_coordinates, add_attr_dict


# Break an egc of a disconnected molecules into several egc's corresponding to each fragment.
def break_into_connected(egc: ExtGraphCompound):
    output = []
    gc = egc.chemgraph.graph.components()
    mvec = np.array(gc.membership)
    sgcs = gc.subgraphs()

    if egc.distances is None:
        new_dist_mat = None
    if egc.coordinates is None:
        new_coords = None

    adjmat = egc.true_adjmat()

    hids = [[] for _ in range(len(sgcs))]
    for h_id in range(egc.num_heavy_atoms(), egc.num_atoms()):
        for ha_id in range(egc.num_heavy_atoms()):
            if adjmat[h_id, ha_id] == 1:
                hids[mvec[ha_id]].append(h_id)
                break

    for sgc_id, _ in enumerate(sgcs):
        new_members = np.where(mvec == sgc_id)[0]
        if len(hids[sgc_id]) != 0:
            new_members = np.append(new_members, hids[sgc_id])
        if egc.distances is not None:
            new_dist_mat = np.copy(egc.distances[new_members, :][:, new_members])
        if egc.coordinates is not None:
            new_coords = np.copy(egc.coordinates[new_members, :])
        new_nuclear_charges = np.copy(egc.true_ncharges()[new_members])
        new_adjacency_matrix = np.copy(adjmat[new_members, :][:, new_members])
        output.append(
            ExtGraphCompound(
                adjacency_matrix=new_adjacency_matrix,
                distances=new_dist_mat,
                nuclear_charges=new_nuclear_charges,
                coordinates=new_coords,
            )
        )
    return output


def generate_unrepeated_database(egc_list):
    """
    Create an unrepeated list of ExtGraphCompound objects.
    """
    output = SortedList()
    for egc in egc_list:
        if egc not in output:
            output.add(egc)
    return output


# TO-DO: should be fixed if I get to using xbgf again.
def xbgf2gc(xbgf_file):
    xbgf_input = open(xbgf_file, "r")
    graph = Graph()
    while True:
        line = xbgf_input.readline()
        if not line:
            break
        lsplit = line.split()
        if (lsplit[0] == "REMARK") and (lsplit[1] == "NATOM"):
            natom = int(lsplit[2])
            nuclear_charges = np.empty((natom,))
            graph.add_vertices(natom)
        if lsplit[0] == "ATOM":
            atom_id = int(lsplit[1]) - 1
            nuclear_charges[atom_id] = int(lsplit[15])
        if lsplit[0] == "CONECT":
            if len(lsplit) > 2:
                atom_id = int(lsplit[1]) - 1
                for con_id_str in lsplit[2:]:
                    con_id = int(con_id_str) - 1
                    graph.add_edges([(atom_id, con_id)])
    xbgf_input.close()
    return ExtGraphCompound(graph=graph, nuclear_charges=nuclear_charges)


#   Some procedures that often appear in scripts.


def egc2xyz_string(egc: ExtGraphCompound, extra_string=""):
    return xyz_string(
        egc.coordinates, nuclear_charges=egc.nuclear_charges, extra_string=extra_string
    )


def write_egc2xyz(egc: ExtGraphCompound, xyz_file_name, extra_string=""):
    write_xyz_file(
        egc.coordinates,
        xyz_file_name,
        nuclear_charges=egc.nuclear_charges,
        extra_string=extra_string,
    )


# Related to pkl files.

compress_fileopener = {True: bz2.BZ2File, False: open}
pkl_compress_ending = {True: ".pkl.bz2", False: ".pkl"}


def dump2pkl(obj, filename: str, compress: bool = False):
    """
    Dump an object to a pickle file.
    obj : object to be saved
    filename : name of the output file
    compress : whether bz2 library is used for compressing the file.
    """
    output_file = compress_fileopener[compress](filename, "wb")
    pickle.dump(obj, output_file)
    output_file.close()


def loadpkl(filename: str, compress: bool = False):
    """
    Load an object from a pickle file.
    filename : name of the imported file
    compress : whether bz2 compression was used in creating the loaded file.
    """
    input_file = compress_fileopener[compress](filename, "rb")
    obj = pickle.load(input_file)
    input_file.close()
    return obj


def ispklfile(filename: str):
    """
    Check whether filename is a pickle file.
    """
    return filename[-4:] == ".pkl"


# Directory management.


def mktmp(directory=False):
    extra_args = ()
    if directory:
        extra_args = ("-d", *extra_args)
    return subprocess.check_output(["mktemp", *extra_args, "-p", "."], text=True).rstrip("\n")


def mktmpdir():
    return mktmp(True)


def run(*cmd):
    subprocess.run(list(cmd))


def mkdir(dirname):
    run("mkdir", "-p", dirname)


def rmdir(dirname):
    run("rm", "-Rf", dirname)


# For using comparison lists for vector searches (e.g. in a Milvus client).
def calc_max_comparison_list_length(max_nhatoms, max_valence=6):
    """
    Maximum length of a graph comparison list (w. color_defining_neighborhood_radius=0) obtained for ChemGraphs with up to nhatoms heavy atoms that have a maximum of max_valence valence.
    """
    # 1 entry optionally corresponds to charge
    max_comparison_list_length = 1
    # for each heavy atom write its nuclear charge, number of hydrogens, and number of neighbors listed after the corresponding entry
    max_comparison_list_length += 3 * max_nhatoms
    # total length of neighbor lists.
    max_comparison_list_length += max_nhatoms * max_valence // 2
    return max_comparison_list_length


def padded_comparison_list(
    cg: ChemGraph, max_comparison_list_length=None, max_nhatoms=None, max_valence=6
):
    """
    Generated padded representation vector for ChemGraph.
    """
    if max_comparison_list_length is None:
        assert max_nhatoms is not None
        max_comparison_list_length = calc_max_comparison_list_length(
            max_nhatoms, max_valence=max_valence
        )
    padded_comp_list = np.zeros(max_comparison_list_length, dtype=int)
    comp_list = cg.get_comparison_list()
    padded_comp_list[: len(comp_list)] = comp_list[:]
    return padded_comp_list


def comparison_list_to_chemgraph(comp_list):
    """
    Recover a ChemGraph from a comparison list.
    """
    nhatoms = comp_list[0]
    hatoms = []
    graph = Graph(n=nhatoms, directed=False)
    current_entry_id = 1
    for hatom_id in range(nhatoms):
        ncharge = comp_list[current_entry_id]
        current_entry_id += 1
        nhydrogens = comp_list[current_entry_id]
        hatoms.append(HeavyAtom(ncharge, nhydrogens))
        current_entry_id += 1
        num_neighbors = comp_list[current_entry_id]
        for _ in range(num_neighbors):
            current_entry_id += 1
            neighbor_internal_id = comp_list[current_entry_id]
            graph.add_edge(hatom_id, neighbor_internal_id)
        current_entry_id += 1
    charge = comp_list[current_entry_id]
    return ChemGraph(graph=graph, hatoms=hatoms, charge=charge)
