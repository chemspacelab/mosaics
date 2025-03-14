import copy

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToSmiles
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from .ext_graph_compound import ExtGraphCompound
from .misc_procedures import int_atom_checked
from .test_utils import print_to_separate_file_wprefix
from .valence_treatment import (
    ChemGraph,
    canonically_permuted_ChemGraph,
    default_valence,
    split_chemgraph_into_connected_fragments,
    str2ChemGraph,
)


class RdKitFailure(Exception):
    pass


#   For going between rdkit and egc objects.
def rdkit_to_egc(rdkit_mol, egc_hydrogen_autofill=False):
    nuclear_charges = [atom.GetAtomicNum() for atom in rdkit_mol.GetAtoms()]
    adjacency_matrix = GetAdjacencyMatrix(rdkit_mol)

    try:
        coordinates = rdkit_mol.GetConformer().GetPositions()
    except ValueError:
        coordinates = None
    return ExtGraphCompound(
        adjacency_matrix=adjacency_matrix,
        nuclear_charges=nuclear_charges,
        coordinates=coordinates,
        hydrogen_autofill=egc_hydrogen_autofill,
    )


#   For converting SMILES to egc.
def SMILES_to_egc(smiles_string, egc_hydrogen_autofill=False):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        raise RdKitFailure
    # We can fill the hydrogens either at this stage or at EGC creation stage;
    # introduced when I had problems with rdKIT.
    mol = Chem.AddHs(mol, explicitOnly=egc_hydrogen_autofill)
    egc_out = rdkit_to_egc(mol, egc_hydrogen_autofill=egc_hydrogen_autofill)
    egc_out.additional_data["SMILES"] = smiles_string
    return egc_out


def SMILES_list_to_egc(smiles_list):
    return [SMILES_to_egc(smiles) for smiles in smiles_list]


#   For converting InChI to egc.
def InChI_to_egc(InChI_string, egc_hydrogen_autofill=False):
    mol = Chem.inchi.MolFromInchi(InChI_string, removeHs=False)
    if mol is None:
        raise RdKitFailure
    mol = Chem.AddHs(mol, explicitOnly=egc_hydrogen_autofill)
    return rdkit_to_egc(mol, egc_hydrogen_autofill=egc_hydrogen_autofill)


rdkit_bond_type = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.QUADRUPLE,
}


# TODO in later stages combine with graph_to_rdkit function in G2S.
def egc_to_rdkit(egc):
    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for atom_id, atom_ncharge in enumerate(egc.true_ncharges()):
        a = Chem.Atom(int_atom_checked[atom_ncharge])
        mol_idx = mol.AddAtom(a)
        node_to_idx[atom_id] = mol_idx

    # add bonds between adjacent atoms
    for ix, row in enumerate(egc.true_adjmat()):
        for iy, bond in enumerate(row):
            # only traverse half the matrix
            if (iy <= ix) or (iy >= egc.num_atoms()):
                continue
            # add relevant bond type (there are many more of these)
            if bond == 0:
                continue
            else:
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], rdkit_bond_type[bond])

    # Convert RWMol to Mol object
    mol = mol.GetMol()
    # TODO: Do we need to sanitize?
    Chem.SanitizeMol(mol)
    return mol


def chemgraph_to_rdkit(
    cg: ChemGraph,
    explicit_hydrogens=True,
    resonance_struct_adj=None,
    extra_valence_hydrogens=False,
    get_rw_mol=False,
    include_SMILES=False,
):
    """
    Create an rdkit mol object from a ChemGraph object.
    """
    # create empty editable mol object
    mol = Chem.RWMol()
    nhydrogens = np.zeros((cg.nhatoms(),), dtype=int)

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for atom_id, ha in enumerate(cg.hatoms):
        a = Chem.Atom(int(ha.ncharge))
        mol_idx = mol.AddAtom(a)
        node_to_idx[atom_id] = mol_idx
        nhydrogens[atom_id] = ha.nhydrogens

    # add bonds between adjacent atoms
    for ix in range(cg.nhatoms()):
        for iy in cg.neighbors(ix):
            if iy < ix:
                continue
            btuple = (ix, iy)
            bo = cg.bond_orders[btuple]
            if resonance_struct_adj is not None:
                if btuple in cg.resonance_structure_map:
                    res_struct_id = cg.resonance_structure_map[btuple]
                    if res_struct_id in resonance_struct_adj:
                        bo = cg.aa_all_bond_orders(*btuple, unsorted=True)[
                            resonance_struct_adj[res_struct_id]
                        ]
            # add relevant bond type (there are many more of these)
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], rdkit_bond_type[bo])

    if include_SMILES:
        SMILES = MolToSmiles(mol)

    # TODO Didn't I have a DEFAULT_ATOM somewhere?
    if explicit_hydrogens:
        for ha_id, nhyd in enumerate(nhydrogens):
            for _ in range(nhyd):
                a = Chem.Atom(1)
                hidx = mol.AddAtom(a)
                mol.AddBond(node_to_idx[ha_id], hidx, rdkit_bond_type[1])
    elif extra_valence_hydrogens:
        for ha_id, ha in enumerate(cg.hatoms):
            if (resonance_struct_adj is None) or (ha.possible_valences is None):
                cur_valence = ha.valence
            else:
                # TODO do we need a function for finding which resonance structure contains a given atom?
                for i, extra_val_ids in enumerate(cg.resonance_structure_inverse_map):
                    if ha_id in extra_val_ids:
                        cur_valence = ha.possible_valences[
                            cg.resonance_structure_valence_vals[i][
                                resonance_struct_adj[res_struct_id]
                            ]
                        ]
            if cur_valence != default_valence(ha.ncharge):
                for _ in range(ha.nhydrogens):
                    a = Chem.Atom(1)
                    hidx = mol.AddAtom(a)
                    mol.AddBond(node_to_idx[ha_id], hidx, rdkit_bond_type[1])

    if not get_rw_mol:
        # Convert RWMol to Mol object
        mol = mol.GetMol()
        # TODO: Do we need to sanitize?
        Chem.SanitizeMol(mol)
    if include_SMILES:
        return mol, SMILES
    else:
        return mol


def chemgraph_to_canonical_rdkit(
    cg: ChemGraph,
    SMILES_only=False,
    print_all_SMILES_prefix=None,
    print_all_SMILES_suffix=".txt",
):
    canon_cg = canonically_permuted_ChemGraph(cg)

    canon_rdkit, canon_SMILES = chemgraph_to_rdkit(canon_cg, include_SMILES=True)

    if print_all_SMILES_prefix is not None:
        print_to_separate_file_wprefix(
            canon_SMILES,
            print_all_SMILES_prefix,
            print_file_suffix=print_all_SMILES_suffix,
        )

    if SMILES_only:
        return canon_SMILES
    else:
        return canon_rdkit, canon_SMILES


# Different optimizers available for rdkit.
class FFInconsistent(Exception):
    pass


rdkit_coord_optimizer = {
    "MMFF": AllChem.MMFFOptimizeMolecule,
    "UFF": AllChem.UFFOptimizeMolecule,
}


def RDKit_FF_optimize_coords(mol, coord_optimizer, num_attempts=1, corresponding_cg=None):
    AllChem.EmbedMolecule(mol)
    # KK: If we start working with this again uncomment and add individual exceptions.
    # try:
    #    AllChem.EmbedMolecule(mol)
    # except:
    #    if VERBOSITY != VERBOSITY_MUTED:
    #        print("#PROBLEMATIC_EMBED_MOLECULE:", corresponding_cg)
    #    raise FFInconsistent
    for _ in range(num_attempts):
        try:
            converged = coord_optimizer(mol)
        except ValueError:
            raise FFInconsistent
        if converged != 1:
            return
    raise FFInconsistent


rdkit_ff_creator = {
    "MMFF": AllChem.MMFFGetMoleculeForceField,
    "UFF": AllChem.UFFGetMoleculeForceField,
}

rdkit_properties_creator = {"MMFF": Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties}


def RDKit_FF_min_en_conf(mol, ff_type, num_attempts=1, corresponding_cg=None):
    """
    Repeats FF coordinate optimization several times to make sure the used configuration is the smallest one.
    """

    min_en = None
    min_coords = None
    min_nuclear_charges = None
    for seed in range(num_attempts):
        cur_mol = copy.deepcopy(mol)

        AllChem.EmbedMolecule(cur_mol, randomSeed=seed)
        #   KK: If we start working with this again uncomment and add individual exceptions.
        #        try:
        #            AllChem.EmbedMolecule(cur_mol, randomSeed=seed)
        #        except:
        #            if VERBOSITY != VERBOSITY_MUTED:
        #                print("#PROBLEMATIC_EMBED_MOLECULE:", corresponding_cg)
        #            raise RdKitFailure

        args = (cur_mol,)

        if ff_type in rdkit_properties_creator:
            prop_obj = rdkit_properties_creator[ff_type](cur_mol)
            args = (*args, prop_obj)
        try:
            ff = rdkit_ff_creator[ff_type](*args)
        except ValueError:
            cur_en = None
        # TODO: If we start working on this again add exceptions here.
        # try:
        converted = ff.Minimize()
        cur_en = ff.CalcEnergy()
        # except:
        #    raise FFInconsistent
        if converted != 0:
            continue
        try:
            cur_coords = np.array(np.array(cur_mol.GetConformer().GetPositions()))
            cur_nuclear_charges = np.array([atom.GetAtomicNum() for atom in cur_mol.GetAtoms()])
        except ValueError:
            cur_coords = None
            cur_nuclear_charges = None
        if ((cur_en is not None) and (cur_coords is not None)) and (
            (min_en is None) or (min_en > cur_en)
        ):
            min_en = cur_en
            min_coords = cur_coords
            min_nuclear_charges = cur_nuclear_charges

    if min_en is None:
        raise FFInconsistent

    return min_coords, min_nuclear_charges, min_en


# For generating MMFF94 coordinates corresponding to different graph objects.
# More reliable versions that check the coordinates make sense are available in xyz2graph module.
def chemgraph_to_canonical_rdkit_wcoords_no_check(
    cg, ff_type="MMFF", num_attempts=1, pick_minimal_conf=False
):
    """
    Creates an rdkit Molecule object whose heavy atoms are canonically ordered along with coordinates. See xyz2graph for version that also checks they make sense.
    cg : ChemGraph input chemgraph object
    ff_type : which forcefield to use; currently MMFF and UFF are available
    num_attempts : how many times the optimization is attempted
    output : RDKit molecule, indices of the heavy atoms, indices of heavy atoms to which a given hydrogen is connected,
    SMILES generated from the canonical RDKit molecules, and the RDKit's coordinates
    """
    (
        mol,
        canon_SMILES,
    ) = chemgraph_to_canonical_rdkit(cg)

    if pick_minimal_conf:
        rdkit_coords, _, _ = RDKit_FF_min_en_conf(
            mol, ff_type, num_attempts=num_attempts, corresponding_cg=cg
        )
    else:
        RDKit_FF_optimize_coords(
            mol,
            rdkit_coord_optimizer[ff_type],
            num_attempts=num_attempts,
            corresponding_cg=cg,
        )
        rdkit_coords = np.array(mol.GetConformer().GetPositions())

    return mol, canon_SMILES, rdkit_coords


def egc_with_coords_no_check(
    egc, coords=None, ff_type="MMFF", num_attempts=1, pick_minimal_conf=False
):
    """
    Create a copy of an ExtGraphCompound object with coordinates. If coordinates are set to None they are generated with RDKit.
    See xyz2graph for version that also checks they make sense.
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
        ) = chemgraph_to_canonical_rdkit_wcoords_no_check(
            egc.chemgraph,
            ff_type=ff_type,
            num_attempts=num_attempts,
            pick_minimal_conf=pick_minimal_conf,
        )
    else:
        (
            _,
            canon_SMILES,
        ) = chemgraph_to_canonical_rdkit(egc.chemgraph)

    output.additional_data["canon_rdkit_SMILES"] = canon_SMILES
    output.add_canon_rdkit_coords(coords)
    return output


def coord_info_from_tp_no_check(tp, **kwargs):
    """
    Coordinates corresponding to a TrajectoryPoint object. See xyz2graph for version that also checks they make sense.
    tp : TrajectoryPoint object
    num_attempts : number of attempts taken to generate MMFF coordinates (introduced because for QM9 there is a ~10% probability that the coordinate generator won't converge)
    **kwargs : keyword arguments for the egc_with_coords procedure
    """
    output = {"coordinates": None, "canon_rdkit_SMILES": None, "nuclear_charges": None}
    try:
        egc_wc = egc_with_coords_no_check(tp.egc, **kwargs)
        output["coordinates"] = egc_wc.coordinates
        output["canon_rdkit_SMILES"] = egc_wc.additional_data["canon_rdkit_SMILES"]
        output["nuclear_charges"] = egc_wc.true_ncharges()
    except FFInconsistent:
        pass

    return output


def ChemGraphStr_to_SMILES(chemgraph_str):
    cg = str2ChemGraph(chemgraph_str)
    return chemgraph_to_canonical_rdkit(cg, SMILES_only=True)


def canonical_SMILES_from_tp(tp):
    return chemgraph_to_canonical_rdkit(tp.egc.chemgraph, SMILES_only=True)


def canonical_connected_rdkit_list_from_tp(tp, SMILES_only=False):
    """
    Returns rdkit (with SMILES) of all connected fragments.
    """
    connected_chemgraphs = split_chemgraph_into_connected_fragments(tp.egc.chemgraph)
    return [
        chemgraph_to_canonical_rdkit(cg, SMILES_only=SMILES_only) for cg in connected_chemgraphs
    ]


def canonical_connected_SMILES_list_from_tp(tp):
    """
    Returns SMILES of all connected fragments.
    """
    return canonical_connected_rdkit_list_from_tp(tp, SMILES_only=True)
