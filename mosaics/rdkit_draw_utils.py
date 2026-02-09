# Collection of routines that use rdkit.Chem.Draw for easy display of objects used throughout the chemxpl module.
# TODO PROPER ABBREVIATION SUPPORT! MAY CRASH WITH HIGHLIGHTS!
import itertools
import os
import subprocess
from copy import deepcopy

from rdkit.Chem import RemoveHs, rdAbbreviations
from rdkit.Chem.Draw import rdMolDraw2D

from .chem_graph import ChemGraph
from .crossover import (
    FragmentPair,
    crossover_outcomes,
    frag_size_status_list,
    matching_frag_size_status_list,
)
from .ext_graph_compound import ExtGraphCompound
from .misc_procedures import sorted_tuple, str_atom_corr
from .modify import (
    add_heavy_atom_chain,
    change_bond_order,
    change_bond_order_valence,
    change_valence,
    change_valence_add_atoms,
    change_valence_remove_atoms,
    egc_change_func,
    remove_heavy_atom,
    replace_heavy_atom,
)
from .random_walk import TrajectoryPoint, full_change_list
from .rdkit_utils import SMILES_to_egc, chemgraph_to_rdkit, rdkit_bond_type
from .utils import mkdir

# Some colors that I think look good on print.
RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)
BLUE = (0.0, 0.0, 1.0)

LIGHTRED = (1.0, 0.5, 0.5)
LIGHTGREEN = (0.5, 1.0, 0.5)
LIGHTBLUE = (0.5, 0.5, 1.0)

png = "PNG"
svg = "SVG"
pdf = "PDF"
default_drawing_format = png
drawing_formats = [png, svg, pdf]
drawing_generator = {
    png: rdMolDraw2D.MolDraw2DCairo,
    svg: rdMolDraw2D.MolDraw2DSVG,
    pdf: rdMolDraw2D.MolDraw2DSVG,
}
drawing_file_suffix = {png: ".png", svg: ".svg", pdf: ".pdf"}


class ChemGraphDrawing:
    def __init__(
        self,
        chemgraph=None,
        SMILES=None,
        size=(300, 300),
        bw_palette=True,
        kekulize=True,
        explicit_hydrogens=False,
        highlightAtoms=None,
        highlightAtomColor=None,
        highlightAtomColors=None,
        highlight_connecting_bonds=False,
        highlightBondTuples=None,
        highlightBondTupleColor=None,
        highlightBondTupleColors=None,
        highlightAtomRadius=None,
        highlightAtomRadii=None,
        highlightBondWidthMultiplier=None,
        bondLineWidth=None,
        baseFontSize=None,
        resonance_struct_adj=None,
        abbrevs=None,
        abbreviate_max_coverage=1.0,
        rotate=None,
        centreMoleculesBeforeDrawing=None,
        padding=None,
        post_added_bonds=None,
        file_format=default_drawing_format,
    ):
        """
        Create an RdKit illustration depicting a partially highlighted ChemGraph.
        """
        self.base_init(
            size=size,
            chemgraph=chemgraph,
            SMILES=SMILES,
            bw_palette=bw_palette,
            highlightBondWidthMultiplier=highlightBondWidthMultiplier,
            kekulize=kekulize,
            explicit_hydrogens=explicit_hydrogens,
            bondLineWidth=bondLineWidth,
            baseFontSize=baseFontSize,
            resonance_struct_adj=resonance_struct_adj,
            highlightAtoms=highlightAtoms,
            highlightAtomColor=highlightAtomColor,
            highlightAtomColors=highlightAtomColors,
            highlight_connecting_bonds=highlight_connecting_bonds,
            highlightBondTuples=highlightBondTuples,
            highlightBondTupleColor=highlightBondTupleColor,
            highlightBondTupleColors=highlightBondTupleColors,
            highlightAtomRadius=highlightAtomRadius,
            highlightAtomRadii=highlightAtomRadii,
            abbrevs=abbrevs,
            abbreviate_max_coverage=abbreviate_max_coverage,
            rotate=rotate,
            centreMoleculesBeforeDrawing=centreMoleculesBeforeDrawing,
            padding=padding,
            post_added_bonds=post_added_bonds,
            file_format=file_format,
        )
        self.prepare_and_draw()

    def base_init(
        self,
        chemgraph=None,
        SMILES=None,
        bw_palette=True,
        highlightBondWidthMultiplier=None,
        size=(300, 300),
        resonance_struct_adj=None,
        kekulize=False,
        explicit_hydrogens=False,
        bondLineWidth=None,
        baseFontSize=None,
        highlightAtoms=None,
        highlightAtomColor=None,
        highlightAtomColors=None,
        highlight_connecting_bonds=False,
        highlightBondTuples=None,
        highlightBondTupleColor=None,
        highlightBondTupleColors=None,
        highlightAtomRadius=None,
        highlightAtomRadii=None,
        abbrevs=None,
        abbreviate_max_coverage=1.0,
        rotate=None,
        centreMoleculesBeforeDrawing=None,
        padding=None,
        post_added_bonds=None,
        file_format=default_drawing_format,
    ):
        if chemgraph is None:
            if SMILES is not None:
                self.chemgraph = SMILES_to_egc(SMILES).chemgraph
        else:
            self.chemgraph = chemgraph
        self.resonance_struct_adj = resonance_struct_adj

        self.explicit_hydrogens = explicit_hydrogens

        self.kekulize = kekulize
        self.bw_palette = bw_palette
        self.file_format = file_format
        self.drawing = drawing_generator[file_format](*size)

        do = self.drawing.drawOptions()
        if bw_palette:
            do.useBWAtomPalette()
        if rotate is not None:
            do.rotate = rotate
        if centreMoleculesBeforeDrawing is not None:
            do.centreMoleculesBeforeDrawing = centreMoleculesBeforeDrawing
        if padding is not None:
            do.padding = padding
        if highlightBondWidthMultiplier is not None:
            do.highlightBondWidthMultiplier = highlightBondWidthMultiplier
        if bondLineWidth is not None:
            do.bondLineWidth = bondLineWidth
        if baseFontSize is not None:
            do.baseFontSize = baseFontSize

        self.highlightAtoms = deepcopy(highlightAtoms)
        self.highlightAtomColors = deepcopy(highlightAtomColors)
        self.highlightAtomColor = highlightAtomColor

        self.highlightAtomRadius = highlightAtomRadius
        self.highlightAtomRadii = deepcopy(highlightAtomRadii)

        self.highlightBondTupleColor = highlightBondTupleColor
        self.highlightBondTupleColors = deepcopy(highlightBondTupleColors)
        self.highlight_connecting_bonds = highlight_connecting_bonds
        self.highlightBondTuples = deepcopy(highlightBondTuples)

        self.abbrevs = abbrevs
        self.abbreviate_max_coverage = abbreviate_max_coverage

        if self.highlightAtomColor is not None:
            self.highlight_atoms(self.highlightAtoms, self.highlightAtomColor)

        # TODO cleaner way to do it?
        if self.highlightAtomColors is not None:
            self.highlight_atoms(
                list(self.highlightAtomColors.keys()),
                list(self.highlightAtomColors.values())[0],
            )

        if self.highlightBondTupleColor:
            self.highlight_bonds(self.highlightBondTuples, self.highlightBondTupleColor)

        if self.highlight_connecting_bonds:
            self.highlight_bonds_connecting_atoms(self.highlightAtoms, self.highlightAtomColor)

        self.post_added_bonds = post_added_bonds

    def highlight_atoms_bonds(self, atoms_bonds_list, highlight_color, overwrite=False):
        for ab in atoms_bonds_list:
            if isinstance(ab, tuple):
                self.highlight_bonds([ab], highlight_color, overwrite=overwrite)
            else:
                self.highlight_atoms([ab], highlight_color, overwrite=overwrite)

    def highlight_atoms(self, atom_ids, highlight_color, wbonds=False, overwrite=False):
        if highlight_color is None:
            return
        if self.highlightAtomColors is None:
            self.highlightAtomColors = {}
        if self.highlightAtoms is None:
            self.highlightAtoms = list(self.highlightAtomColors.keys())
        for ha in atom_ids:
            if (ha not in self.highlightAtomColors) or overwrite:
                self.highlightAtomColors[ha] = highlight_color
            if ha not in self.highlightAtoms:
                self.highlightAtoms.append(ha)
        if self.highlightAtomRadius is not None:
            if self.highlightAtomRadii is None:
                self.highlightAtomRadii = {}
            for ha in atom_ids:
                if (ha not in self.highlightAtomRadii) or overwrite:
                    self.highlightAtomRadii[ha] = self.highlightAtomRadius
        if wbonds:
            self.highlight_bonds_connecting_atoms(atom_ids, highlight_color, overwrite=overwrite)

    def highlight_bonds(self, bond_tuples, highlight_color, overwrite=False):
        if highlight_color is None:
            return
        self.check_highlightBondTupleColors()
        for bt in bond_tuples:
            if (bt not in self.highlightBondTupleColors) or overwrite:
                self.highlightBondTupleColors[bt] = highlight_color

    def highlight_bonds_connecting_atoms(self, atom_ids, highlight_color, overwrite=False):
        connecting_bts = self.connecting_bond_tuples(atom_ids)
        self.highlight_bonds(connecting_bts, highlight_color, overwrite=overwrite)

    def check_highlightBondTupleColors(self):
        if self.highlightBondTupleColors is None:
            self.highlightBondTupleColors = {}

    def prepare_and_draw(self):
        self.mol = chemgraph_to_rdkit(
            self.chemgraph,
            resonance_struct_adj=self.resonance_struct_adj,
            explicit_hydrogens=self.explicit_hydrogens,
            extra_valence_hydrogens=True,
            get_rw_mol=True,
        )
        # Hydrogen atoms are added anyway via extra_valence_hydrogens to mark valences, they need to be deleted.
        self.check_rdkit_mol_extra_mods()
        if not self.explicit_hydrogens:
            self.mol = RemoveHs(self.mol)
        if self.abbrevs is not None:
            used_abbrevs = rdAbbreviations.GetDefaultAbbreviations()
            self.full_mol = self.mol
            self.mol = rdAbbreviations.CondenseMolAbbreviations(
                self.full_mol, used_abbrevs, maxCoverage=self.abbreviate_max_coverage
            )

        if self.highlightBondTupleColors is not None:
            highlightBonds = []
            highlightBondColors = {}
            for bt in self.highlightBondTupleColors:
                bond_id = self.mol.GetBondBetweenAtoms(int(bt[0]), int(bt[1])).GetIdx()
                highlightBonds.append(bond_id)
                highlightBondColors[bond_id] = self.highlightBondTupleColors[bt]
        else:
            highlightBonds = None
            highlightBondColors = None

        rdMolDraw2D.PrepareAndDrawMolecule(
            self.drawing,
            self.mol,
            kekulize=self.kekulize,
            highlightAtomColors=self.highlightAtomColors,
            highlightAtoms=self.highlightAtoms,
            highlightBonds=highlightBonds,
            highlightBondColors=highlightBondColors,
            highlightAtomRadii=self.highlightAtomRadii,
        )

    def check_rdkit_mol_extra_mods(self):
        if self.post_added_bonds is not None:
            for pab in self.post_added_bonds:
                self.mol.AddBond(pab[0], pab[1], rdkit_bond_type[pab[2]])
        # Convert RWMol to Mol object
        self.mol = self.mol.GetMol()
        # TODO: Do we need to sanitize?
        # Chem.SanitizeMol(self.mol)

    def save(self, filename):
        text = self.drawing.GetDrawingText()
        print_output_io_type = "w"
        if self.file_format == pdf:
            print_output = filename + drawing_file_suffix[svg]
        else:
            print_output = filename
            if self.file_format == png:
                print_output_io_type = "wb"

        with open(print_output, print_output_io_type) as f:
            f.write(text)
        if self.file_format == pdf:
            subprocess.run(
                [
                    "inkscape",
                    "-D",
                    "-z",
                    print_output,
                    "--export-area-drawing",
                    "-o",
                    filename,
                ]
            )

    def connecting_bond_tuples(self, atom_ids):
        output = []
        for atom_id in atom_ids:
            for neigh in self.chemgraph.neighbors(atom_id):
                if neigh < atom_id:
                    if neigh in atom_ids:
                        output.append((neigh, atom_id))
        return output


class FragmentPairDrawing(ChemGraphDrawing):
    def __init__(
        self,
        fragment_pair=None,
        bw_palette=True,
        size=(300, 300),
        resonance_struct_adj=None,
        highlight_fragment_colors=[LIGHTBLUE, LIGHTRED],
        bondLineWidth=None,
        highlight_fragment_boundary=LIGHTGREEN,
        highlightAtomRadius=None,
        highlightBondWidthMultiplier=None,
        baseFontSize=None,
        abbrevs=None,
        abbreviate_max_coverage=1.0,
        rotate=None,
        file_format=default_drawing_format,
    ):
        """
        Create an RdKit illustration depicting a FragmentPair with atoms and bonds highlighted according to membership.
        """
        # Initialize all basic quantities.
        self.base_init(
            chemgraph=fragment_pair.chemgraph,
            bw_palette=bw_palette,
            size=size,
            resonance_struct_adj=resonance_struct_adj,
            highlightBondWidthMultiplier=highlightBondWidthMultiplier,
            bondLineWidth=bondLineWidth,
            baseFontSize=baseFontSize,
            abbrevs=abbrevs,
            abbreviate_max_coverage=abbreviate_max_coverage,
            highlightAtomRadius=highlightAtomRadius,
            rotate=rotate,
            file_format=file_format,
        )
        # For starters only highlight the bonds connecting the two fragments.
        self.highlight_fragment_colors = highlight_fragment_colors
        self.highlight_fragment_boundary = highlight_fragment_boundary

        self.connection_tuples = []
        for tuples in fragment_pair.affected_status[0].bond_tuple_dict.values():
            self.connection_tuples += tuples

        if self.highlight_fragment_boundary is not None:
            self.highlight_bonds(self.connection_tuples, self.highlight_fragment_boundary)
        if self.highlight_fragment_colors is not None:
            for frag_id, fragment_highlight in enumerate(self.highlight_fragment_colors):
                vertices = fragment_pair.get_sorted_vertices(frag_id)
                if fragment_highlight is not None:
                    self.highlight_atoms(vertices, fragment_highlight, wbonds=True)
        self.prepare_and_draw()


def ObjDrawing(obj, **kwargs):
    if isinstance(obj, ChemGraph):
        return ChemGraphDrawing(obj, **kwargs)
    if isinstance(obj, FragmentPair):
        return FragmentPairDrawing(obj, **kwargs)


bond_changes = [change_bond_order, change_bond_order_valence]

valence_changes = [
    change_valence,
    change_valence_add_atoms,
    change_valence_remove_atoms,
]
atom_removals = [remove_heavy_atom, change_valence_remove_atoms]
atom_additions = [add_heavy_atom_chain, change_valence_add_atoms]


class ModificationPathIllustration(ChemGraphDrawing):
    def __init__(
        self,
        cg,
        modification_path,
        change_function,
        color_change_main=None,
        color_change_minor=None,
        color_change_special=None,
        **other_image_params,
    ):
        """
        Illustrate a modification path with simple moves as applied to a ChemGraph object.
        """
        self.base_init(chemgraph=cg, **other_image_params)
        self.modification_path = modification_path
        self.change_function = change_function
        self.color_change_main = color_change_main
        self.color_change_minor = color_change_minor
        self.color_change_special = color_change_special
        self.highlight_atoms_bonds(self.change_main(), self.color_change_main)
        self.highlight_atoms_bonds(self.change_minor(), self.color_change_minor)
        self.highlight_atoms_bonds(self.change_special(), self.color_change_special)
        self.init_resonance_struct_adj()
        self.prepare_and_draw()

    def init_resonance_struct_adj(self):
        if self.chemgraph.resonance_structure_map is None:
            return
        affected_resonance_region = None
        res_struct_id = None

        if self.change_function in bond_changes:
            res_struct_id = self.modification_path[1][-1]

        if self.change_function in [replace_heavy_atom, change_valence]:
            res_struct_id = self.modification_path[1][1]

        for changed_atom in self.changed_atoms():
            for i, extra_valence_ids in enumerate(self.chemgraph.resonance_structure_inverse_map):
                if changed_atom in extra_valence_ids:
                    affected_resonance_region = i

        if (affected_resonance_region is not None) and (res_struct_id is not None):
            self.resonance_struct_adj = {affected_resonance_region: res_struct_id}

    def removed_atoms(self):
        if self.change_function == change_valence_remove_atoms:
            return list(self.modification_path[2][0])
        if self.change_function == remove_heavy_atom:
            return [self.modification_path[1][0]]
        raise Exception

    def neighbor_to_removed(self):
        return self.chemgraph.neighbors(self.removed_atoms()[0])[0]

    def changed_bond(self):
        return sorted_tuple(*self.modification_path[1][:2])

    def change_main(self):
        if self.change_function in [change_valence_remove_atoms, remove_heavy_atom]:
            atoms = self.removed_atoms()
            neighbor = self.neighbor_to_removed()
            bonds = [sorted_tuple(atom, neighbor) for atom in atoms]
            return atoms + bonds
        if self.change_function == replace_heavy_atom:
            return [self.modification_path[1][0]]
        return []

    def change_minor(self):
        if self.change_function == add_heavy_atom_chain:
            return [self.modification_path[1]]
        if self.change_function == remove_heavy_atom:
            return [self.neighbor_to_removed()]
        if self.change_function == change_bond_order:
            return list(self.changed_bond())
        if self.change_function == change_bond_order_valence:
            return [self.modification_path[1][1]]
        return []

    def change_special(self):
        if self.change_function == change_valence_add_atoms:
            return [self.modification_path[1]]
        if self.change_function == change_valence:
            return [self.modification_path[0]]
        if self.change_function == change_valence_remove_atoms:
            return [self.neighbor_to_removed()]
        if self.change_function in bond_changes:
            output = []
            bond = self.changed_bond()
            if self.chemgraph.bond_order(*bond) != 0:
                output.append(bond)
            if self.change_function == change_bond_order_valence:
                output.append(self.modification_path[1][0])
            return output
        return []

    def changed_atoms(self):
        output = []
        for atom_bonds in itertools.chain(
            self.change_main(), self.change_minor(), self.change_special()
        ):
            if isinstance(atom_bonds, int):
                output.append(atom_bonds)
        return output


def first_mod_path(tp):
    output = []
    subd = tp.modified_possibility_dict
    while isinstance(subd, list) or isinstance(subd, dict):
        if isinstance(subd, list):
            output.append(subd[0])
            subd = None
        if isinstance(subd, dict):
            new_key = list(subd.keys())[0]
            subd = subd[new_key]
            output.append(new_key)
    return output


def all_mod_paths(cg, **randomized_change_params):
    cur_tp = TrajectoryPoint(cg=cg)
    cur_tp.init_possibility_info(**randomized_change_params)
    cur_tp.modified_possibility_dict = cur_tp.possibility_dict
    output = []
    while cur_tp.modified_possibility_dict != {}:
        full_mod_path = first_mod_path(cur_tp)
        output.append(deepcopy(full_mod_path))
        cur_tp.delete_mod_path(full_mod_path)
    return output


default_randomized_change_params = {
    "change_prob_dict": full_change_list,
    "possible_elements": ["C"],
    "added_bond_orders": [1],
    "chain_addition_tuple_possibilities": False,
    "bond_order_changes": [-1, 1],
    "bond_order_valence_changes": [-2, 2],
    "max_fragment_num": 1,
    "added_bond_orders_val_change": [1, 2],
}


class NoAfter(Exception):
    pass


class BeforeAfterIllustration:
    def __init__(
        self,
        cg,
        modification_path,
        change_function,
        prefixes=["forward_", "inv_"],
        randomized_change_params=default_randomized_change_params,
        **other_image_params,
    ):
        """
        Create a pair of illustrations corresponding to a modification_path.
        """

        self.prefixes = prefixes
        egc = ExtGraphCompound(chemgraph=cg)
        new_egc = egc_change_func(egc, modification_path, change_function)
        if new_egc is None:
            raise NoAfter

        new_cg = new_egc.chemgraph

        inv_mod_path = None
        for full_mod_path in all_mod_paths(new_cg, **randomized_change_params):
            inv_egc = egc_change_func(new_egc, full_mod_path[1:], full_mod_path[0])
            if inv_egc == egc:
                inv_mod_path = full_mod_path
                break
        if inv_mod_path is None:
            raise Exception()
        self.chemgraphs = [cg, new_cg]
        self.modification_paths = [modification_path, inv_mod_path[1:]]
        self.change_functions = [change_function, inv_mod_path[0]]
        self.illustrations = []
        for cur_cg, cur_mod_path, cur_change_function in zip(
            self.chemgraphs, self.modification_paths, self.change_functions
        ):
            self.illustrations.append(
                ModificationPathIllustration(
                    cur_cg, cur_mod_path, cur_change_function, **other_image_params
                )
            )

    def save(self, base_filename):
        for ill, pref in zip(self.illustrations, self.prefixes):
            filename = pref + base_filename
            ill.save(filename)


def draw_chemgraph_to_file(cg, filename, **kwargs):
    """
    Draw a chemgraph in an image file.
    """
    cgd = ChemGraphDrawing(chemgraph=cg, **kwargs)
    cgd.save(filename)


def draw_fragment_pair_to_file(fragment_pair, filename, **kwargs):
    """
    Draw a fragment pair in an image file.
    """
    fpd = FragmentPairDrawing(fragment_pair=fragment_pair, **kwargs)
    fpd.save(filename)


def all_possible_resonance_struct_adj(obj):
    """
    All values of resonance_struct_adj dictionnary appearing in *Drawing objects that are valid for a given object.
    """
    if isinstance(obj, ChemGraph):
        cg = obj
    else:
        cg = obj.chemgraph
    iterators = [
        list(range(len(res_struct_orders))) for res_struct_orders in cg.resonance_structure_orders
    ]
    if len(iterators) == 0:
        return [None]
    output = []
    for res_adj_ids in itertools.product(*iterators):
        new_dict = {}
        for reg_id, adj_id in enumerate(res_adj_ids):
            new_dict[reg_id] = adj_id
        output.append(new_dict)
    return output


def check_filename_suffix(filename_suffix, kwargs):
    if filename_suffix is not None:
        return filename_suffix
    file_format_key = "file_format"
    if file_format_key in kwargs:
        return drawing_file_suffix[kwargs[file_format_key]]
    return drawing_file_suffix[default_drawing_format]


def draw_all_possible_resonance_structures(obj, filename_prefix, filename_suffix=None, **kwargs):
    """
    Draw variants of an object with all possible resonance structures.
    """
    filename_suffix_checked = check_filename_suffix(filename_suffix, kwargs)

    for rsa_id, resonance_struct_adj in enumerate(all_possible_resonance_struct_adj(obj)):
        cur_drawing = ObjDrawing(obj, resonance_struct_adj=resonance_struct_adj, **kwargs)
        cur_drawing.save(filename_prefix + str(rsa_id) + filename_suffix_checked)


def draw_all_modification_possibilities(
    cg,
    filename_prefix,
    filename_suffix=None,
    randomized_change_params=default_randomized_change_params,
    draw_pairs=True,
    dump_directory=None,
    **kwargs,
):
    # Check that cg satisfies the randomized_change_params_dict
    randomized_change_params = deepcopy(randomized_change_params)

    if "possible_elements" not in randomized_change_params:
        randomized_change_params["possible_elements"] = []

    for ha in cg.hatoms:
        el = str_atom_corr[ha.ncharge]
        if el not in randomized_change_params["possible_elements"]:
            randomized_change_params["possible_elements"].append(el)

    creator_kwargs = deepcopy(kwargs)
    if draw_pairs:
        creator = BeforeAfterIllustration
        creator_kwargs = {
            **creator_kwargs,
            "randomized_change_params": randomized_change_params,
        }
    else:
        creator = ModificationPathIllustration

    if dump_directory is not None:
        workdir = os.getcwd()
        os.chdir(dump_directory)

    filename_suffix_checked = check_filename_suffix(filename_suffix, kwargs)

    for counter, full_mod_path in enumerate(all_mod_paths(cg, **randomized_change_params)):
        try:
            mpi = creator(cg, full_mod_path[1:], full_mod_path[0], **creator_kwargs)
        except NoAfter:
            mpi = ModificationPathIllustration(cg, full_mod_path[1:], full_mod_path[0], **kwargs)
        mpi.save(filename_prefix + str(counter) + filename_suffix_checked)

    if dump_directory is not None:
        os.chdir(workdir)


def draw_all_possible_fragment_pairs(
    cg: ChemGraph, filename_prefix="fragment_pair_", max_num_affected_bonds=3, **other_kwargs
):
    for origin_point_id, origin_point in enumerate(cg.unrepeated_atom_list()):
        for size_id, (size, _) in enumerate(
            frag_size_status_list(cg, origin_point, max_num_affected_bonds=max_num_affected_bonds)
        ):
            filename_final_prefix = (
                filename_prefix + str(origin_point_id) + "_" + str(size_id) + "_"
            )
            cur_frag = FragmentPair(cg, origin_point, neighborhood_size=size)
            draw_all_possible_resonance_structures(cur_frag, filename_final_prefix, **other_kwargs)


def draw_all_crossovers(
    cg_pair,
    init_folder_prefix="init_opt_",
    filename_prefixes=["old_", "new_"],
    max_num_affected_bonds=3,
    nhatoms_range=None,
    smallest_exchange_size=2,
    **other_kwargs,
):
    init_option = 0

    orig_point_iterators = itertools.product(*[cg.unrepeated_atom_list() for cg in cg_pair])

    for origin_points in orig_point_iterators:
        forward_mfssl = matching_frag_size_status_list(
            cg_pair,
            origin_points,
            max_num_affected_bonds=max_num_affected_bonds,
            nhatoms_range=nhatoms_range,
            smallest_exchange_size=smallest_exchange_size,
        )
        for chosen_sizes in forward_mfssl:
            init_dir = init_folder_prefix + str(init_option)
            mkdir(init_dir)
            os.chdir(init_dir)
            old_fragments = [
                FragmentPair(cg, origin_point, neighborhood_size=chosen_size)
                for cg, origin_point, chosen_size in zip(cg_pair, origin_points, chosen_sizes)
            ]
            for old_frag_id, old_fragment in enumerate(old_fragments):
                draw_all_possible_resonance_structures(
                    old_fragment, filename_prefixes[0] + str(old_frag_id) + "_", **other_kwargs
                )
            trial_option = 0
            new_cg_pairs, new_origin_points = crossover_outcomes(
                cg_pair, chosen_sizes, origin_points
            )
            if new_cg_pairs is None:
                os.chdir("..")
                continue
            for new_cg_pair in new_cg_pairs:
                new_fragments = [
                    FragmentPair(cg, origin_point, neighborhood_size=chosen_size)
                    for cg, origin_point, chosen_size in zip(
                        new_cg_pair, new_origin_points, chosen_sizes
                    )
                ]
                for new_frag_id, new_fragment in enumerate(new_fragments):
                    draw_all_possible_resonance_structures(
                        new_fragment,
                        filename_prefixes[1] + str(trial_option) + "_" + str(new_frag_id) + "_",
                        **other_kwargs,
                    )
                trial_option += 1
            init_option += 1
            os.chdir("..")
    return
