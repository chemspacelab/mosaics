from .valence_treatment import ChemGraph, sorted_by_membership
from .data import NUCLEAR_CHARGE
import numpy as np

hydrogen_equivalence_class = -1  # Equivalence class reserved for hydrogen.


class ExtGraphCompound:
    def __init__(
        self,
        adjacency_matrix=None,
        nuclear_charges=None,
        chemgraph=None,
        coordinates=None,
        elements=None,
        hydrogen_autofill=False,
        bond_orders=None,
        additional_data={},
    ):
        """
        Contains all data known about a compound along with its ChemGraph.
        """
        if (nuclear_charges is None) and (elements is not None):
            nuclear_charges = [NUCLEAR_CHARGE[element] for element in elements]
        self.original_nuclear_charges = nuclear_charges
        self.original_adjacency_matrix = adjacency_matrix
        self.coordinates = coordinates
        if chemgraph is None:
            if (nuclear_charges is not None) and (
                (adjacency_matrix is not None) or (bond_orders is not None)
            ):
                self.chemgraph = ChemGraph(
                    adj_mat=adjacency_matrix,
                    nuclear_charges=nuclear_charges,
                    hydrogen_autofill=hydrogen_autofill,
                    bond_orders=bond_orders,
                )
                # Mapping between hatoms in ChemGraph and atoms in init_nuclear_charges and init_adjacency_matrix.
        else:
            self.chemgraph = chemgraph
        self.init_imported_chemgraph_mapping()
        # Create nuclear_charges and adjacency_matrix consistent with chemgraph.
        self.nuclear_charges = self.chemgraph.full_ncharges()
        self.adjacency_matrix = self.chemgraph.full_adjmat()
        # In case we want to attach more data to the same entry.
        self.additional_data = additional_data

    def original_hydrogen_hatom(self, hydrogen_id):
        adj_mat_row = self.original_adjacency_matrix[hydrogen_id]
        bond_order = 1
        if bond_order in adj_mat_row:
            hatoms_connected = np.where(adj_mat_row == bond_order)[0]
            if hatoms_connected.shape[0] == 1:
                return hatoms_connected[0]
        return None

    def init_imported_chemgraph_mapping(self):
        """
        Initialize arrays related to mapping between hatoms objects in self.chemgraph and the original nuclear_charges and adjacency_matrix.
        """
        if (self.original_adjacency_matrix is None) or (
            self.original_nuclear_charges is None
        ):
            self.original_chemgraph_mapping = None
            self.inv_original_chemgraph_mapping = None
            return
        self.original_chemgraph_mapping = np.zeros(
            (self.chemgraph.natoms(),), dtype=int
        )
        self.inv_original_chemgraph_mapping = np.zeros(
            (self.chemgraph.nhatoms(),), dtype=int
        )
        cur_hatom_index = 0
        for atom_id, nuclear_charge in enumerate(self.original_nuclear_charges):
            if nuclear_charge == 1:
                self.original_chemgraph_mapping[atom_id] = self.original_hydrogen_hatom(
                    atom_id
                )
            else:
                self.original_chemgraph_mapping[atom_id] = cur_hatom_index
                self.inv_original_chemgraph_mapping[cur_hatom_index] = atom_id
                cur_hatom_index += 1

    def hatom_ids_to_original_ids(self, hatom_ids):
        """
        Convert an array of indices corresponding to ChemGraph's heavy atoms to indices of original imported data.
        """
        return np.array(
            [self.inv_original_chemgraph_mapping[hatom_id] for hatom_id in hatom_ids]
        )

    def reverse_sorting_with_invariance_delta(self, sort_vals, delta):
        """
        Generate indices corresponding to sorting of sort_vals*(1+delta*canonical_premutation) for heavy atoms and hydrogens moved to the end.
        """
        h_val = min(sort_vals) - 1.0
        modified_sort_vals = np.repeat(h_val, self.chemgraph.natoms())
        for i in range(len(sort_vals)):
            modified_sort_vals[i] -= i * delta
        for canonical_id, original_id in zip(
            self.chemgraph.get_inv_canonical_permutation(),
            self.inv_original_chemgraph_mapping,
        ):
            modified_sort_vals[original_id] = sort_vals[original_id] * (
                1.0 + delta * canonical_id
            )
        comp_tuples = [
            (modified_val, atom_id)
            for atom_id, modified_val in enumerate(modified_sort_vals)
        ]
        comp_tuples.sort(reverse=True)
        return np.array([comp_tuple[1] for comp_tuple in comp_tuples])

    def reverse_sorting_with_invariance_indices(self, sort_vals):
        """
        Generate atom indices such that:
        * all hydrogens are put in the end of the list with no rearrangement.
        * equivalent heavy atoms inside equivalence group are placed according to canonical ordering;
        * different equivalence groups are placed relative to each other according to sort_vals average.

        Mainly introduced for KRR-related norm sorting resolved with respect to invariant atoms.
        """
        # Generate averages of sort_vals over different equivalence classes.
        original_equivalence_classes = self.original_equivalence_classes()
        equiv_class_sort_vals = sorted_by_membership(
            original_equivalence_classes, l=sort_vals
        )
        # Find averages corresponding to equivalence classes.
        equiv_class_comp_tuples = []
        for ec, vals in enumerate(equiv_class_sort_vals):
            if ec != hydrogen_equivalence_class:
                equiv_class_comp_tuples.append((np.mean(vals), ec))
        # Sort equivalence classes by descending sort_vals.
        equiv_class_comp_tuples.sort(reverse=True)
        # Dictionnary for ordering of equivalence classes according to sort_vals.
        equiv_class_order_dict = {}
        for equiv_class_ordering, (_, equiv_class_id) in enumerate(
            equiv_class_comp_tuples
        ):
            equiv_class_order_dict[equiv_class_id] = equiv_class_ordering
        # Make sure hydrogens are put in the back.
        equiv_class_order_dict[hydrogen_equivalence_class] = len(
            equiv_class_comp_tuples
        )
        # Tuples corresponding to the final ordering of atom ids.
        atom_comp_tuples = []
        for atom_id, equiv_class in enumerate(original_equivalence_classes):
            if equiv_class == hydrogen_equivalence_class:
                canonical_ordering = 0
            else:
                self.chemgraph.init_canonical_permutation()
                hatom_id = self.original_chemgraph_mapping[atom_id]
                canonical_ordering = self.chemgraph.canonical_permutation[hatom_id]
            atom_comp_tuples.append(
                (equiv_class_order_dict[equiv_class], canonical_ordering, atom_id)
            )
        atom_comp_tuples.sort()
        return np.array([atom_comp_tuple[-1] for atom_comp_tuple in atom_comp_tuples])

    def original_equivalence_classes(self):
        """
        self.chemgraph.checked_equivalence_vector output mapped on original nuclear_charges and adjacency_matrix.
        hydrogen atoms are assigned equivalence class -1.
        """
        chemgraph_equiv_vector = self.chemgraph.checked_equivalence_vector()
        output = np.repeat(hydrogen_equivalence_class, self.chemgraph.natoms())
        for hatom_id, equiv_class in enumerate(chemgraph_equiv_vector):
            output[self.inv_original_chemgraph_mapping[hatom_id]] = equiv_class
        return output

    def is_connected(self):
        return self.chemgraph.is_connected()

    def num_connected(self):
        return self.chemgraph.num_connected()

    def fragment_member_vector(self):
        return self.chemgraph.graph.components().membership

    def num_heavy_atoms(self):
        return self.chemgraph.nhatoms()

    def num_atoms(self):
        return self.chemgraph.natoms()

    def __lt__(self, egc2):
        return self.chemgraph < egc2.chemgraph

    def __gt__(self, egc2):
        return self.chemgraph > egc2.chemgraph

    def __eq__(self, egc2):
        if not isinstance(egc2, ExtGraphCompound):
            return False
        return self.chemgraph == egc2.chemgraph

    def __str__(self):
        return str(self.chemgraph)

    def __repr__(self):
        return str(self)
