import streamlit as st
from mosaics.minimized_functions import chemspace_potentials
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, display
import cairosvg
from rdkit.Chem import AllChem
import csv
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
random.seed(42)
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mosaics.minimized_functions import chemspace_sampler_default_params


import pdb


fs = 12
plt.rc("font", size=fs)
plt.rc("axes", titlesize=fs)
plt.rc("axes", labelsize=fs)
plt.rc("xtick", labelsize=fs)
plt.rc("ytick", labelsize=fs)
plt.rc("legend", fontsize=fs)
plt.rc("figure", titlesize=fs)


def make_pretty(axi):
    """
    Method to make the axes look pretty
    """
    axi.spines["right"].set_color("none")
    axi.spines["top"].set_color("none")
    axi.spines["bottom"].set_position(("axes", -0.05))
    axi.spines["bottom"].set_color("black")
    axi.spines["left"].set_color("black")
    axi.yaxis.set_ticks_position("left")
    axi.xaxis.set_ticks_position("bottom")
    axi.spines["left"].set_position(("axes", -0.05))
    return axi



sns.set_style("whitegrid")  # Set style to whitegrid for better readability
sns.set_context("notebook")  # Set context to "notebook"

st.set_page_config(
   page_title="Chemspace",
   page_icon=":shark:",
   layout="wide",
)

def check_string_in_list(lst):
    for string in lst:
        if string not in ["N", "O", "C"]:
            return True
    return False

def mol_to_img(mol):
    mol = AllChem.RemoveHs(mol)
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return cairosvg.svg2png(bytestring=svg.encode('utf-8'))

def mol_to_3d(mol):
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    return mol

def str_to_tuple_list(string):
    string = string[1:-1]
    tuples = string.split('),')
    tuples = [tuple(map(int, t.replace('(', '').replace(')', '').split(','))) for t in tuples]
    return tuples


descriptor_options = ['RDKit', 'ECFP4','BoB', 'SOAP','CM', 'MBDF'] #,'atomic_energy']

import streamlit as st

st.title('ChemSpace Sampler App')

st.write('📘 README:')
st.write('🔬 This application generates new chemical structures starting from a given molecule. Just enter the parameters below and click "Run ChemSpace Sampler!"')

st.write('🌈 Ensemble representation will make distances less noisy. Without it, you may get distances vastly outside of the defined target interval (min_d, max_d). However, it will take longer to run.')

st.write('⚠️ Note: When the maximal distance is too small, you may get no molecules in the interval, but this also depends on the representation.')

st.write('🔍 If you want to find the closest molecules, set both distances to 0 and uncheck "RETURN MOLECULES STRICTLY IN THE INTERVAL".')


# Parameters input
st.sidebar.image("/app/examples/05_chemspacesampler/app/radar.png", caption='Navigate chemical space', use_column_width=True)

st.sidebar.subheader('Input Parameters')



help_text_selected_descriptor = 'Choose the representation (descriptor) used to calculate the distance between molecules. \
                                 The descriptors are: RDKit (check rdkit.Chem.Descriptors), ECFP4 (hashed group based), BoB (Bag of Bonds), (global) SOAP (Smooth Overlap of Atomic Positions)  doi:10.1039/c6cp00415f, CM (Coulomb Matrix) Phys. Rev. Lett. 108, 058301 . MBDF (many-body distribution functionals) https://doi.org/10.1063/5.0152215'

smiles = st.sidebar.text_input('Start molecule', value="OC1=CC=CC=C1", help='Enter the SMILES string of your starting molecule.')
selected_descriptor = st.sidebar.selectbox('Select Representation', descriptor_options, help=help_text_selected_descriptor)
min_d = st.sidebar.number_input('Min distance', value=0.0, help='Enter the minimal desired distance from the central molecule.')
max_d = st.sidebar.number_input('Max distance', value=0.0, help='Enter the maximal desired distance from the central molecule.')
Nsteps = st.sidebar.number_input('#MC steps', value=30, help='Enter the number of Monte Carlo iterations to be performed.')
possible_elements = st.sidebar.multiselect(
    'Select allowed elements in the generated molecules',
    options=['C', 'O', 'N', 'F', 'P', 'S', 'Si', 'Br', 'Cl', 'B'],
    default=['C', 'O', 'N', 'F'],  help='Enter the elements that are allowed in the generated molecules.')
nhatoms_range = st.sidebar.text_input('Number of heavy atoms (non-hydrogen)', value="2, 9", help='Enter the range of the number of heavy atoms that should be in the generated molecules.').split(', ')
synth_cut_soft, synth_cut_hard = st.sidebar.slider('Select soft and hard cutoff for Synthesizability (1 easy to 10 impossible to make) read the (?) for more info',
                                           min_value=1.0,
                                           max_value=10.0,
                                           value=(6.8, 9.5),
                                           step=0.1,
                                           help='Move the slider to set the soft and hard synthesizability cut-off. A lower value means easier to synthesize. Left slider at 2 and right 5 means up to 2 is always accepted, above 5 is always rejected. ')

strictly_in =  st.sidebar.checkbox('Only return molecules strictly in the distance interval?', value=False, help='During MC you also accept molecuels outside of the 0 interval if temperature allows, this just affects postprocessing')
mmff_check = st.sidebar.checkbox('MMFF94 parameters exist? (another sanity check)', value=True, help='Check if the generated molecules should have MMFF94 parameters.')
ensemble   = st.sidebar.checkbox('Ensemble representation (affects only geometry-based representations, BoB & SOAP)', value=False, help='Check if the ensemble representation should be used. It affects only geometry-based representations (BoB & SOAP).')
default_bonds = "(8, 9), (8, 8), (9, 9), (7, 7)"

# Input field for forbidden bonds
bonds_input = st.sidebar.text_input("Enter forbidden bonds as pairs (a, b), separated by commas:", value=default_bonds, help="Enter forbidden bonds as pairs (a, b), separated by commas where a anb b are the atomic numbers of the atoms forming the bond.")


#just add some text to the sidebar
st.sidebar.write('📚 References: Understanding Representations by Exploring Galaxies in Chemical Space, Jan Weinreich, Konstantin Karandashev, Guido von Rudorff arXiv:xxxx.xxxxx')

# Convert input string to list of tuples
try:
    forbidden_bonds = [tuple(map(int, bond.strip(" ()").split(","))) for bond in bonds_input.split("),")]
    st.success("Press 'Run ChemSpace Sampler' to start with these parameters.")
    st.success("Start molecule: {}".format(smiles))
    st.success("Minimal distance: {}".format(min_d))
    st.success("Maximal distance: {}".format(max_d))
    st.success("Number of MC iterations: {}".format(Nsteps))
    st.success("Allowed elements: {}".format(possible_elements))
    st.success("Number of heavy atoms: {}".format(nhatoms_range))
    st.success("Synthesizability cutoff: {}".format((synth_cut_soft, synth_cut_hard)))
    st.success("Ensemble representation: {}".format(ensemble))
    st.success("MMFF94 parameters exist: {}".format(mmff_check))
    st.success("Forbidden bonds: {}".format(forbidden_bonds))
except ValueError:
    st.error("Invalid input parameters. Please check your input and try again.")

# Convert input string to list of tuples
try:
    forbidden_bonds = [tuple(map(int, bond.strip(" ()").split(","))) for bond in bonds_input.split("),")]
    st.success("Forbidden bonds: {}".format(forbidden_bonds))
except ValueError:
    st.error("Invalid input. Please enter forbidden bonds as pairs (a, b), separated by commas.")


if selected_descriptor == 'RDKit':
    chemspace_function = chemspace_potentials.chemspacesampler_MolDescriptors
elif selected_descriptor == 'ECFP4':
    chemspace_function = chemspace_potentials.chemspacesampler_ECFP
elif selected_descriptor == 'BoB':
    chemspace_function = chemspace_potentials.chemspacesampler_BoB
elif selected_descriptor == 'SOAP':
    chemspace_function = chemspace_potentials.chemspacesampler_SOAP
elif selected_descriptor == 'CM':
    chemspace_function = chemspace_potentials.chemspacesampler_CM
elif selected_descriptor == 'MBDF':
    chemspace_function = chemspace_potentials.chemspacesampler_MBDF
elif selected_descriptor == 'atomic_energy':
    chemspace_function = chemspace_potentials.chemspacesampler_atomization_rep
    if check_string_in_list(possible_elements):
        print("Only C, N, O are allowed for atomic energy")
        st.error("Only C, N, O are allowed for atomic energy")
        possible_elements = ["C", "O", "N"]
    else:
        pass

else:
    st.error('Unknown Descriptor selected')

if st.button('Run ChemSpace Sampler'):
    try:
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol != 'H' and symbol not in possible_elements:
                possible_elements.append(symbol)
                st.success(f'The starting molecule contains element {symbol} which is not in the allowed elements list. It was therefore added...')
                st.write("Updated possible elements:", possible_elements)

        atom_symbols = [s.GetSymbol() for s in mol.GetAtoms()]
        atom_symbols_no_H = [symbol for symbol in atom_symbols if symbol != 'H']
        if int(nhatoms_range[1]) <  len(atom_symbols_no_H):
            nhatoms_range[1] = str(len(atom_symbols_no_H))
            st.success(f'The starting molecule contains more heavy atoms than the upper bound of the permitted range. It was therefore updated to {nhatoms_range[1]}...')
            st.write("Updated number of heavy atoms:", nhatoms_range)

        params = chemspace_sampler_default_params.make_params_dict(selected_descriptor, min_d, max_d,strictly_in, Nsteps, possible_elements, forbidden_bonds, nhatoms_range, synth_cut_soft,synth_cut_hard, ensemble, mmff_check)

        MOLS, D = chemspace_function(smiles=smiles, params=params)
        print(MOLS)
        if len(MOLS) == 0:
            st.error('No molecules found. Try to change the parameters such as increasing the minimal distance or the number of iterations.')
            st.stop()
        ALL_RESULTS =  pd.DataFrame(MOLS, columns=['SMILES']) 
        ALL_RESULTS['Distance'] = D

        print(ALL_RESULTS)
        if len(ALL_RESULTS) > 4:
        # Calculate fingerprints for all molecules
            FP_array = chemspace_potentials.get_all_FP( [Chem.MolFromSmiles(smi) for smi in ALL_RESULTS['SMILES'].values ]  , nBits=2048)
                                            
            # Perform PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(FP_array)
            pca_start = pca.transform(chemspace_potentials.get_all_FP([Chem.MolFromSmiles(smiles)], nBits=2048))

            # Add PCA results to DataFrame
            ALL_RESULTS['PCA1'] = pca_result[:,0]
            ALL_RESULTS['PCA2'] = pca_result[:,1]

        csv = ALL_RESULTS.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}" download="chemspace_sampler_results.csv">Download Results CSV File</a>'

        # Add download link to Streamlit
        st.markdown(href, unsafe_allow_html=True)

        # Assuming D contains distances and has the same length as MOLS
        D = D[:10]
        print(MOLS)
        # Convert MOLS to dataframe
        mol_df = pd.DataFrame(MOLS[:10], columns=['SMILES'])  # creating DataFrame from MOLS
        mol_df['Distance'] = D

        mol_df['img'] = mol_df['SMILES'].apply(lambda x: mol_to_img(mol_to_3d(Chem.MolFromSmiles(x))))
        mol_df['img'] = mol_df['img'].apply(lambda x: base64.b64encode(x).decode())
        st.image([BytesIO(base64.b64decode(img_str)) for img_str in mol_df['img']])
        rows = []
        for idx, row in mol_df.iterrows():
            rows.append({"SMILES": row["SMILES"], "Distance": row["Distance"]})

        table_data = pd.DataFrame(rows)

        plt.figure(figsize=(3, 3))
        plt.hist(ALL_RESULTS['Distance'].values, bins=20, color='skyblue', edgecolor='black')
        plt.title('Histogram of Distances', fontsize=12, weight='bold')
        plt.xlabel('$D$', fontsize=10)
        plt.ylabel('#', fontsize=10)
        st.pyplot(plt)
        st.write('Table of the 10 closest molecules, see above to download all results.')
        st.table(table_data)


        if len(ALL_RESULTS) > 4:
            # Use a diverging color palette, increase point transparency and change marker style, font size and weight
            st.write('PCA plot of all molecules (using ECFP4 fingerprints for speed)')
            plt.figure(figsize=(3, 3))
            other_mols = ALL_RESULTS[ALL_RESULTS['SMILES'] != smiles]
            scatter_plot = sns.scatterplot(data=other_mols, x='PCA1', y='PCA2', s=100, palette='coolwarm', hue='Distance', alpha=0.7, legend=False, marker='o')

            # Increase size of start molecule marker and its edge color for emphasis
            plt.scatter(pca_start[:,0], pca_start[:,1], color='red', edgecolor='black', marker='*', s=300, label='Start Molecule')
            # Create a custom legend for the start molecule


            plt.title('PCA of Molecular Fingerprints', fontsize=12, weight='bold')
            plt.xlabel('PCA1', fontsize=10)
            plt.ylabel('PCA2', fontsize=10)

            # Create a custom legend for the start molecule
            legend_marker = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, markeredgecolor='black')
            plt.legend(handles=[legend_marker], labels=['Start Molecule'], loc='upper right')
            # Create colorbar
            norm = Normalize(other_mols['Distance'].min(), other_mols['Distance'].max())
            sm = ScalarMappable(norm=norm, cmap='coolwarm')
            plt.colorbar(sm)
            # Remove top and right spines
            sns.despine()

            # Show the plot in Streamlit
            st.pyplot(plt.gcf())


    except Exception as e:
        
        st.error('An error occurred. Please check your input parameters and try again. \
                 Is the starting molecule consistent with the conditions i.e. number of heavy atoms, elements, synthesizability, etc.? \
                 Usually it means no molecules were found that satisfy the conditions.')
        print(e)
