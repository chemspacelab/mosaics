import numpy as np
from copy import deepcopy
from math import comb,cos
import itertools
from itertools import combinations, product
from scipy.spatial.distance import cityblock, euclidean
from scipy.stats import wasserstein_distance
from joblib import Parallel, delayed
from itertools import combinations, product
from scipy.special import comb
from rdkit.Chem import DataStructs, rdMolDescriptors, Descriptors
try:
    from ase import Atoms
    from dscribe.descriptors import SOAP
except:
    print("local_space_sampling: ase or dscribe not installed")
import pdb
from mosaics.data import *
import numba
import pickle

def ExplicitBitVect_to_NumpyArray(fp_vec):
    """
    Convert the RDKit fingerprint to a numpy array.

    Parameters
    ----------
    fp_vec : rdkit.DataStructs.cDataStructs.ExplicitBitVect
        The RDKit fingerprint vector.

    Returns
    -------
    fp2 : numpy.ndarray
        The fingerprint vector converted to a numpy array.
    """
    fp2 = np.zeros((0,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp_vec, fp2)
    return fp2


def extended_get_single_FP(smi, nBits=2048, useFeatures=True):
    """
    Get the fingerprint of a molecule in numpy array form.

    Parameters
    ----------
    smi : str
        SMILES string of the molecule.
    nBits : int, optional
        Length of the fingerprint. Default is 2048.
    useFeatures : bool, optional
        Whether to use feature information when generating the fingerprint. Default is True.

    Returns
    -------
    x : numpy.ndarray
        The fingerprint of the molecule as a numpy array.
    """
    x = ExplicitBitVect_to_NumpyArray(
        get_single_FP(smi, nBits=nBits, useFeatures=useFeatures)
    )
    return x


def get_single_FP(mol, nBits=2048, useFeatures=True):
    """
    Compute the fingerprint of a molecule.

    Parameters
    ----------
    mol : str or rdkit.Chem.rdchem.Mol
        SMILES string or RDKit molecule object.
    nBits : int, optional
        Length of the fingerprint. Default is 2048.
    useFeatures : bool, optional
        Whether to use feature information when generating the fingerprint. Default is True.

    Returns
    -------
    fp_mol : rdkit.DataStructs.cDataStructs.ExplicitBitVect
        The fingerprint of the molecule.
    """
    fp_mol = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol,
        radius=4,
        nBits=nBits,
        useFeatures=useFeatures
    )
    return fp_mol

def tanimoto_similarity(vector1, vector2):
    intersection = np.sum(np.logical_and(vector1, vector2))
    union = np.sum(np.logical_or(vector1, vector2))
    tanimoto_similarity = intersection / union
    return tanimoto_similarity


def tanimoto_distance(vector1, vector2):
    similarity = tanimoto_similarity(vector1, vector2)
    distance = 1 - similarity
    return distance


def get_all_FP(RDKIT_MOLS, **kwargs):
    """
    Return a list of fingerprints for all the molecules in the list of rdkit molecules.
    Parameters
    ----------
    SMILES : list of rdkit
        List of rdkit molecules.

    Returns
    -------
    X : numpy.ndarray
        An array of fingerprints for all the molecules.
    """
    X = []
    for mol in RDKIT_MOLS:
        X.append(extended_get_single_FP(mol, **kwargs))
    return np.array(X)



def generate_CM(cood,charges,pad):
    size=len(charges)
    cm=np.zeros((pad,pad))
    for i in range(size):
        for j in range(size):
            if i==j:
                cm[i,j]=0.5*(charges[i]**(2.4))
            else:
                dist=np.linalg.norm(cood[i,:]-cood[j,:])
                
                cm[i,j]=(charges[i]*charges[j])/dist

    # calculate the norms of the rows
    row_norms = np.linalg.norm(cm, axis=1)
    
    # get the indices that would sort the row norms
    sorted_indices = np.argsort(row_norms)[::-1]
    
    # use these indices to sort the cm matrix
    sorted_cm = cm[sorted_indices]
    
    return sorted_cm



def generate_bob(elements,coords,asize={'C': 12, 'H': 24, 'N': 6, 'O': 6, 'F':5}):
    """
    generates the Bag of Bonds representation
    :param elements: array of arrays of chemical element symbols for all molecules in the dataset
    :type elements: numpy array NxM, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    :param coords: array of arrays of input coordinates of the atoms
    :type coords: numpy array NxMx3, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    :param asize: The maximum number of atoms of each element type supported by the representation
    :type asize: dictionary

    :return: NxD array of D-dimensional BoB vectors for the N molecules
    :rtype: numpy array
    """
    
    elements, coords = [elements], [coords]
    bob_arr = [bob(atoms, coords, asize) for atoms, coords in zip(elements, coords)]

    return np.array(bob_arr)[0]

def bob(atoms, coods, asize={'C': 7, 'H': 16, 'N': 3, 'O': 3, 'S': 1}):
    keys = list(asize.keys())
    elements = {'C': [[], 6], 'H': [[], 1], 'N': [[], 7], 'O': [[], 8], 'F': [[], 9], 'P': [[], 15], 'S': [[], 16],
                'Cl': [[], 17], 'Br': [[], 35], 'I': [[], 53], 'Si': [[], 14], 'B': [[], 5]}
    for i in range(len(atoms)):
        elements[atoms[i]][0].append(coods[i])

    bob = []
    for i in range(len(keys)):
        num = len(elements[keys[i]][0])
        if num != 0:
            bag = np.zeros((asize[keys[i]]))
            bag[:num] = 0.5 * (elements[keys[i]][1] ** 2.4)
            bag = -np.sort(-bag)
            bob.extend(bag)
            for j in range(i, len(keys)):
                if i == j:
                    z = elements[keys[i]][1]
                    bag = np.zeros((int(comb(asize[keys[i]], 2))))
                    vec = []
                    for (r1, r2) in combinations(elements[keys[i]][0], 2):
                        vec.append(z ** 2 / np.linalg.norm(r1 - r2))
                    bag[:len(vec)] = vec
                    bag = -np.sort(-bag)
                    bob.extend(bag)
                elif (i != j) and (len(elements[keys[j]][0]) != 0):
                    z1, z2 = elements[keys[i]][1], elements[keys[j]][1]
                    bag = np.zeros((asize[keys[i]] * asize[keys[j]]))
                    vec = []
                    for (r1, r2) in product(elements[keys[i]][0], elements[keys[j]][0]):
                        vec.append(z1 * z2 / np.linalg.norm(r1 - r2))
                    bag[:len(vec)] = vec
                    bag = -np.sort(-bag)
                    bob.extend(bag)
                else:
                    bob.extend(np.zeros((asize[keys[i]] * asize[keys[j]])))
        else:
            bob.extend(np.zeros((asize[keys[i]])))
            for j in range(i, len(keys)):
                if i == j:
                    bob.extend(np.zeros((int(comb(asize[keys[i]], 2)))))
                else:
                    bob.extend(np.zeros((asize[keys[i]] * asize[keys[j]])))
    return np.array(bob)





def calc_all_descriptors(mol):
    descriptors = {}
    for descriptor_name in Descriptors.descList:
        descriptor_function = descriptor_name[1]
        descriptors[descriptor_name[0]] = descriptor_function(mol)
    return np.array(list(descriptors.values()))



def gen_soap(crds, chgs, species):
    """
    Generate the Smooth Overlap of Atomic Positions (SOAP) descriptor for a molecule.

    Args:
        crds: A list or array of atomic coordinates.
        chgs: A list or array of atomic charges.
        species: A list of species for atoms present in the molecule.
    
    Returns:
        A SOAP descriptor for the molecule.

    Note:
        The average output is a global of the molecule. See https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html for more details.
    """
    average_soap = SOAP(
        r_cut=6.0,
        n_max=8,
        l_max=6,
        average="inner",
        species=species,
        sparse=False,
    )

    molecule = Atoms(numbers=chgs, positions=crds)

    return average_soap.create(molecule)







root2,ipi=2**0.5,np.pi*1j
half_rootpi=(np.pi**0.5)/2
c1,c2,c3=4.08858*(10**12),(np.pi**0.5)/2,(np.pi**0.5)*np.exp(-0.25)*1j/4
c4=-1j*(np.pi**0.5)*np.exp(-1/8)/(4*root2)
a2b = 1.88973


@numba.jit(nopython=True)
def erfunc(z):
    t = 1.0 / (1.0 + 0.5 * np.abs(z))
    ans = 1 - t * np.exp( -z*z -  1.26551223 +
                        t * ( 1.00002368 +
                        t * ( 0.37409196 + 
                        t * ( 0.09678418 + 
                        t * (-0.18628806 + 
                        t * ( 0.27886807 + 
                        t * (-1.13520398 + 
                        t * ( 1.48851587 + 
                        t * (-0.82215223 + 
                        t * ( 0.17087277))))))))))
    return ans


@numba.jit(nopython=True)
def hermite_polynomial(x, degree, a=1):
    if degree == 0:
        return 1
    elif degree == 1:
        return -2*a*x
    elif degree == 2:
        x1 = (a*x)**2
        return 4*x1 - 2*a
    elif degree == 3:
        x1 = (a*x)**3
        return -8*x1 - 12*a*x
    elif degree == 4:
        x1 = (a*x)**4
        x2 = (a*x)**2
        return 16*x1 - 48*x2 + 12*a**2


@numba.jit(nopython=True)
def generate_data(size,z,atom,charges,coods,cutoff_r=12):
    """
    returns 2 and 3-body internal coordinates
    """
    
    twob=np.zeros((size,2))
    threeb=np.zeros((size,size,5))
    z1=z**0.8

    for j in range(size):
        rij=atom-coods[j]
        rij_norm=np.linalg.norm(rij)

        if rij_norm!=0 and rij_norm<cutoff_r:
            z2=charges[j]**0.8
            #twob[j]=rij_norm,z*charges[j]
            #I changed the following two lines to make it compatible with numba
            twob[j, 0] = rij_norm
            twob[j, 1] = z*charges[j]

            for k in range(size):
                if j!=k:
                    rik=atom-coods[k]
                    rik_norm=np.linalg.norm(rik)

                    if rik_norm!=0 and rik_norm<cutoff_r:
                        z3=charges[k]**0.8
                        
                        rkj=coods[k]-coods[j]
                        
                        rkj_norm=np.linalg.norm(rkj)
                        
                        threeb[j][k][0] = np.minimum(1.0,np.maximum(np.dot(rij,rik)/(rij_norm*rik_norm),-1.0))
                        threeb[j][k][1] = np.minimum(1.0,np.maximum(np.dot(rij,rkj)/(rij_norm*rkj_norm),-1.0))
                        threeb[j][k][2] = np.minimum(1.0,np.maximum(np.dot(-rkj,rik)/(rkj_norm*rik_norm),-1.0))
                        
                        atm = rij_norm*rik_norm*rkj_norm
                        
                        charge = z1*z2*z3
                        
                        #threeb[j][k][3:] =  atm, charge
                        #I changed the following two lines to make it compatible with numba
                        threeb[j, k, 3] = atm
                        threeb[j, k, 4] = charge

    return twob, threeb                        

@numba.jit(nopython=True)
def angular_integrals(size,threeb,alength=158,a=2,grid1=None,grid2=None,angular_scaling=2.4):
    """
    evaluates the 3-body functionals using the trapezoidal rule
    """

    arr=np.zeros((alength,2))
    theta=0
    
    for i in range(alength):
        f1,f2=0,0
        num1,num2=grid1[i],grid2[i]
        
        for j in range(size):

            for k in range(size):

                if threeb[j][k][-1]!=0:
                    
                    angle1,angle2,angle3,atm,charge=threeb[j][k]

                    x=theta-np.arccos(angle1)

                    exponent,h1=np.exp(-a*x**2),hermite_polynomial(x,1,a)
                    
                    f1+=(charge*exponent*num1)/(atm**4)
                    
                    f2+=(charge*h1*exponent*(1+(num2*angle1*angle2*angle3)))/(atm**angular_scaling)
        
        arr[i]=f1,f2
        theta+=0.02

    trapz=[np.trapz(arr[:,i],dx=0.02) for i in range(arr.shape[1])]

    return trapz


@numba.jit(nopython=True)
def radial_integrals(size,rlength,twob,step_r,a=1,normalized=False):
    """
    evaluates the 2-body functionals using the trapezoidal rule
    """
    
    arr=np.zeros((rlength,4))
    r=0
    
    for i in range(rlength):
        f1,f2,f3,f4=0,0,0,0
        
        for j in range(size):

            if twob[j][-1]!=0:
                dist,charge=twob[j]
                x=r-dist

                if normalized==True:
                    norm=(erfunc(dist)+1)*half_rootpi
                    exponent=np.exp(-a*(x)**2)/norm
                
                else:
                    exponent=np.exp(-a*(x)**2)

                h1,h2=hermite_polynomial(x,1,a),hermite_polynomial(x,2,a)
                
                f1+=charge*exponent*np.exp(-10.8*r)
                
                f2+=charge*exponent/(2.2508*(r+1)**3)
                
                f3+=charge*(h1*exponent)/(2.2508*(r+1)**6)
                
                f4+=charge*h2*exponent*np.exp(-1.5*r) 
        
        r+=step_r
        arr[i]=f1,f2,f3,f4
    
    trapz=[np.trapz(arr[:,i],dx=step_r) for i in range(arr.shape[1])]

    return trapz


@numba.jit(nopython=True)
def mbdf_local(charges,coods,grid1,grid2,rlength,alength,pad=29,step_r=0.1,cutoff_r=12,angular_scaling=2.4):
    """
    returns the local MBDF representation for a molecule
    """
    size = len(charges)
    mat=np.zeros((pad,6))
    
    assert size > 1, "No implementation for monoatomics"

    if size>2:
        for i in range(size):

            twob,threeb = generate_data(size,charges[i],coods[i],charges,coods,cutoff_r)

            mat[i][:4] = radial_integrals(size,rlength,twob,step_r)     

            mat[i][4:] = angular_integrals(size,threeb,alength,grid1=grid1,grid2=grid2,angular_scaling=angular_scaling)

    elif size==2:
        z1, z2, rij = charges[0]**0.8, charges[1]**0.8, coods[0]-coods[1]
        
        pref, dist = z1*z2, np.linalg.norm(rij)
        
        twob = np.array([[pref, dist], [pref, dist]])
        
        mat[0][:4] = radial_integrals(size,rlength,twob,step_r)

        mat[1][:4] = mat[0][:4]

    return mat


def mbdf_global(charges,coods,asize,rep_size,keys,grid1,grid2,step_r=0.1,step_a=0.02,cutoff_r=8.0,angular_scaling=4):
    """
    returns the flattened, bagged MBDF feature vector for a molecule
    """
    elements = {k:[[],k] for k in keys}

    a2b = 1.88973

    rlength = int(cutoff_r/step_r) + 1
    alength = int(np.pi/step_a) + 1

    coods, cutoff_r = a2b*coods, a2b*cutoff_r

    size = len(charges)

    for i in range(size):
        elements[charges[i]][0].append(coods[i])

    mat, ind = np.zeros((rep_size,6)), 0

    assert size > 1, "No implementation for monoatomics"

    if size>2:

        for key in keys:
            
            num = len(elements[key][0])
            
            if num!=0:
                bags = np.zeros((num,6))
                
                for j in range(num):
                    twob,threeb = generate_data(size,key,elements[key][0][j],charges,coods,cutoff_r)

                    bags[j][:4] = radial_integrals(size,rlength,twob,step_r)     

                    bags[j][4:] = angular_integrals(size,threeb,alength,grid1=grid1,grid2=grid2,angular_scaling=angular_scaling)

                mat[ind:ind+num] = -np.sort(-bags,axis=0)
                
            ind += asize[key]
    
    elif size == 2:

        for key in keys:
            
            num = len(elements[key][0])
            
            if num!=0:
                bags = np.zeros((num,6))
                
                for j in range(num):
                    z1, z2, rij = charges[0]**0.8, charges[1]**0.8, coods[0]-coods[1]
        
                    pref, dist = z1*z2, np.linalg.norm(rij)

                    twob = np.array([[pref, dist], [pref, dist]])
                    
                    bags[j][:4] = radial_integrals(size,rlength,twob,step_r)     

                mat[ind:ind+num] = -np.sort(-bags,axis=0)
                
            ind += asize[key]

    return mat.ravel(order='F')  

@numba.jit(nopython=True)
def fourier_grid():
    
    angles = np.arange(0,np.pi,0.02)
    
    grid1 = np.cos(angles)
    grid2 = np.cos(2*angles)
    grid3 = np.cos(3*angles)
    
    return (3+(100*grid1)+(-200*grid2)+(-164*grid3),grid1)


@numba.jit(nopython=True)
def normalize(A,normal='mean'):
    """
    normalizes the functionals based on the given method
    """
    
    A_temp = np.zeros(A.shape)
    
    if normal=='mean':
        for i in range(A.shape[2]):
            
            avg = np.mean(A[:,:,i])

            if avg!=0.0:
                A_temp[:,:,i] = A[:,:,i]/avg
            
            else:
                pass
   
    elif normal=='min-max':
        for i in range(A.shape[2]):
            
            diff = np.abs(np.max(A[:,:,i])-np.min(A[:,:,i]))
            
            if diff!=0.0:
                A_temp[:,:,i] = A[:,:,i]/diff
            
            else:
                pass
    
    return A_temp




def generate_mbdf(nuclear_charges,coords,local=True,n_jobs=-1,pad=None,step_r=0.1,cutoff_r=8.0,step_a=0.02,angular_scaling=4,normalized='min-max',progress_bar=False):
    """
    Generates the local MBDF representation arrays for a set of given molecules

    :param nuclear_charges: array of arrays of nuclear_charges for all molecules in the dataset
    :type nuclear_charges: numpy array NxM, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    :param coords : array of arrays of input coordinates of the atoms
    :type coords: numpy array NxMx3, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    ordering of the molecules in the nuclear_charges and coords arrays should be consistent
    :param n_jobs: number of cores to parallelise the representation generation over. Default value is -1 which uses all available cores in the system
    :type n_jobs: integer
    :param pad: Number of atoms in the largest molecule in the dataset. Can be left to None and the function will calculate it using the nuclear_charges array
    :type pad: integer
    :param step_r: radial step length in Angstrom
    :type step_r: float
    :param cutoff_r: local radial cutoff distance for each atom
    :type cutoff_r: float
    :param step_a: angular step length in Radians
    :type step_a: float
    :param angular_scaling: scaling of the inverse distance weighting used in the angular functionals
    :type : float
    :param normalized: type of normalization to be applied to the functionals. Available options are 'min-max' and 'mean'. Can be turned off by passing False
    :type : string
    :param progress: displays a progress bar for representation generation process. Requires the tqdm library
    :type progress: Bool

    :return: NxPadx6 array containing Padx6 dimensional MBDF matrices for the N molecules
    """
    assert nuclear_charges.shape[0] == coords.shape[0], "charges and coordinates array length mis-match"
    
    lengths, charges = [], []

    for i in range(len(nuclear_charges)):
        
        q, r = nuclear_charges[i], coords[i]
        
        assert q.shape[0] == r.shape[0], "charges and coordinates array length mis-match for molecule at index" + str(i)

        lengths.append(len(q))

        charges.append(q.astype(np.float64))

    if pad==None:
        pad = max(lengths)

    charges = np.array(charges)

    rlength = int(cutoff_r/step_r) + 1
    alength = int(np.pi/step_a) + 1

    grid1,grid2 = fourier_grid()
    
    coords, cutoff_r = a2b*coords, a2b*cutoff_r

    if local:
        if progress_bar==True:

            from tqdm import tqdm    
            mbdf = Parallel(n_jobs=n_jobs)(delayed(mbdf_local)(charge,cood,grid1,grid2,rlength,alength,pad,step_r,cutoff_r,angular_scaling) for charge,cood in tqdm(list(zip(charges,coords))))

        else:
            mbdf = Parallel(n_jobs=n_jobs)(delayed(mbdf_local)(charge,cood,grid1,grid2,rlength,alength,pad,step_r,cutoff_r,angular_scaling) for charge,cood in zip(charges,coords))

        mbdf=np.array(mbdf)

        if normalized==False:

            return mbdf

        else:

            return normalize(mbdf,normal=normalized)
        
    else:
        keys = np.unique(np.concatenate(charges))

        asize = {key:max([(mol == key).sum() for mol in charges]) for key in keys}

        rep_size = sum(asize.values())

        if progress_bar==True:

            from tqdm import tqdm    
            arr = Parallel(n_jobs=n_jobs)(delayed(mbdf_global)(charge,cood,asize,rep_size,keys,grid1,grid2,rlength,alength,step_r,cutoff_r,angular_scaling) for charge,cood in tqdm(list(zip(charges,coords))))

        else:
            arr = Parallel(n_jobs=n_jobs)(delayed(mbdf_global)(charge,cood,asize,rep_size,keys,grid1,grid2,rlength,alength,step_r,cutoff_r,angular_scaling) for charge,cood in zip(charges,coords))

        arr = np.array(arr)

        if normalized==False:

            mbdf = np.array([mat.ravel(order='F') for mat in arr])
            
            return mbdf

        else:

            arr = normalize(arr,normal=normalized)

            mbdf = np.array([mat.ravel(order='F') for mat in arr])
            
            return mbdf


@numba.jit(nopython=True)
def wKDE(rep,bin,bandwidth,kernel,scaling=False):
    """
    returns the weighted kernel density estimate for a given array and bins
    """
    if kernel=='gaussian':
        if scaling=='root':
            a = bin.reshape(-1,1)-rep
            
            basis = np.exp(-(a**2)/bandwidth)
            
            k = (np.sqrt(np.abs(rep)))*basis
            
            return np.sum(k,axis=1)

        else:
            a = bin.reshape(-1,1)-rep
            
            basis = np.exp(-(a**2)/bandwidth)
            
            return np.sum(basis,axis=1)

    elif kernel=='laplacian':
        if scaling=='root':
            a = bin.reshape(-1,1)-rep
            
            basis = np.exp(-(np.abs(a))/bandwidth)
            
            k = (np.abs(rep))*basis
            
            return np.sum(k,axis=1)

        else:
            a = bin.reshape(-1,1)-rep
            
            basis = np.exp(-(np.abs(a))/bandwidth)
            
            return np.sum(basis,axis=1)


def density_estimate(reps,nuclear_charges,keys,bin,bandwidth,kernel='gaussian',scaling='root'):
    """
    returns the density functions of MBDF functionals for a set of given molecules.
    """
    
    size=len(bin)
    big_rep=np.zeros((reps.shape[0],size*len(keys)))
    

    if kernel=='gaussian':
        for i in range(len(nuclear_charges)):

            for j,k in enumerate(keys):
                ii = np.where(nuclear_charges[i] == k)[0]

                if len(ii)!=0:
                    big_rep[i,j*size:(j+1)*size]=wKDE(reps[i][ii]/k,bin,bandwidth,kernel,scaling)

                else:
                    big_rep[i,j*size:(j+1)*size]=np.zeros(size)

    return big_rep


def generate_df(mbdf,nuclear_charges,bw=0.07,binsize=0.2,kernel='gaussian'):
    """
    Generates the Density of Functionals representation for a given set of molecules. Requires their MBDF arrays as input
    
    :param mbdf: array of arrays containing the MBDF representation matrices for all molecules in the dataset
    :type mbdf: numpy array, output of the generate_mbdf function can be directly used here
    :param nuclear_charges: array of arrays of nuclear_charges for all molecules in the dataset, should be in the same order as in the MBDF arrays
    :type nuclear_charges: numpy array NxM, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    :param bw: the bandwidth hyper-parameter of the kernel density estimate
    :type bw: float
    :param binsize: grid-spacing used for discretizing the density function
    :type binsize: float
    :param kernel: kernel function to be used in the kernel density estimation
    :type kernel: string

    :return: NxM array containing the M dimensional representation vectors for N molecules
    """
    fs=mbdf.shape[-1]

    reps=[10*mbdf[:,:,i]/(np.max(np.abs(mbdf[:,:,i]))) for i in range(fs)]
    
    keys=np.unique(np.concatenate(nuclear_charges))
    
    bin=np.arange(-10,10,binsize)
    
    gridsize=len(keys)*len(bin)
    
    kde=np.zeros((mbdf.shape[0],gridsize*fs))
    
    for i in range(fs):
        kde[:,i*gridsize:(i+1)*gridsize]=density_estimate(reps[i],nuclear_charges,keys,bin,bw,kernel)
    
    return kde

@numba.jit(nopython=True)
def generate_CM(cood,charges,pad):
    size=len(charges)
    cm=np.zeros((pad,pad))
    for i in range(size):
        for j in range(size):
            if i==j:
                cm[i,j]=0.5*(charges[i]**(2.4))
            else:
                dist=np.linalg.norm(cood[i,:]-cood[j,:])
                
                cm[i,j]=(charges[i]*charges[j])/dist
    summation = np.array([sum(x**2) for x in cm])
    sorted_mat = cm[np.argsort(summation)[::-1,],:]    
    return sorted_mat.ravel()


def pad_vector(vec, N=10000):
    # Ensure the input is a numpy array
    vec = np.asarray(vec)
    
    # Check if the input shape is 2D and the second dimension is less than N
    if vec.ndim == 2 and vec.shape[1] < N:
        # If so, pad the second dimension with zeros
        return np.pad(vec, ((0, 0), (0, N - vec.shape[1])), 'constant')

    # If the input is larger than N, slice it to size N
    elif vec.shape[1] > N:
        return vec[:,:N]

    else:
        return vec
    

def global_MBDF_bagged_wrapper(charges,coords, params):
    """
    Wrapper function for generating the global MBDF representation for a molecule.
    """
    rep_size =  params['max_n']
    cutoff_r =  a2b*8.0
    grid1,grid2 = params['grid1'],params['grid2']
    coords = a2b*coords
    asize2 = params['asize2']

    
    return mbdf_global(charges,coords,asize2,rep_size,asize2.keys(),grid1,grid2,cutoff_r=cutoff_r)



class AtomicEnergyRepresentation():
    
    def __init__(self, fragment_dictionary):
        self.mean_energy_by_fragment = fragment_dictionary
        self.unique_charges = sorted(list(self.mean_energy_by_fragment.keys()))
        
    def get_pattern_from_ase(self, cutoff_radius = 1.6):
        # pairwise distances for all atoms
        distances = self.mol.get_all_distances()
        pattern = [] # patterns of atoms
        # d is the pairwise distance of an atom with Znuc, we check which elements have to the lowest distance to the atom
        # idx is the index we use to get the respective atomic energy
        for i ,d in enumerate(distances):
            ind_nn = np.where((d < cutoff_radius) & (d > 0))
            Znuc_nn = self.mol.get_atomic_numbers()[ind_nn]
            pattern.append(self.pattern_from_nuccharge(Znuc_nn))
        self.pattern = pattern
        return(pattern)
    
    def pattern_from_nuccharge(self, Z_nn):
        Z_nn.sort()
        # four partners
        if len(Z_nn) == 4:
            charges = itertools.combinations_with_replacement([1,6,7,8], 4)
            string = itertools.combinations_with_replacement('HCNO', 4)
            for c, s in zip(charges, string):
                if np.array_equal(Z_nn, np.array(c)):
                    concatenated = ''
                    for el in s:
                        concatenated += el
                    return(concatenated)
            if np.array_equal(Z_nn, np.array([6,9,9,9])):
                return('CFFF')
        # three partners
        elif len(Z_nn) == 3:
            charges = itertools.combinations_with_replacement([1,6,7,8], 3)
            string = itertools.combinations_with_replacement('HCNO', 3)
            for c, s in zip(charges, string):
                if np.array_equal(Z_nn, np.array(c)):
                    concatenated = ''
                    for el in s:
                        concatenated += el
                    return(concatenated)
        # two partners    
        elif len(Z_nn) == 2:
            charges = itertools.combinations_with_replacement([1,6,7,8], 2)
            string = itertools.combinations_with_replacement('HCNO', 2)
            for c, s in zip(charges, string):
                if np.array_equal(Z_nn, np.array(c)):
                    concatenated = ''
                    for el in s:
                        concatenated += el
                    return(concatenated)
        # one partner
        elif len(Z_nn) == 1:
            charges = itertools.combinations_with_replacement([1,6,7,8], 1)
            string = itertools.combinations_with_replacement('HCNO', 1)
            for c, s in zip(charges, string):
                if np.array_equal(Z_nn, np.array(c)):
                    concatenated = ''
                    for el in s:
                        concatenated += el
                    return(concatenated)
        else:
            raise ValueError(f'Cannot identify binding partners for {Z_nn}')
    
    def generate_repsentation(self, mol, num_atoms):
        self.mol = mol
        # find binding partners for each atom
        fragments = self.get_pattern_from_ase() 
        # get mean atomic energy for each atom depending on its environment
        # elementwise_vectors = {Z:np.zeros(num_atoms[Z]) for Z in self.unique_charges}
        # index_counter = {Z:0 for Z in self.unique_charges}
        elementwise_vectors = {Z:[] for Z in self.unique_charges}
        for Z, frag in zip(self.mol.get_atomic_numbers(), fragments):
            try:
                elementwise_vectors[Z].append(self.mean_energy_by_fragment[Z][frag])
            except KeyError:
                print(f'No atomic energy for {frag} available, returning nan')
                return(None)
            
        # sort atomic energies by value and pad with zeros
        elementwise_vectors = self.sort_and_add_padding(elementwise_vectors, num_atoms)
            
        # concatenate elementwise vectors to single vector and transform in numpy-array
        representation_vector = self.construct_vector(elementwise_vectors)
        
        return(representation_vector)
    
    def sort_and_add_padding(self, elementwise_vectors, num_atoms):
        for Z in elementwise_vectors.keys():
            elementwise_vectors[Z].sort()
            assert len(elementwise_vectors[Z]) <= num_atoms[Z], f'Number of atoms (={len(elementwise_vectors[Z])}) with Z = {Z} exceeds maximum number of atoms (={num_atoms[Z]})'
            elementwise_vectors[Z] = elementwise_vectors[Z] + (num_atoms[Z] - len(elementwise_vectors[Z]))*[0.0]
        return(elementwise_vectors)
    
    def construct_vector(self, elementwise_vectors):
        vector = []
        for Z in elementwise_vectors.keys():
            vector.extend(elementwise_vectors[Z])
        return(np.array(vector))
    
def gen_atomic_energy_rep(charges, positions, num_atoms, rep_dict="atomic_energy"):
     
    # # Table with mean energies from alchemy
    # with open("./alchemy_mean_energy_lookup.pkl", "rb") as file:
    #     mean_energy_dict = pickle.load(file)
        
    # # Table with mean energies from IQA
    # with open("./IQA_mean_energy_lookup.pkl", "rb") as file:
    #     mean_energy_dict = pickle.load(file)

    # Table with mean energies from MO-decomposition
    if rep_dict == "atomic_energy": 
        #with open("./atomic_dict/mf_mean_energy_lookup.pkl", "rb") as file:
        with open("./atomic_dict/alchemy_mean_energy_lookup.pkl", "rb") as file:
            mean_energy_dict = pickle.load(file)
    else:
        print("No valid representation type selected")
    

    # # Table with mean energies from schnet
    # with open("./schnet_mean_energy_lookup.pkl", "rb") as file:
    #     mean_energy_dict = pickle.load(file)
        
    # initialize representation generator
    RepGenerator = AtomicEnergyRepresentation(mean_energy_dict)

    """
    """

    # generate ase atoms object
    mol = Atoms(numbers = charges, positions=positions) 
    # max number of atoms per element, key = nuclear charge, value = max number of atoms
    
    # generate representation for molecule
    # elements are ordered by increasing charge
    # for each element atomic energie are ordered by increasing value
    rep = RepGenerator.generate_repsentation(mol, num_atoms)
    
    return rep