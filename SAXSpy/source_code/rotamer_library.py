#rotamer_library.py
# This script loads the Dunbrack rotamer library and extracts the coordinates of all atoms for a given residue type.
# The coordinates are stored in a NumPy array, where the first dimension corresponds to the rotamer index, the second
# dimension corresponds to the atom index, and the third dimension corresponds to the Cartesian coordinates.
# The script also extracts the atom names and the probabilities of each rotamer.
# The script is intended to be used as a module in other scripts.

import matplotlib
import sys
import matplotlib.pyplot as plt
import tqdm
import collections
import os
import pyrosetta as pr
import numpy as np
import pandas as pd

restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
chi_angles_mask = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 1.0, 0.0, 0.0],  # PRO
    [1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 0.0, 0.0, 0.0],  # VAL
]

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
restypes3 = [v for v in restype_1to3.values()]

def load_rotamor_library(libname, PATH):
    # Loads the rotamor library
    amino_acids = [
        "arg",
        "asp",
        "asn",
        "cys",
        "glu",
        "gln",
        "his",
        "ile",
        "leu",
        "lys",
        "met",
        "phe",
        "pro",
        "ser",
        "thr",
        "trp",
        "tyr",
        "val",
    ]
    db = {}

    columns = collections.OrderedDict()
    columns["T"] = str
    columns["Phi"] = np.int64
    columns["Psi"] = np.int64
    columns["Count"] = np.int64
    columns["r1"] = np.int64
    columns["r2"] = np.int64
    columns["r3"] = np.int64
    columns["r4"] = np.int64
    columns["Probabil"] = np.float64
    columns["chi1Val"] = np.float64
    columns["chi2Val"] = np.float64
    columns["chi3Val"] = np.float64
    columns["chi4Val"] = np.float64
    columns["chi1Sig"] = np.float64
    columns["chi2Sig"] = np.float64
    columns["chi3Sig"] = np.float64
    columns["chi4Sig"] = np.float64

    for amino_acid in amino_acids:
        db[amino_acid] = pd.read_csv(
            os.path.join(PATH, f"{libname}/{amino_acid}.bbdep.rotamers.lib"),
            names=list(columns.keys()),
            dtype=columns,
            comment="#",
            delim_whitespace=True,
            engine="c",
        )

    return db

def names_from_pose(pose):
    res= pose.residue(1)
    atom_names = []
    num_atoms = pose.total_atoms()
    for i in range(1, num_atoms + 1):
        atom_names.append(res.atom_name(i))
    return atom_names

def coords_from_pose(pose):
    res= pose.residue(1)

    # Create an empty list to store the coordinates
    coordinates = []
    num_atoms = pose.total_atoms()

    # Loop through all atoms and get the Cartesian coordinates
    for i in range(1, num_atoms + 1):  # Atom indices in PyRosetta are 1-based
        atom = res.atom(i)
        # Get the Cartesian coordinates of the atom
        coord = atom.xyz()  # This gives a PyRosetta XYZ object
        coordinates.append([coord.x, coord.y, coord.z])  # Convert to list

    # Convert the list to a NumPy array
    return np.array(coordinates)

def all_atom_coordinates_from_restype(restype, db):
    residx = restypes.index(restype)
    print(residx)
    restype3 = restype_1to3[restype]
    print(restype3.lower())
    num_chi = int(sum(chi_angles_mask[residx]))
    print(db[restype3.lower()])
    db_res = db[restype3.lower()]
    pose = pr.pose_from_sequence(restype)


    db_res["Normalized_Probabil"] = db_res.groupby(["Phi", "Psi"])["Probabil"].transform(lambda x: x / x.sum())
    db_res['bb_idx'] = pd.factorize(db_res[['Phi', 'Psi']].apply(tuple, axis=1))[0]

    names = names_from_pose(pose)
    all_coordinates=[]
    for rotamer  in tqdm.tqdm(db_res.itertuples()):
        pose.set_phi(1, rotamer.Phi)
        pose.set_psi(1, rotamer.Psi)
        for i in range(num_chi):
            pose.set_chi(1+1, 1, rotamer.__getattribute__("chi%iVal"%(i+1)))

        coords = coords_from_pose(pose)
        all_coordinates.append(coords)

    all_coordinates = np.array(all_coordinates)
    probs = np.array(db_res["Normalized_Probabil"])
    return all_coordinates, names, probs


