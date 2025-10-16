# load_testdata.py
# Author: Isabel Vinterbladh
# Date: June 2025
# Description: Load test data for SAXS calculations.

import numpy as np
import pandas as pd


# Read in the poly coefficients
def read_poly_coefficients(file_path):
    df = pd.read_csv(file_path)
    return df[['Restype', 'Coefficient Index', 'Value']].to_numpy()

# List of standard amino acid three-letter codes
amino_acid_codes = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]


def read_profile_pepsi(out_path, log_path):
    with open(out_path, 'r') as file:
        lines = file.readlines()
    data = [line.split() for line in lines[6:]]
    pepsi_curve = pd.DataFrame(data, columns=[ "q", "Iq", "Iat", "Iev", "Ihs", "atev", "aths", "evhs"])
    with open( log_path, "r") as f:
        for line in f:
            if "Intensity scaling" in line:
                pepsi_scaling = float(line.split()[-1])

    for i in list(pepsi_curve.columns):
        if i !="q":
            pepsi_curve[i] = pepsi_curve[i].astype(float).to_numpy()/pepsi_scaling
  
    print("Scaling:", pepsi_scaling)
    Iq_tot = pepsi_curve["Iq"]
    Iat = pepsi_curve["Iat"]
    Iev = pepsi_curve["Iev"]
    Ihs = pepsi_curve["Ihs"]
    atev = pepsi_curve["atev"]
    aths = pepsi_curve["aths"]
    evhs = pepsi_curve["evhs"]
    q_pepsi = pepsi_curve["q"].astype(float).to_numpy()

    return Iq_tot, Iat, Iev, Ihs, atev, aths, evhs, q_pepsi


def load_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Remove line with number of atoms and comment line
    data = np.array([line.split() for line in lines[2:]])
    # Parse data into a DataFrame
    amino_acids = data[:, 0]
    coordinates = data[:, 1:].astype(float)
    return amino_acids, coordinates

def read_sasbdb(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = np.array([line.split() for line in lines[1:]])
    sasbdb_q = data[:,0].astype(float) # q values
    sasbdb_I = data[:,1].astype(float) # I(q) values
    sasbdb_Imean = data[:,2].astype(float) # mean value of I(q)
    return sasbdb_q, sasbdb_I, sasbdb_Imean