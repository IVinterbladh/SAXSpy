# debyecalc.py
# Author: Isabel Vinterbladh
# Date: June 2024
# Description: Calculate the scattering intensity I(q) using the Debye formula for a protein structure.

#import voronotalt as voro
import numpy as np
import matplotlib.pyplot as plt


class DebyeCalculator:
    def __init__(self, q_min=0.0, q_max=0.5, num_q_points=101):
        self.q_min = q_min
        self.q_max = q_max
        self.num_q_points = num_q_points    

    def calculate_Iq(self, structure_file, poly_ff, solvent_coords, Explicit_dummy=False): #solvent_file=None,
        """ Calculate the scattering intensity I(q) using the Debye formula.
        Args:
            structure_file (str): Path to the protein structure file.
            poly_ff (np.array): Polynomial coefficients for form factors.
            solvent_file (str): Path to the solvent structure file.
            Explicit_dummy (bool): Whether to use amino acid volumes for explicit solvent particles.
        Returns:
            q_values (np.array): Array of q values.
            Iq_values (np.array): Corresponding I(q) values.
        """
        # Load the protein structure
        amino_code, structure = self.load_structure(structure_file)
        # Load solvent structure and reduce points
        water_struct = solvent_coords
        # Generate q values
        q_values = np.linspace(self.q_min, self.q_max, self.num_q_points)
        # If using amino acid volumes from voronota, adjust form factors
        if Explicit_dummy:
             # Calculate amino acid volumes
            aa_volumes = self.amino_acid_volume(structure, amino_code)
            aa_FormFactors = np.array([self.getfitted_ff(aa, q_values, poly_ff) - self.overallexpansion_factor(q_values, aa_volumes[i])*self.dummyFactor(q_values, aa_volumes[i]) for i, aa in enumerate(amino_code)])
        else:
            aa_FormFactors = np.array([self.getfitted_ff(aa, q_values, poly_ff) for aa in amino_code])
        # Get water form factors
        water_FormFactors = np.array([self.getfitted_ff('H2O', q_values, poly_ff) for w in range(len(water_struct))])
        # Calculate I(q) using the Debye formula
        Iq_values = self.debye_formula(structure, q_values, aa_FormFactors, water_struct, water_FormFactors)
        return q_values, Iq_values

    def plot_Iq(self, q_values, Iq_values, q_pepsi, Iq_pepsi, q_exp, Iq_exp, I_expmean, label='Protein'):
        plt.figure(figsize=(8, 6))
        plt.plot(q_exp, Iq_exp, alpha =0.5, label="SASBDB - Experimental data")
        plt.plot(q_exp, I_expmean, label="SASBDB - Mean of Experimental data")
        plt.plot(q_values, Iq_values, label="SAXSpy")
        plt.plot(q_pepsi, Iq_pepsi, label="PepsiSAXS")
        #plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('q (1/Å)')
        plt.ylabel('I(q)')
        plt.title('Scattering Intensity I(q) of ' + label)
        plt.legend()
        plt.show()

    def load_structure(self, file_path):
        """ Load the protein structure from a file.
        Args:
            file_path (str): Path to the structure file.
        Returns:
            amino_code (list): List of amino acid codes.
            df_struct (np.array): Array of amino acid coordinates.
        """
        # Load the protein structure from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        data =np.array([line.split() for line in lines[2:]])
        amino_code = data[:, 0]
        df_struct = data[:, 1:].astype(float)
        return amino_code, df_struct
    
    def getfitted_ff(self, amino_acid, q, poly_ff):
        """ Get the fitted form factor for a given amino acid
            q is the scattering vector
        Args:
            amino_acid (str): Amino acid
            q (np.array): scattering vector
            poly_ff (np.array): polynomial coefficients for each amino acid
        returns:
            result (np.array): fitted form factor
        """
        result = np.zeros(len(q))
        poly = poly_ff[poly_ff[:,0]==amino_acid,2]
        result = poly[0]+poly[1]*q+poly[2]*q**2+poly[3]*q**3+poly[4]*q**4+poly[5]*q**5+poly[6]*q**6
        return result

    def dummyFactor(self, q, V):
        """ Dummy factor for testing purposes. """
        q2 = q*q/(4*np.pi**2)
        return 0.334 * V * np.exp(-q2 * pow(V, 2/3))

    def overallexpansion_factor(self, q, V):
        q2 = q*q /(4*np.pi)
        
        r0 = (3*V/(4*np.pi))**(1/3)
        rm = 4.188
        #if r0 <1.04*rm and r0 >0.96*rm:
        #    print("Using calculated expansion factor")
        #    rbest = r0
        #else:
        #    print("Using max expansion factor")
        #    rbest = 0.5*(1.04*rm + 0.96*rm)
        
        return (r0/rm)**3 * np.exp(-q2 * pow(4*np.pi/3, 2/3)*(r0**2-rm**2))

    def debye_mem(self, dgram, q, ff, eps=1e-6):
        """ Calculate I(q) using the Debye formula for a single type of scatterer.
        Args:
            dgram (np.array): Distance matrix between scatterers.
            q (float): Scattering vector.
            ff (np.array): Form factors for the scatterers.
            eps (float): Small value to avoid division by zero.
        Returns:
            Iq (float): Scattering intensity I(q).
        """
        ff2 = (ff[None] * ff[:, None])
        if q == 0:
                return np.sum(ff2)
        else:
            Iq = np.zeros_like(dgram)
            dq = dgram * q
            indices_zeros = dq<eps
            indices_nonzeros = ~indices_zeros
            
            Iq[indices_nonzeros] = ff2[indices_nonzeros]  * np.sin(dq[indices_nonzeros]) / dq[indices_nonzeros]
            Iq[indices_zeros] = ff2[indices_zeros] * (1 - (1/6) * (dq[indices_zeros])**2)
            return np.sum(Iq, axis=(0,1))

    def debye_hs(self,dgram, q, waterff, aminoff, eps=1e-6):
        """ Calculate I(q) using the Debye formula for two types of scatterers (e.g., protein and solvent).
        Args:
            dgram (np.array): Distance matrix between scatterers.
            q (float): Scattering vector.
            waterff (np.array): Form factors for the solvent scatterers.
            aminoff (np.array): Form factors for the protein scatterers.
            eps (float): Small value to avoid division by zero.
        Returns:
            Iq (float): Scattering intensity I(q).
        """
        # Calculate all combinations of form factors between water and amino acids
        ff2 = waterff[:, None] * aminoff[None] #all water form factors times all amino acid form factor
        if q == 0:
                return np.sum(ff2)
        else:
            Iq = np.zeros_like(dgram)
            dq = dgram * q
            indices_zeros = dq<eps
            indices_nonzeros = ~indices_zeros

            Iq[indices_nonzeros] = ff2[indices_nonzeros]  * np.sin(dq[indices_nonzeros]) / dq[indices_nonzeros]
            Iq[indices_zeros] = ff2[indices_zeros] * (1 - (1/6) * (dq[indices_zeros])**2)
            return np.sum(Iq, axis=(0,1))
        
    def calculate_distogram(self, coords):
        dgram = np.sqrt(np.sum(
            (coords[..., None, :] - coords[..., None, :, :]) ** 2, axis=-1
        ))
        return dgram

    def calculate_dist2(self, water_coords, aa_coords):
        dgram = np.sqrt(np.sum(
            (water_coords[..., None, :] - aa_coords[..., None, :, :]) ** 2, axis=-1
        ))
        return dgram


    def debye_formula(self, structure, q_values, aa_FormFactors, water_struct, water_FormFactors):
        """ Calculate I(q) using the Debye formula.
        Args:
            structure (np.array): Coordinates of the structure.
            q_values (np.array): Scattering vector values.
            aa_FormFactors (np.array): Form factors for the amino acid scatterers.
            water_struct (np.array): Coordinates of the solvent.
            water_FormFactors (np.array): Form factors for the solvent scatterers.
        Returns:
            np.array: Scattering intensity I(q) for each q value where 
            I(q) = I_aa(q) + I_solv(q) + 2*I_cross(q). (atomistic, excluded solvent and hydration shell contributions)
        """
        # Calculate I(q) using the Debye formula
        Iq_values = np.zeros(len(q_values))
        for i, q in enumerate(q_values):
            Iq_values[i] = self.debye_mem(self.calculate_distogram(structure), q, aa_FormFactors[:,i]) 
            + self.debye_mem(self.calculate_distogram(water_struct), q, water_FormFactors[:,i])
            + 2 * self.debye_hs(self.calculate_dist2(water_struct, structure), q, water_FormFactors[:,i], aa_FormFactors[:,i])
        return Iq_values

