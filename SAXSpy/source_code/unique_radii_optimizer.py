"""
Unique Radii Optimization Module

This module provides functions to optimize atomic radii based on unique atom types
instead of optimizing each individual atom radius separately. This approach is more
efficient and physically meaningful.

Author: Generated for SAXSpy project
"""

import numpy as np
from scipy.optimize import minimize


def assign_radii_to_atoms(atoms_char, unique_radii_dict):
    """
    Assign radii to each atom based on its type.
    
    Args:
        atoms_char: Array of atom types/characters for each atom
        unique_radii_dict: Dictionary mapping atom type to radius
    
    Returns:
        Array of radii for each atom
    """
    return np.array([unique_radii_dict[atom] for atom in atoms_char])


def get_unique_atom_info(atoms_char, fraser_radii):
    """
    Get unique atom types and their corresponding radii.
    
    Args:
        atoms_char: Array of atom types for each atom
        fraser_radii: Array of radii for each atom
    
    Returns:
        unique_atoms: Array of unique atom types
        unique_radii: Array of radii for unique atom types
        atom_type_map: Dictionary mapping atom type to radius
        atom_indices: Indices to map unique radii back to all atoms
    """
    unique_atoms, indices = np.unique(atoms_char, return_inverse=True)
    unique_radii = np.array([fraser_radii[np.where(atoms_char == atom)[0][0]] for atom in unique_atoms])
    atom_type_map = dict(zip(unique_atoms, unique_radii))
    
    return unique_atoms, unique_radii, atom_type_map, indices


def Icalc_fixed(q, structure, radii, probe, getDummyAtomsFactorCorr0, create_voro_balls, 
                calculate_distogram, debye_mem, voro):
    """
    Calculate scattering intensity I(q). Fixed version that handles array q properly.
    
    Args:
        q: Scattering vector (scalar or array)
        structure: Atom coordinates
        radii: Atom radii
        probe: Probe radius
        getDummyAtomsFactorCorr0: Function to calculate dummy form factors
        create_voro_balls: Function to create Voronoi balls
        calculate_distogram: Function to calculate distance matrix
        debye_mem: Function to calculate Debye scattering
        voro: Voronoi tessellation module
    
    Returns:
        Scattering intensity I(q)
    """
    balls = create_voro_balls(structure, radii)
    rt = voro.RadicalTessellation(balls, probe)
    cells = list(rt.cells)
    vols = np.array([cell.volume for cell in cells])
    dgram = calculate_distogram(structure)
    
    # Handle both scalar and array q values
    if np.isscalar(q):
        q_array = [q]
    else:
        q_array = q
    
    Iqvals = np.zeros(len(q_array))
    for i, q_val in enumerate(q_array):
        dummyFFs = np.array([getDummyAtomsFactorCorr0(q_val, V) for V in vols])
        Iqvals[i] = debye_mem(dgram, q_val, dummyFFs)
    
    return Iqvals if not np.isscalar(q) else Iqvals[0]


def scattering_objective_unique_radii(x, unique_atoms, atom_indices, q, I_exp, structure, 
                                    probe, getDummyAtomsFactorCorr0, create_voro_balls,
                                    calculate_distogram, debye_mem, voro, sigma=None, lam=0.0, verbose=False):
    """
    Objective function that optimizes only unique radii values.
    NOW USES PROPER SCALING TO AVOID NUMERICAL INSTABILITY.
    
    Args:
        x: Array of unique radii values to optimize
        unique_atoms: Array of unique atom types
        atom_indices: Indices to map unique radii back to all atoms
        q: Scattering vector
        I_exp: Experimental intensity
        structure: Atom coordinates
        probe: Probe radius
        getDummyAtomsFactorCorr0: Function to calculate dummy form factors
        create_voro_balls: Function to create Voronoi balls
        calculate_distogram: Function to calculate distance matrix
        debye_mem: Function to calculate Debye scattering
        voro: Voronoi tessellation module
        sigma: Experimental uncertainty (optional)
        lam: Regularization strength
    
    Returns:
        Objective function value (scaled chi-squared + regularization)
    """
    
    # Safety: enforce positivity
    if np.any(x <= 0.0):
        return 1e10
    
    # Map unique radii back to all atoms
    radii = x[atom_indices]
    
    # Debug output (can be removed later)
    if verbose and np.random.random() < 0.1:  # Only print 10% of the time to avoid spam
        print(f"Current unique radii: {x}")
    
    try:
        # Calculate scattering intensity
        I_calc = Icalc_fixed(q, structure, radii, probe, getDummyAtomsFactorCorr0,
                            create_voro_balls, calculate_distogram, debye_mem, voro)

        # PROPERLY SCALED chi-squared calculation
        # Use relative residuals to avoid huge objective function values
        # that make optimization unstable
        if sigma is None:
            relative_residuals = (I_calc - I_exp) / I_exp
            chi2 = np.sum(relative_residuals**2)
        else:
            chi2 = np.sum(((I_calc - I_exp) / sigma)**2)

        # Optional regularization (toward reference radii)
        reg = 0  # No regularization for now
        
        return chi2 + reg
    
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 1e10


def optimize_unique_radii(atoms_char, fraser_radii, q, I_exp, structure, probe=0.2,
                         getDummyAtomsFactorCorr0=None, create_voro_balls=None,
                         calculate_distogram=None, debye_mem=None, voro=None,
                         bounds_range=(0.1, 5.0), method="L-BFGS-B", 
                         max_iter=100, tolerance=1e-6, verbose=True):
    """
    Main optimization function that optimizes unique radii.
    
    Args:
        atoms_char: Array of atom types for each atom
        fraser_radii: Initial radii for each atom
        q: Scattering vector array
        I_exp: Experimental intensity
        structure: Atom coordinates
        probe: Probe radius for Voronoi tessellation
        getDummyAtomsFactorCorr0: Function to calculate dummy form factors
        create_voro_balls: Function to create Voronoi balls
        calculate_distogram: Function to calculate distance matrix
        debye_mem: Function to calculate Debye scattering
        voro: Voronoi tessellation module
        bounds_range: Tuple of (min, max) bounds for radii
        method: Optimization method
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
        verbose: Print optimization progress
        
    Returns:
        result: Optimization result object
        optimized_radii: Array of optimized radii for all atoms
        unique_info: Dictionary with unique atom information
    """
    
    # Get unique atom information
    unique_atoms, unique_radii, atom_type_map, atom_indices = get_unique_atom_info(atoms_char, fraser_radii)
    
    if verbose:
        print(f"Unique atom types: {unique_atoms}")
        print(f"Initial unique radii: {unique_radii}")
        print(f"Number of unique atom types to optimize: {len(unique_atoms)}")
        print(f"Optimizing {len(unique_atoms)} parameters instead of {len(structure)}")
    
    # Set up optimization variables
    x0 = unique_radii.copy()  # Start with the actual initial unique radii values
    bounds = [bounds_range] * len(unique_atoms)  # Bounds for each unique radius
    
    if verbose:
        print(f"Initial x0: {x0}")
        print(f"Bounds: {bounds}")
    # Run optimization with better settings for the scaled objective
    result = minimize(
        scattering_objective_unique_radii,
        x0,
        args=(unique_atoms, atom_indices, q, I_exp, structure, probe,
              getDummyAtomsFactorCorr0, create_voro_balls, calculate_distogram,
              debye_mem, voro, None, 0.0, verbose),
        method=method,
        bounds=bounds,
        options={
            "ftol": tolerance,
            "maxiter": max_iter,
            "maxcor": 10,  # Reduce memory usage
            "maxls": 20,   # Limit line search steps
        }
    )
    
    if verbose:
        print("\nOptimization result:")
        print(f"Success: {result.success}")
        print(f"Final unique radii: {result.x}")
        print(f"Final objective value: {result.fun}")
    
    # Map optimized unique radii back to all atoms
    optimized_radii = result.x[atom_indices]
    
    # Create info dictionary
    unique_info = {
        'unique_atoms': unique_atoms,
        'initial_unique_radii': unique_radii,
        'final_unique_radii': result.x,
        'atom_type_map': dict(zip(unique_atoms, result.x)),
        'atom_indices': atom_indices
    }
    
    if verbose:
        print(f"Shape of optimized radii for all atoms: {optimized_radii.shape}")
        print("\nOptimized radii by atom type:")
        for atom, radius in zip(unique_atoms, result.x):
            print(f"  {atom}: {radius:.4f}")
    
    return result, optimized_radii, unique_info


# Example usage function
def example_usage():
    """
    Example of how to use the unique radii optimizer in your notebook.
    """
    print("""
    Example usage in your notebook:
    
    # Import the module
    import sys
    sys.path.append('.')  # Add current directory to path
    from unique_radii_optimizer import optimize_unique_radii
    
    # Run optimization
    result, optimized_radii, unique_info = optimize_unique_radii(
        atoms_char=atoms_char,
        fraser_radii=fraser_radii,
        q=q_pepsi,
        I_exp=Iev,
        structure=extracted_coords,
        probe=0.2,
        getDummyAtomsFactorCorr0=getDummyAtomsFactorCorr0,
        create_voro_balls=create_voro_balls,
        calculate_distogram=calculate_distogram,
        debye_mem=debye_mem,
        voro=voro,
        verbose=True
    )
    
    # The result contains:
    # - result: scipy optimization result
    # - optimized_radii: array of radii for all atoms
    # - unique_info: dictionary with unique atom information
    """)


if __name__ == "__main__":
    example_usage()