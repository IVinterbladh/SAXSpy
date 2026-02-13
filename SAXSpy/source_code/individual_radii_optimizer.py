"""
Individual Radii Optimization Module

This module provides functions to optimize each individual atomic radius separately.
This approach optimizes all atom radii simultaneously to find the best fit.

Author: Generated for SAXSpy project
"""

import numpy as np
from scipy.optimize import minimize


def Icalc_individual(q, structure, radii, probe, getDummyAtomsFactorCorr0, create_voro_balls, 
                    calculate_distogram, debye_mem, voro):
    """
    Calculate scattering intensity I(q) for individual atom radii.
    Fixed version that handles array q properly.
    
    Args:
        q: Scattering vector (scalar or array)
        structure: Atom coordinates
        radii: Individual atom radii (one per atom)
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


def scattering_objective_individual_radii(x, q, I_exp, structure, probe, 
                                        getDummyAtomsFactorCorr0, create_voro_balls,
                                        calculate_distogram, debye_mem, voro, 
                                        sigma=None, lam=0.0, reference_radii=None):
    """
    Objective function that optimizes all individual radii values.
    
    Args:
        x: Array of all individual radii values to optimize
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
        reference_radii: Reference radii for regularization (optional)
    
    Returns:
        Objective function value (chi-squared + regularization)
    """
    
    # Safety: enforce positivity
    if np.any(x <= 0.0):
        return 1e20
    
    # Use all radii directly (no mapping needed)
    radii = x
    
    try:
        # Calculate scattering intensity
        I_calc = Icalc_individual(q, structure, radii, probe, getDummyAtomsFactorCorr0,
                                create_voro_balls, calculate_distogram, debye_mem, voro)

        # Data misfit
        if sigma is None:
            chi2 = np.sum((I_calc - I_exp)**2)
        else:
            chi2 = np.sum(((I_calc - I_exp) / sigma)**2)

        # Optional regularization (toward reference radii)
        if lam > 0 and reference_radii is not None:
            reg = lam * np.sum((x - reference_radii)**2)
        else:
            reg = 0

        return chi2 + reg
    
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 1e20


def optimize_individual_radii(fraser_radii, q, I_exp, structure, probe=0.2,
                            getDummyAtomsFactorCorr0=None, create_voro_balls=None,
                            calculate_distogram=None, debye_mem=None, voro=None,
                            bounds_range=(0.1, 5.0), method="L-BFGS-B", 
                            max_iter=2, tolerance=1e-6, verbose=True,
                            regularization=0.0):
    """
    Main optimization function that optimizes all individual radii.
    
    Args:
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
        regularization: Regularization strength (0 = no regularization)
        
    Returns:
        result: Optimization result object
        optimized_radii: Array of optimized radii for all atoms
        optimization_info: Dictionary with optimization information
    """
    
    n_atoms = len(fraser_radii)
    
    if verbose:
        print(f"Optimizing {n_atoms} individual atom radii")
        print(f"Initial radii range: {np.min(fraser_radii):.4f} to {np.max(fraser_radii):.4f}")
        print(f"Bounds per radius: {bounds_range}")
    
    # Set up optimization variables
    x0 = fraser_radii.copy()  # Start with the initial radii values
    bounds = [bounds_range] * n_atoms  # Bounds for each individual radius
    
    # Run optimization
    result = minimize(
        scattering_objective_individual_radii,
        x0,
        args=(q, I_exp, structure, probe,
              getDummyAtomsFactorCorr0, create_voro_balls, calculate_distogram,
              debye_mem, voro, None, regularization, fraser_radii),
        method=method,
        bounds=bounds,
        options={
            "ftol": tolerance,
            "maxiter": max_iter,
            "disp": verbose
        }
    )
    
    if verbose:
        print(f"\nOptimization result:")
        print(f"Success: {result.success}")
        if not result.success:
            print(f"Message: {result.message}")
        print(f"Final objective value: {result.fun}")
        print(f"Number of iterations: {result.nit}")
        print(f"Optimized radii range: {np.min(result.x):.4f} to {np.max(result.x):.4f}")
        
        # Show radius changes
        changes = result.x - fraser_radii
        print(f"Radius changes range: {np.min(changes):.4f} to {np.max(changes):.4f}")
        print(f"Mean absolute change: {np.mean(np.abs(changes)):.4f}")
        
        # Show largest changes
        largest_changes_idx = np.argsort(np.abs(changes))[-5:]
        print(f"\nLargest radius changes:")
        for i in reversed(largest_changes_idx):
            print(f"  Atom {i}: {fraser_radii[i]:.4f} → {result.x[i]:.4f} (Δ{changes[i]:+.4f})")
    
    # Create info dictionary
    optimization_info = {
        'initial_radii': fraser_radii.copy(),
        'final_radii': result.x.copy(),
        'radius_changes': result.x - fraser_radii,
        'n_atoms_optimized': n_atoms,
        'optimization_success': result.success,
        'final_objective': result.fun,
        'n_iterations': result.nit
    }
    
    return result, result.x, optimization_info


def compare_radii(initial_radii, optimized_radii, atom_labels=None, show_top_n=10):
    """
    Compare initial and optimized radii.
    
    Args:
        initial_radii: Initial radii array
        optimized_radii: Optimized radii array
        atom_labels: Optional labels for atoms
        show_top_n: Number of largest changes to show
    """
    changes = optimized_radii - initial_radii
    abs_changes = np.abs(changes)
    
    print(f"Radius optimization summary:")
    print(f"{'='*50}")
    print(f"Total atoms: {len(initial_radii)}")
    print(f"Initial range: {np.min(initial_radii):.4f} to {np.max(initial_radii):.4f}")
    print(f"Final range: {np.min(optimized_radii):.4f} to {np.max(optimized_radii):.4f}")
    print(f"Mean change: {np.mean(changes):+.4f}")
    print(f"Mean absolute change: {np.mean(abs_changes):.4f}")
    print(f"Max absolute change: {np.max(abs_changes):.4f}")
    
    # Show top changes
    top_changes_idx = np.argsort(abs_changes)[-show_top_n:]
    print(f"\nTop {show_top_n} largest changes:")
    print(f"{'Atom':<8} {'Initial':<8} {'Final':<8} {'Change':<8}")
    print(f"{'-'*35}")
    
    for i in reversed(top_changes_idx):
        atom_label = f"{i}" if atom_labels is None else f"{atom_labels[i]}"
        print(f"{atom_label:<8} {initial_radii[i]:<8.4f} {optimized_radii[i]:<8.4f} {changes[i]:+8.4f}")


# Example usage function
def example_usage():
    """
    Example of how to use the individual radii optimizer in your notebook.
    """
    print("""
    Example usage in your notebook:
    
    # Import the module
    import sys
    sys.path.append('.')  # Add current directory to path
    from individual_radii_optimizer import optimize_individual_radii, compare_radii
    
    # Run optimization for ALL individual radii
    result, optimized_radii, opt_info = optimize_individual_radii(
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
        bounds_range=(0.1, 5.0),
        regularization=0.0,  # No regularization for maximum flexibility
        verbose=True
    )
    
    # Compare results
    compare_radii(fraser_radii, optimized_radii, atom_labels=atoms_char)
    
    # The result contains:
    # - result: scipy optimization result
    # - optimized_radii: array of optimized radii for all atoms
    # - opt_info: dictionary with optimization information
    """)


if __name__ == "__main__":
    example_usage()