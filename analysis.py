import matplotlib.pyplot as plt
import numpy as np

def plot_force_correlation(dft_forces, method_forces, method_name):
    """
    Plots a correlation plot for forces.

    :param dft_forces: Forces from DFT calculations.
    :param method_forces: Forces from the specified method (ALLEGRO or LAMMPS).
    :param method_name: Name of the method for labeling.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(dft_forces, method_forces, alpha=0.5, label=f'{method_name} Forces')
    plt.plot(dft_forces, dft_forces, color='red', label='Ideal Match')
    plt.xlabel('DFT Forces')
    plt.ylabel(f'{method_name} Forces')
    plt.title(f'Force Correlation: DFT vs {method_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_energy_correlation(dft_energy, method_energy, method_name):
    """
    Plots a correlation plot for energy.

    :param dft_energy: Energy from DFT calculations.
    :param method_energy: Energy from the specified method (ALLEGRO or LAMMPS).
    :param method_name: Name of the method for labeling.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(dft_energy, method_energy, alpha=0.5, label=f'{method_name} Energy')
    plt.plot(dft_energy, dft_energy, color='red', label='Ideal Match')
    plt.xlabel('DFT Energy')
    plt.ylabel(f'{method_name} Energy')
    plt.title(f'Energy Correlation: DFT vs {method_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_force_error_scaling(dft_forces, method_forces, method_name):
    """
    Plots the scaling of force error magnitude.

    :param dft_forces: Forces from DFT calculations.
    :param method_forces: Forces from the specified method (ALLEGRO or LAMMPS).
    :param method_name: Name of the method for labeling.
    """
    force_magnitude = np.linalg.norm(dft_forces, axis=1)
    error_magnitude = np.linalg.norm(dft_forces - method_forces, axis=1)

    plt.figure(figsize=(8, 6))
    plt.scatter(force_magnitude, error_magnitude, alpha=0.5)
    plt.xlabel('Magnitude of DFT Forces')
    plt.ylabel('Magnitude of Force Error')
    plt.title(f'Force Error Scaling: DFT vs {method_name}')
    plt.grid(True)
    plt.show()

# Example usage (You should replace these with your actual data)
# plot_force_correlation(dft_forces, allegro_forces, "ALLEGRO")
# plot_energy_correlation(dft_energy, lammps_energy, "LAMMPS")
# plot_force_error_scaling(dft_forces, lammps_forces, "LAMMPS")
