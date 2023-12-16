import sys
import data_loader
import dft_reader
import allegro_calculator
import lammps_calculator
import analysis
import config

def main():
    # Load atomic configurations
    atomic_configs = data_loader.load_atomic_configurations(config.atomic_config_path)

    # Calculate forces and energies using ALLEGRO
    allegro_results = allegro_calculator.calculate_forces_and_energies(atomic_configs)

    # Calculate forces and energies using LAMMPS
    lammps_results = lammps_calculator.calculate_forces_and_energies(atomic_configs)

    # Read pre-computed DFT results
    dft_results = dft_reader.read_dft_results(config.dft_output_path)

    # Compare the results from DFT, ALLEGRO, and LAMMPS
    comparison_results = analysis.compare_methods(dft_results, allegro_results, lammps_results)

    # Optionally, save or display the comparison results
    analysis.save_comparison_results(comparison_results, config.results_output_path)

if __name__ == "__main__":
    main()

