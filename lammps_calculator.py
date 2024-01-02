from ase.io import read
from ase.calculators.lammpslib import LAMMPSlib


class LammpsCalculator:
    def __init__(self, xyz_file, run_lammps=True):
        """
        Initializes the LammpsCalculator with the path to the XYZ file.
        Optionally, LAMMPS simulations can be run upon initialization.

        :param xyz_file: Path to the XYZ file containing the atomic structure.
        :param run_lammps: Boolean flag to run LAMMPS simulation
                           during initialization.
        """
        self.atom = read(xyz_file)

        # Initialize attributes to store calculated properties.
        self.total_energy = None
        self.total_forces = None
        self.intra_L1_energy = None
        self.intra_L1_forces = None
        self.intra_L2_energy = None
        self.intra_L2_forces = None
        self.inter_energy = None
        self.inter_forces = None
        # Run LAMMPS simulations if specified during initialization.
        if run_lammps:
            self.run_lammps()

    def run_lammps(self):
        """
        Runs both intralayer and interlayer LAMMPS simulations.
        """
        self.run_intralayer()
        self.run_interlayer()

    def run_interlayer(self):
        """
        Performs the interlayer LAMMPS simulation. 
        It calculates the interlayer interactions,
        including forces and energy. 
        Assumes intralayer simulations have been run.
        """
        # Ensure intralayer forces and energy have been calculated.
        if self.intra_L1_energy is None or self.intra_L1_forces is None:
            self.run_intralayer()
        # Copy the atomic structure to perform interlayer calculations.
        atom_interlayer = self.atom.copy()
        original_atom_types = atom_interlayer.get_chemical_symbols()
        atom_interlayer.numbers = atom_interlayer.arrays["atom_types"] + 1
        # Define the LAMMPS settings for the interlayer simulation.
        cmds = [
            # LAMMPS commands go here.
            "pair_style hybrid/overlay sw sw kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 lj/cut 10.0",
            f"pair_coeff * * sw 1 tmd.sw {original_atom_types[0]} {original_atom_types[1]} {original_atom_types[2]} NULL NULL NULL",
            f"pair_coeff * * sw 2 tmd.sw NULL NULL NULL {original_atom_types[3]} {original_atom_types[4]} {original_atom_types[5]}",
            f"pair_coeff 1 6 kolmogorov/crespi/z 1 WS.KC  {original_atom_types[0]} NULL NULL NULL NULL  {original_atom_types[5]}",
            f"pair_coeff 2 4 kolmogorov/crespi/z 2 WS.KC NULL  {original_atom_types[1]} NULL {original_atom_types[3]} NULL NULL",
            f"pair_coeff 2 6 kolmogorov/crespi/z 3 WS.KC NULL  {original_atom_types[1]} NULL NULL NULL  {original_atom_types[5]}",
            f"pair_coeff 1 4 kolmogorov/crespi/z 4 WS.KC  {original_atom_types[0]} NULL NULL {original_atom_types[3]} NULL NULL",
            "pair_coeff * * lj/cut 0.0 3.0",
            "neighbor        2.0 bin",
            "neigh_modify every 1 delay 0 check yes"]

        # Define fixed atom types and masses for the simulation.
        fixed_atom_types = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6}
        fixed_atom_type_masses = {'H': 95.95,
                                  'He': 32.06,
                                  'Li': 32.06,
                                  'Be': 95.95,
                                  'B': 32.06,
                                  'C': 32.06}
        # Set up the LAMMPS calculator with the specified commands.
        lammps_calc = LAMMPSlib(lmpcmds=cmds,
                                atom_types=fixed_atom_types,
                                atom_type_masses=fixed_atom_type_masses)
        # Calculate and store the total energy and forces.
        atom_interlayer.calc = lammps_calc

        # Calculates the total energy and interlayer energy
        self.total_energy = atom_interlayer.get_potential_energy()
        intralayer_energy = (self.intra_L1_energy + self.intra_L2_energy)
        self.inter_energy = self.total_energy - intralayer_energy

        # Calculates the total forces and interlayer forces
        self.total_forces = atom_interlayer.get_forces()

        # Calculate the interlayer forces.
        L1_cond = self.atom.positions[:, 2] < self.atom.positions[:, 2].mean()
        L2_cond = self.atom.positions[:, 2] >= self.atom.positions[:, 2].mean()
        self.inter_forces = self.total_forces.copy()
        self.inter_forces[L1_cond] -= self.intra_L1_forces
        self.inter_forces[L2_cond] -= self.intra_L2_forces

    def run_intralayer(self):        
        """
        Performs intralayer LAMMPS simulations for each layer separately.
        Calculates intralayer interactions, including forces and energy.
        """
        # Intralayer calculation for the first layer.
        atom_L1 = self.atom[
            self.atom.positions[:, 2] < self.atom.positions[:, 2].mean()
            ]
        atom_L1.numbers = atom_L1.arrays["atom_types"] + 1
        # Define LAMMPS commands for the first layer.
        cmds = [
            "pair_style hybrid/overlay sw lj/cut 10.0",
            "pair_coeff * * sw tmd.sw Mo S S",
            "pair_coeff * * lj/cut 0.0 3.0",
            "neighbor        2.0 bin",
            "neigh_modify every 1 delay 0 check yes"]

        atom_types = {'H': 1, 'He': 2, 'Li': 3}
        atom_type_masses = {'H': 95.95, 'He': 32.06, 'Li': 32.06}
        # Set up and run the LAMMPS simulation for the first layer.
        lammps_L1 = LAMMPSlib(lmpcmds=cmds,
                              atom_types=atom_types,
                              atom_type_masses=atom_type_masses)
        atom_L1.calc = lammps_L1
        self.intra_L1_energy = atom_L1.get_potential_energy()
        self.intra_L1_forces = atom_L1.get_forces()

        # Repeat similar calculations for the second layer.

        atom_L2 = self.atom[
            self.atom.positions[:, 2] >= self.atom.positions[:, 2].mean()
            ]
        atom_L2.numbers = atom_L2.arrays["atom_types"] - 3 + 1

        cmds = [
            "pair_style hybrid/overlay sw lj/cut 10.0",
            "pair_coeff * * sw tmd.sw Mo S S",
            "pair_coeff * * lj/cut 0.0 3.0",
            "neighbor        2.0 bin",
            "neigh_modify every 1 delay 0 check yes"]

        atom_types = {'H': 1, 'He': 2, 'Li': 3}
        atom_type_masses = {'H': 95.95, 'He': 32.06, 'Li': 32.06}
        lammps_L2 = LAMMPSlib(lmpcmds=cmds,
                              atom_types=atom_types,
                              atom_type_masses=atom_type_masses)
        atom_L2.calc = lammps_L2
        self.intra_L2_energy = atom_L2.get_potential_energy()
        self.intra_L2_forces = atom_L2.get_forces()
        
    # Getter methods for calculated properties
    def get_intralayer_forces(self):
        """
        Retrieves intralayer forces from the results.
        :return: Intralayer forces.
        """
        return [self.intra_L1_forces, self.intra_L2_forces]

    def get_interlayer_forces(self):
        """
        Retrieves interlayer forces from the results.
        :return: Interlayer forces.
        """
        return self.inter_forces

    def get_intralayer_energy(self):
        """
        Retrieves intralayer energy from the results.
        :return: Intralayer energy.
        """
        return [self.intra_L1_energy, self.intra_L2_energy]

    def get_interlayer_energy(self):
        """
        Retrieves interlayer energy from the results.
        :return: Interlayer energy.
        """
        return self.inter_energy

# Example usage:
# # Example usage of LammpsCalculator

# # Path to your XYZ file
# xyz_file_path = "MoS2-Bilayer.xyz"

# # Create an instance of LammpsCalculator
# # The `run_lammps` parameter is set to True to automatically run simulations upon initialization
# lammps_calc = LammpsCalculator(xyz_file_path, run_lammps=True)

# # If simulations are not run during initialization, you can manually run them
# # lammps_calc.run_lammps()

# # Retrieve total forces and energy
# total_forces = lammps_calc.total_forces
# total_energy = lammps_calc.total_energy

# # Retrieve intralayer forces and energy for Layer 1 and Layer 2
# intralayer_forces_L1, intralayer_forces_L2 = lammps_calc.get_intralayer_forces()
# intralayer_energy_L1, intralayer_energy_L2 = lammps_calc.get_intralayer_energy()

# # Retrieve interlayer forces and energy
# interlayer_forces = lammps_calc.get_interlayer_forces()
# interlayer_energy = lammps_calc.get_interlayer_energy()

# # Print out the results
# print("Total Forces:", total_forces)
# print("Total Energy:", total_energy)
# print("Intralayer Forces Layer 1:", intralayer_forces_L1)
# print("Intralayer Forces Layer 2:", intralayer_forces_L2)
# print("Intralayer Energy Layer 1:", intralayer_energy_L1)
# print("Intralayer Energy Layer 2:", intralayer_energy_L2)
# print("Interlayer Forces:", interlayer_forces)
# print("Interlayer Energy:", interlayer_energy)

