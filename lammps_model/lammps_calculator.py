from ase.io import read
from ase.calculators.lammpslib import LAMMPSlib


class LammpsCalculator:
    def __init__(self, xyz_file, run_lammps=True):
        """
        Initializes the LammpsCalculator with the path to
        the XYZ file and LAMMPS executable.
        :param xyz_file: Path to the XYZ file containing the atomic structure.
        """
        self.atom = read(xyz_file)
        self.total_energy = None
        self.total_forces = None
        self.intra_L1_energy = None
        self.intra_L1_forces = None
        self.intra_L2_energy = None
        self.intra_L2_forces = None
        self.inter_energy = None
        self.inter_forces = None

        if run_lammps:
            self.run_lammps()

    def run_lammps(self):
        """
        Runs the intralayer and intralayer simulations.
        """
        self.run_intralayer()
        self.run_interlayer()

    def run_interlayer(self):
        # Checks if the intralayer forces and energy have been calculated.
        if self.intra_L1_energy is None or self.intra_L1_forces is None:
            self.run_intralayer()

        atom_interlayer = self.atom.copy()
        original_atom_types = atom_interlayer.get_chemical_symbols()
        atom_interlayer.numbers = atom_interlayer.arrays["atom_types"] + 1

        cmds = [
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

        fixed_atom_types = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6}
        fixed_atom_type_masses = {'H': 95.95,
                                  'He': 32.06,
                                  'Li': 32.06,
                                  'Be': 95.95,
                                  'B': 32.06,
                                  'C': 32.06}
        lammps_calc = LAMMPSlib(lmpcmds=cmds,
                                atom_types=fixed_atom_types,
                                atom_type_masses=fixed_atom_type_masses)
        atom_interlayer.calc = lammps_calc

        # Calculates the total energy and interlayer energy
        self.total_energy = atom_interlayer.get_potential_energy()
        intralayer_energy = (self.intra_L1_energy + self.intra_L2_energy)
        self.inter_energy = self.total_energy - intralayer_energy

        # Calculates the total forces and interlayer forces
        self.total_forces = atom_interlayer.get_forces()

        # Layer conditions
        L1_cond = self.atom.positions[:, 2] < self.atom.positions[:, 2].mean()
        L2_cond = self.atom.positions[:, 2] >= self.atom.positions[:, 2].mean()
        self.inter_forces = self.total_forces.copy()
        self.inter_forces[L1_cond] -= self.intra_L1_forces
        self.inter_forces[L2_cond] -= self.intra_L2_forces

    def run_intralayer(self):
        atom_L1 = self.atom[
            self.atom.positions[:, 2] < self.atom.positions[:, 2].mean()
            ]
        atom_L1.numbers = atom_L1.arrays["atom_types"] + 1

        cmds = [
            "pair_style hybrid/overlay sw lj/cut 10.0",
            "pair_coeff * * sw tmd.sw Mo S S",
            "pair_coeff * * lj/cut 0.0 3.0",
            "neighbor        2.0 bin",
            "neigh_modify every 1 delay 0 check yes"]

        atom_types = {'H': 1, 'He': 2, 'Li': 3}
        atom_type_masses = {'H': 95.95, 'He': 32.06, 'Li': 32.06}
        lammps_L1 = LAMMPSlib(lmpcmds=cmds,
                              atom_types=atom_types,
                              atom_type_masses=atom_type_masses)
        atom_L1.calc = lammps_L1
        self.intra_L1_energy = atom_L1.get_potential_energy()
        self.intra_L1_forces = atom_L1.get_forces()

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
        print(atom_L2.get_forces())

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
# lammps_calc = LammpsCalculator("MoS2-Bilayer.xyz")
# lammps_calc.run_simulation()
# forces = lammps_calc.get_forces()
# energy = lammps_calc.get_energy()
