from ase.io import read

class DFTCalculator:
    def __init__(self, total_scf_file, scf_file_L1, scf_file_L2, read_dft=True):
        """
        Initializes the DFTCalculator with paths to the SCF output files for two layers and the total system.
        Optionally, reads data from SCF output files upon initialization.

        :param scf_file_L1: Path to the SCF output file for Layer 1.
        :param scf_file_L2: Path to the SCF output file for Layer 2.
        :param total_scf_file: Path to the SCF output file for the total system.
        :param run_dft: Boolean flag to read data from SCF files during initialization.
        """
        self.total_scf_file = total_scf_file
        
        self.scf_file_L1 = None
        self.scf_file_L2 = None
        
        if scf_file_L1:
            self.scf_file_L1 = scf_file_L1
        if scf_file_L2:
            self.scf_file_L2 = scf_file_L2

        


        # Initialize properties
        self.intra_L1_energy = None
        self.intra_L1_forces = None
        self.intra_L2_energy = None
        self.intra_L2_forces = None
        self.total_energy = None
        self.total_forces = None
        self.inter_energy = None
        self.inter_forces = None

        self.atom_total = None

        if read_dft:
            self.read_dft_data()

    def read_dft_data(self):
        """
        Reads forces and energy data from SCF output files for both layers and the total system.
        """

        self.read_total_data()

        if self.scf_file_L1 and self.scf_file_L2:
            self.read_layer_data(layer='L1')
            self.read_layer_data(layer='L2')
            intralayer_energy = (self.intra_L1_energy + self.intra_L2_energy)
            self.inter_energy = self.total_energy - intralayer_energy

            # Calculate the interlayer forces.
            L1_cond = self.atom_total.positions[:, 2] < self.atom_total.positions[:, 2].mean()
            L2_cond = self.atom_total.positions[:, 2] >= self.atom_total.positions[:, 2].mean()
            self.inter_forces = self.total_forces.copy()
            self.inter_forces[L1_cond] -= self.intra_L1_forces
            self.inter_forces[L2_cond] -= self.intra_L2_forces


    def read_layer_data(self, layer):
        """
        Reads energy and forces from the SCF output file for a specified layer.

        :param layer: Specifies the layer ('L1' or 'L2').
        """

        if layer == 'L1':
            atom_layer = read(self.scf_file_L1, format='espresso-out')
            self.intra_L1_energy = atom_layer.get_total_energy()
            self.intra_L1_forces = atom_layer.get_forces()
        elif layer == 'L2':
            atom_layer = read(self.scf_file_L2, format='espresso-out')
            self.intra_L2_energy = atom_layer.get_total_energy()
            self.intra_L2_forces = atom_layer.get_forces()

    def read_total_data(self):
        """
        Reads the total energy and forces from the total system SCF output file.
        """
        self.atom_total = read(self.total_scf_file, format='espresso-out')
        self.total_energy = self.atom_total.get_total_energy()
        self.total_forces = self.atom_total.get_forces()

    # Getter methods for calculated properties
    def get_intralayer_forces(self, layer):
        """
        Retrieves intralayer forces from the results.
        :return: Intralayer forces.
        """
        if layer == 'L1':
            return self.intra_L1_forces
        elif layer == 'L2':
            return self.intra_L2_forces

    def get_interlayer_forces(self):
        """
        Retrieves interlayer forces from the results.
        :return: Interlayer forces.
        """
        return self.inter_forces

    def get_intralayer_energy(self, layer):
        """
        Retrieves intralayer energy from the results.
        :return: Intralayer energy.
        """
        if layer == 'L1':
            return self.intra_L1_energy
        elif layer == 'L2':
            return self.intra_L2_energy

    def get_interlayer_energy(self):
        """
        Retrieves interlayer energy from the results.
        :return: Interlayer energy.
        """
        return self.inter_energy

    def get_total_energy(self):
        """
        Retrieves total energy from the results.
        :return: Total energy.
        """
        return self.total_energy
    
    def get_total_forces(self):
        """
        Retrieves total forces from the results.
        :return: Total forces.
        """
        return self.total_forces

# Example usage:
# dft_calc = DFTCalculator("scf_L1.out", "scf_L2.out", "total_scf.out")
# total_energy = dft_calc.get_total_energy()
# interlayer_energy = dft_calc.get_interlayer_energy()
# interlayer_forces = dft_calc.get_interlayer_forces()
