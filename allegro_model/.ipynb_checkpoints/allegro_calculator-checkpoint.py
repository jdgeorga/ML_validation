from ase.io import read
import numpy as np
from pathlib import Path
from .ase_allegro import NeuqIPMultiLayerCalculator


class AllegroCalculator:
    def __init__(self, xyz_file, intralayer_model_paths, interlayer_model_path, device='cuda'):
        """
        Initializes the AllegroCalculator with a specified computing device, atomic structure, and model paths.
        :param xyz_file: Path to the XYZ file containing the atomic structure.
        :param intralayer_model_paths: List of paths to intralayer model files.
        :param interlayer_model_path: Path to the interlayer model file.
        :param device: The device to use for calculations ('cuda' or 'cpu').
        """
        self.atoms = read(xyz_file)
        #atom_types = self._generate_atom_types_list(self.atoms)
        #Hardcoded
        atom_types = ["MoL1", "SL1", "SeL1", "MoL2", "SL2", "SeL2"]
        self.calc = NeuqIPMultiLayerCalculator(4,
                                               atom_types,
                                               device=device)

        # Ensure the model paths are converted to Path objects
        intralayer_paths = [Path(path) for path in intralayer_model_paths]
        interlayer_path = Path(interlayer_model_path)

        self.calc.setup_models(
            2,  # Assuming this is a fixed parameter for your setup
            intralayer_paths,
            [interlayer_path]
        )
        
         # Initialize attributes to store forces and energies
            
        self.total_forces = None
        self.interlayer_forces = None
        self.intralayer_forces = None
        self.total_energy = None
        self.intralayer_energy = None
        self.interlayer_energy = None
        
        self.calculate_forces_and_energy()

    def _generate_atom_types_list(self, atoms):
        """
        Generates a list of atom types from the ASE atoms object.
        :param atoms: ASE atoms object.
        :return: List of atom types.
        """
        atom_types = []
        layers = np.where(atoms.positions[:, 2] < atoms.positions[:, 2].mean(), 0, 1)
        for symbol, layer in zip(atoms.get_chemical_symbols(), layers):
            atom_types.append(f"{symbol}L{layer + 1}")  # +1 if layers are 0-indexed
        return list(set(atom_types))

    def load_structure(self, xyz_file):
        """
        Loads an atomic structure from an XYZ file and stores it internally.
        :param xyz_file: Path to the XYZ file.
        """
        self.atoms = read(xyz_file)

    def calculate_forces_and_energy(self):
        """
        Calculates forces and total energy for the internally stored atomic structure.
        """
        if self.atoms is None:
            raise ValueError("Atomic structure not loaded. Please load an XYZ file first.")

        self.atoms.calc = self.calc
        self.total_forces = self.atoms.get_forces()
        self.interlayer_forces = self.atoms.calc.results["forces_interlayer"]
        self.intralayer_forces = self.total_forces - self.interlayer_forces 
        self.total_energy = self.atoms.calc.results["energy"]
        self.atomic_energies = self.atoms.calc.results["energies"]
        self.interlayer_energy = self.calc.results["energy_interlayer"]
        self.intralayer_energy = self.total_energy -  self.interlayer_energy


    def get_intralayer_forces(self):
        """
        Retrieves intralayer forces from the results.
        :return: Intralayer forces.
        """
        return self.intralayer_forces

    def get_interlayer_forces(self):
        """
        Retrieves interlayer forces from the results.
        :return: Interlayer forces.
        """
        return self.interlayer_forces

    def get_intralayer_energy(self):
        """
        Retrieves intralayer energy from the results.
        :return: Intralayer energy.
        """
        return self.intralayer_energy

    def get_interlayer_energy(self):
        """
        Retrieves interlayer energy from the results.
        :return: Interlayer energy.
        """
        return self.interlayer_energy