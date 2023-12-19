import sys

print(sys.path)
import pytest
from allegro_model.allegro_calculator import AllegroCalculator

# Sample paths for your test
xyz_file = "MoS2-Bilayer.xyz"
intralayer_model_paths = ["./intralayer_beefy.pth", "./intralayer_beefy.pth"]
interlayer_model_path = "./interlayer_beefy_truncated.pth"

@pytest.fixture
def allegro_calc():
    """ Fixture to create an AllegroCalculator instance for testing. """
    return AllegroCalculator(xyz_file, intralayer_model_paths, interlayer_model_path, device='cuda')

def test_load_structure(allegro_calc):
    # Assuming the load_structure method is public and can be tested directly
    allegro_calc.load_structure(xyz_file)
    assert allegro_calc.atoms is not None, "Failed to load atoms from XYZ file"

def test_calculate_forces_and_energy(allegro_calc):
    allegro_calc.calculate_forces_and_energy()
    # Assuming forces and energies are stored in the calculator after calculation
    assert hasattr(allegro_calc, 'total_forces'), "Forces not calculated"
    assert hasattr(allegro_calc, 'total_energy'), "Energy not calculated"

def test_get_intralayer_forces(allegro_calc):
    forces = allegro_calc.get_intralayer_forces()
    assert forces is not None, "Failed to get intralayer forces"

def test_get_interlayer_forces(allegro_calc):
    forces = allegro_calc.get_interlayer_forces()
    assert forces is not None, "Failed to get interlayer forces"

def test_get_intralayer_energy(allegro_calc):
    energy = allegro_calc.get_intralayer_energy()
    assert energy is not None, "Failed to get intralayer energy"

def test_get_interlayer_energy(allegro_calc):
    energy = allegro_calc.get_interlayer_energy()
    assert energy is not None, "Failed to get interlayer energy"

def test_energy_sum(allegro_calc):
    allegro_calc.calculate_forces_and_energy()
    intralayer_energy = allegro_calc.get_intralayer_energy()
    interlayer_energy = allegro_calc.get_interlayer_energy()
    total_energy = allegro_calc.total_energy  # Assuming total energy is stored like this
    assert intralayer_energy + interlayer_energy == total_energy, "Intralayer and interlayer energies do not sum up to the total energy"

# Add more tests as necessary to cover the full functionality of your AllegroCalculator class
