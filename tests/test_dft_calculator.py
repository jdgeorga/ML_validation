import pytest
from dft_reader.dft_calculator import DFTCalculator

# Sample paths for your test SCF output files
scf_file_L1 = "scf_L1.out"
scf_file_L2 = "scf_L2.out"
total_scf_file = "scf_total.out"


@pytest.fixture
def dft_calc():
    """Fixture to create a DFTCalculator instance for testing."""
    return DFTCalculator(scf_file_L1, scf_file_L2, total_scf_file, read_dft=False)


def test_read_dft_data(dft_calc):
    dft_calc.read_dft_data()
    # Assuming forces and energies are stored in the calculator after reading data
    assert hasattr(dft_calc, 'total_forces'), "Total forces not calculated"
    assert hasattr(dft_calc, 'total_energy'), "Total energy not calculated"


def test_get_intralayer_forces(dft_calc):
    dft_calc.read_dft_data()
    forces_L1 = dft_calc.get_intralayer_forces("L1")
    forces_L2 = dft_calc.get_intralayer_forces("L2")
    assert forces_L1 is not None, "Failed to get intralayer forces for Layer 1"
    assert forces_L2 is not None, "Failed to get intralayer forces for Layer 2"


def test_get_interlayer_forces(dft_calc):
    dft_calc.read_dft_data()
    forces = dft_calc.get_interlayer_forces()
    assert forces is not None, "Failed to get interlayer forces"


def test_get_intralayer_energy(dft_calc):
    dft_calc.read_dft_data()
    energy_L1 = dft_calc.get_intralayer_energy("L1")
    energy_L2 = dft_calc.get_intralayer_energy("L2")
    assert energy_L1 is not None, "Failed to get intralayer energy for Layer 1"
    assert energy_L2 is not None, "Failed to get intralayer energy for Layer 2"


def test_get_interlayer_energy(dft_calc):
    dft_calc.read_dft_data()
    energy = dft_calc.get_interlayer_energy()
    assert energy is not None, "Failed to get interlayer energy"


def test_energy_sum(dft_calc):
    dft_calc.read_dft_data()
    intralayer_energy_L1 = dft_calc.get_intralayer_energy("L1")
    intralayer_energy_L2 = dft_calc.get_intralayer_energy("L2")
    interlayer_energy = dft_calc.get_interlayer_energy()
    total_energy = dft_calc.get_total_energy()
    assert intralayer_energy_L1 + intralayer_energy_L2 + interlayer_energy == total_energy, "Sum of intralayer and interlayer energies does not equal total energy"

# Add more tests as necessary to cover the full functionality of your DFTCalculator class
