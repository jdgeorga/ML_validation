import pytest
from lammps_model.lammps_calculator import LammpsCalculator

# Sample path for your test
xyz_file = "MoS2-Bilayer.xyz"


@pytest.fixture
def lammps_calc():
    """Fixture to create a LammpsCalculator instance for testing."""
    return LammpsCalculator(xyz_file, run_lammps=False)


def test_run_lammps(lammps_calc):
    lammps_calc.run_lammps()
    # Assuming forces and energies are stored in the calculator after simulation
    assert hasattr(lammps_calc, 'total_forces'), "Total forces not calculated"
    assert hasattr(lammps_calc, 'total_energy'), "Total energy not calculated"


def test_get_intralayer_forces(lammps_calc):
    lammps_calc.run_lammps()
    forces_L1 = lammps_calc.get_intralayer_forces()[0]
    forces_L2 = lammps_calc.get_intralayer_forces()[1]
    assert forces_L1 is not None, "Failed to get intralayer forces for Layer 1"
    assert forces_L2 is not None, "Failed to get intralayer forces for Layer 2"


def test_get_interlayer_forces(lammps_calc):
    lammps_calc.run_lammps()
    forces = lammps_calc.get_interlayer_forces()
    assert forces is not None, "Failed to get interlayer forces"


def test_get_intralayer_energy(lammps_calc):
    lammps_calc.run_lammps()
    energy_L1 = lammps_calc.get_intralayer_energy()[0]
    energy_L2 = lammps_calc.get_intralayer_energy()[1]
    assert energy_L1 is not None, "Failed to get intralayer energy for Layer 1"
    assert energy_L2 is not None, "Failed to get intralayer energy for Layer 2"


def test_get_interlayer_energy(lammps_calc):
    lammps_calc.run_lammps()
    energy = lammps_calc.get_interlayer_energy()
    assert energy is not None, "Failed to get interlayer energy"


def test_energy_sum(lammps_calc):
    lammps_calc.run_lammps()
    intralayer_energy_L1 = lammps_calc.get_intralayer_energy()[0]
    intralayer_energy_L2 = lammps_calc.get_intralayer_energy()[1]
    interlayer_energy = lammps_calc.get_interlayer_energy()
    total_energy = lammps_calc.total_energy  # Assuming total energy is stored like this
    assert intralayer_energy_L1 + intralayer_energy_L2 + interlayer_energy == total_energy, "Sum of intralayer and interlayer energies does not equal total energy"

# Add more tests as necessary to cover the full functionality of your LammpsCalculator class
