import numpy.testing as npt
import pytest
from darkspirals.disc import Disc
import astropy.units as apu
from galpy.potential import MWPotential2014
import numpy as np
from darkspirals.orbit_util import integrate_single_orbit
from darkspirals.substructure.halo_util import sample_dwarf_galaxy_potential

class TestDisc(object):

    def setup_method(self):

        z_min_max = 1.5
        vz_min_max = 100
        self.phase_space_resolution = 50
        galactic_potential = MWPotential2014
        self.time_Gyr = np.linspace(-1.2, 0.0, 600) * apu.Gyr
        self.disc = Disc(galactic_potential, galactic_potential, z_min_max, vz_min_max, self.phase_space_resolution,
                    self.time_Gyr, parallelize_action_angle_computation=True, compute_upfront=True)

    def test_action_from_angle(self):

        action_eq, angle_eq = self.disc.action, self.disc.angle
        idx1, idx2 = 20, 15
        action_out = self.disc.action_from_angle(angle_eq[idx1, idx2])
        npt.assert_almost_equal(action_eq[idx1, idx2], action_out)

        idx1, idx2 = 0, 15
        action_out = self.disc.action_from_angle(angle_eq[idx1, idx2])
        npt.assert_almost_equal(action_eq[idx1, idx2], action_out)

t = TestDisc()
t.setup_method()
t.test_action_from_angle()
