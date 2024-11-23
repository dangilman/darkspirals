import numpy.testing as npt
import pytest
from darkspirals.disc import Disc
import astropy.units as apu
from galpy.potential import MWPotential2014
import numpy as np
from galpy.orbit import Orbit
from galpy.actionAngle.actionAngleVertical import actionAngleVertical

class TestDisc(object):

    def setup_method(self):

        z_min_max = 1.5
        vz_min_max = 100
        self.phase_space_resolution = 50
        galactic_potential = MWPotential2014
        self.n_time_steps = 1000
        self.time_Gyr = np.linspace(0.0, -1.0, self.n_time_steps) * apu.Gyr
        self.disc = Disc(galactic_potential, galactic_potential, z_min_max, vz_min_max, self.phase_space_resolution,
                    self.time_Gyr, parallelize_action_angle_computation=True, compute_upfront=False)

    def test_internal_units(self):

        ro = 8.0
        vo = 220.0
        npt.assert_almost_equal(ro, self.disc.units['ro'])
        npt.assert_almost_equal(vo, self.disc.units['vo'])

        time_in_Gyr = -0.5
        time_internal = self.disc.time_to_internal_time(time_in_Gyr)
        npt.assert_almost_equal(time_internal/ self.disc.time_internal_eval[int(self.n_time_steps/2)], 1, 2)

    def test_action_angle_freq(self):

        ro = 8.0
        vo = 220.0
        z_phys, vz_phys = 0.4, 35
        action, angle, freq = self.disc.action_angle_interp((z_phys, vz_phys))
        z_internal, vz_internal = z_phys / ro, vz_phys / vo
        orbit = Orbit(vxvv=[z_internal, vz_internal])
        orbit.integrate(self.disc.time_internal_eval, self.disc.local_vertical_potential)
        aav = actionAngleVertical(pot=self.disc.local_vertical_potential)
        jz, freq, theta = aav.actionsFreqsAngles(orbit.x(self.disc.time_internal_eval), orbit.vx(self.disc.time_internal_eval))
        npt.assert_almost_equal(jz[0], action, 2)
        npt.assert_almost_equal(theta[0], angle, 2)
        npt.assert_almost_equal(freq[0], freq, 2)



t = TestDisc()
t.setup_method()
t.test_internal_units()
t.test_action_angle_freq()
