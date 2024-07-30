from darkspirals.disc import Disc
from galpy.potential import MWPotential2014
import astropy.units as apu
import numpy as np
import pickle

z_min_max = 1.5
vz_min_max = 150
phase_space_resolution = 150
z = np.linspace(-z_min_max,z_min_max,phase_space_resolution)
vz = np.linspace(-vz_min_max, vz_min_max, phase_space_resolution)
galactic_potential = MWPotential2014
time_Gyr = np.linspace(-2.4, 0.0, 2400) * apu.Gyr

disc = Disc(galactic_potential, galactic_potential, z_min_max, vz_min_max, phase_space_resolution,
            time_Gyr, parallelize_action_angle_computation=False, compute_upfront=True)

f = open('local_potential_MWPot2014', 'wb')
pickle.dump(disc, f)
f.close()
