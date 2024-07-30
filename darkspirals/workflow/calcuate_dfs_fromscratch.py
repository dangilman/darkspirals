from darkspirals.disc import Disc
from darkspirals.substructure.realization import SubstructureRealization
from galpy.potential import MWPotential2014
from darkspirals.orbit_util import sample_sag_orbit
from darkspirals.substructure.halo_util import sample_dwarf_galaxy_potential
import numpy as np
import astropy.units as apu
import matplotlib.pyplot as plt
from scipy.integrate import simps

from darkspirals.distribution_function.compute_df import compute_df_from_orbits
from darkspirals.diffusion import DiffusionConvolution
import pickle

np.random.seed(9291) # my birthday!

z_min_max = 1.5
vz_min_max = 150
phase_space_resolution = 100
z = np.linspace(-z_min_max,z_min_max,phase_space_resolution)
vz = np.linspace(-vz_min_max, vz_min_max, phase_space_resolution)
galactic_potential = MWPotential2014
time_Gyr = np.linspace(-2.4, 0.0, 3000) * apu.Gyr
LOAD_PRECOMPUTED = False
if LOAD_PRECOMPUTED:
    f = open('./cache/disc_highres', 'rb')
    disc = pickle.load(f)
    f.close()
else:
    disc = Disc(galactic_potential, galactic_potential, z_min_max, vz_min_max, phase_space_resolution,
                time_Gyr, parallelize_action_angle_computation=True, compute_upfront=True)

    f = open('./cache/disc_highres', 'wb')
    pickle.dump(disc, f)
    f.close()

r_min = 30
realization = SubstructureRealization.withDistanceCut(disc, norm=100 * 1.0, r_min=r_min,
                                                      num_halos_scale=1.0, m_low=10**6., m_high=10**8)

sag_orbit_init, _ = sample_sag_orbit(0.0)
additional_mpeak = {'Sagittarius': 8.5}
additional_orbits = {'Sagittarius': sag_orbit_init}

realization.add_dwarf_galaxies(add_orbit_uncertainties=False,
                              additional_mpeak=additional_mpeak,
                              additional_orbits=additional_orbits)

print('realization contains '+str(len(realization.subhalo_orbits))+' subhalos')
print('realization contains '+str(len(realization.dwarf_galaxy_potentials))+' dwarf galaxies')
satellite_orbit_list = realization.orbits
satellite_potential_list = realization.potentials

velocity_dispersion = 20.0

df_iso_eq, _ = compute_df_from_orbits(disc,
                           velocity_dispersion,
                           [],
                           [],
                           'ISOTHERMAL')
df_LW_eq, _ = compute_df_from_orbits(disc,
                           velocity_dispersion,
                           [],
                           [],
                           'LI&WIDROW')

df_model = 'ISOTHERMAL'
df_iso_no_diffusion, dJ_no_diffusion = compute_df_from_orbits(disc,
                           velocity_dispersion,
                           satellite_orbit_list,
                           satellite_potential_list,
                           df_model)


df_model = 'LI&WIDROW'
df_LW_no_diffusion, _ = compute_df_from_orbits(disc,
                           velocity_dispersion,
                           satellite_orbit_list,
                           satellite_potential_list,
                           df_model)

plt.imshow(df_iso_no_diffusion.function.T / df_iso_eq.function.T, origin='lower')
plt.show()
plt.imshow(df_LW_no_diffusion.function.T / df_LW_eq.function.T, origin='lower')
plt.show()

plt.plot(df_iso_no_diffusion.vertical_asymmetry)
plt.show()
plt.plot(df_LW_no_diffusion.vertical_asymmetry)
plt.show()

diffusion_model_class = DiffusionConvolution(disc)

df_iso_with_diffusion, dJ_with_diffusion = compute_df_from_orbits(disc,
                           velocity_dispersion,
                           satellite_orbit_list,
                           satellite_potential_list,
                            'ISOTHERMAL',
                           diffusion_model=diffusion_model_class
                           )
df_LW_with_diffusion, _ = compute_df_from_orbits(disc,
                           velocity_dispersion,
                           satellite_orbit_list,
                           satellite_potential_list,
                           'LI&WIDROW',
                           diffusion_model=diffusion_model_class
                           )

plt.imshow(df_iso_with_diffusion.function.T / df_iso_eq.function.T, origin='lower')
plt.show()
plt.imshow(df_LW_with_diffusion.function.T / df_LW_eq.function.T, origin='lower')
plt.show()

plt.plot(df_iso_with_diffusion.vertical_asymmetry)
plt.show()
plt.plot(df_LW_with_diffusion.vertical_asymmetry)
plt.show()
