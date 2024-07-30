from darkspirals.disc import Disc
from darkspirals.substructure.realization import SubstructureRealization
from galpy.potential import MWPotential2014
from darkspirals.workflow.cache_class import CachedSingleRealization
import pickle
import numpy as np
import astropy.units as apu
from darkspirals.diffusion import DiffusionConvolution
import sys
# z_min_max = 1.5
# vz_min_max = 150
# phase_space_resolution = 150
# z = np.linspace(-z_min_max,z_min_max,phase_space_resolution)
# vz = np.linspace(-vz_min_max, vz_min_max, phase_space_resolution)
# galactic_potential = MWPotential2014
# time_Gyr = np.linspace(-2.4, 0.0, 2400) * apu.Gyr
# print('creating disc instance... ')
# disc = Disc(galactic_potential, galactic_potential, z_min_max, vz_min_max, phase_space_resolution,
#             time_Gyr, parallelize_action_angle_computation=True, compute_upfront=True)


index_realization = int(sys.argv[1])
np.random.seed(index_realization)

z_min_max = 1.5
vz_min_max = 100
phase_space_resolution = 76
z = np.linspace(-z_min_max,z_min_max,phase_space_resolution)
vz = np.linspace(-vz_min_max, vz_min_max, phase_space_resolution)
galactic_potential = MWPotential2014
time_Gyr = np.linspace(-2.4, 0.0, 1200) * apu.Gyr
LOAD_PRECOMPUTED = False
if LOAD_PRECOMPUTED:
    f = open('./cache/disc_highres', 'rb')
    disc = pickle.load(f)
    f.close()

else:
    disc = Disc(galactic_potential, galactic_potential, z_min_max, vz_min_max, phase_space_resolution,
                time_Gyr, parallelize_action_angle_computation=False, compute_upfront=True)

    f = open('./cache/disc_highres_250', 'wb')
    pickle.dump(disc, f)
    f.close()
    exit(1)

print('creating realization... ')
r_min = 30
cached = CachedSingleRealization([], [], [], [], [])
n_split = 1

diffusion_model = DiffusionConvolution(disc)
deltaJ_with_diffusion_iso = []
deltaJ_with_diffusion_LW = []
for _ in range(0, n_split):
    realization = SubstructureRealization.withDistanceCut(disc, norm=200 / n_split,
                                                      r_min=r_min,
                                                      num_halos_scale=1.0,
                                                      m_low=10**5.7,
                                                      m_high=10**8)
    print('realization contains '+str(len(realization.subhalo_orbits))+' halos')
    print('computing forces... ')
    force_list, impact_times, _ = disc.compute_satellite_forces(realization.subhalo_orbits,
                                                                realization.subhalo_potentials,
                                                                verbose=True)
    print('computing action perturbations... ')
    deltaJ_list = disc.compute_deltaJ_from_forces(force_list, verbose=True)

    print('applying diffusion model')
    for dj, timpact in zip(deltaJ_list, impact_times):
        deltaJ_with_diffusion_iso.append(diffusion_model(dj, timpact, df_model='ISOTHERMAL'))
        deltaJ_with_diffusion_LW.append(diffusion_model(dj, timpact, df_model='LI&WIROW'))
    cached.force_list += force_list
    cached.deltaJ_list += deltaJ_list
    cached.impact_times += list(impact_times)
    cached.deltaJ_diffusion_iso += deltaJ_with_diffusion_iso
    cached.deltaJ_diffusion_LW += deltaJ_with_diffusion_LW

f = open('./cache/realization_'+str(index_realization), 'wb')
pickle.dump(cached, f)
f.close()
