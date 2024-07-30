from darkspirals.disc import Disc
from darkspirals.substructure.realization import SubstructureRealization
from galpy.potential import MWPotential2014
from darkspirals.orbit_util import sample_sag_orbit
import numpy as np
import astropy.units as apu
import matplotlib.pyplot as plt
from scipy.integrate import simps
import pickle

np.random.seed(9291) # my birthday!

z_min_max = 1.5
vz_min_max = 150
phase_space_resolution = 100
z = np.linspace(-z_min_max,z_min_max,phase_space_resolution)
vz = np.linspace(-vz_min_max, vz_min_max, phase_space_resolution)
galactic_potential = MWPotential2014
time_Gyr = np.linspace(-2.4, 0.0, 3000) * apu.Gyr
LOAD_PRECOMPUTED = True
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

LOAD_PRECOMPUTED_REALIZATION_1 = True
if LOAD_PRECOMPUTED_REALIZATION_1:
    f = open('./cache/realization_precomputed_1', 'rb')
    real1 = pickle.load(f)
    f.close()
else:
    r_min = 30
    real1 = SubstructureRealization.withDistanceCut(disc, norm=1000, r_min=r_min,
                                                    num_halos_scale=1.0, m_low=10 ** 5.5, m_high=10 ** 6.25)
    f = open('./cache/realization_precomputed_1', 'wb')
    pickle.dump(real1, f)
    f.close()

LOAD_PRECOMPUTED_REALIZATION_2 = True
if LOAD_PRECOMPUTED_REALIZATION_2:
    f = open('./cache/realization_precomputed_2', 'rb')
    real2 = pickle.load(f)
    f.close()
else:
    r_min = 30
    real2 = SubstructureRealization.withDistanceCut(disc, norm=1000, r_min=r_min,
                                                    num_halos_scale=1.0, m_low=10 ** 6.25, m_high=10 ** 8.0)
    f = open('./cache/realization_precomputed_2', 'wb')
    pickle.dump(real2, f)
    f.close()
#exit(1)
realization = SubstructureRealization.join(real1, real2)
#realization = SubstructureRealization.withDistanceCut(disc, norm=10, r_min=30,
#                                                    num_halos_scale=1.0, m_low=10 ** 6.25, m_high=10 ** 8.0)

sag_orbit_init, _ = sample_sag_orbit(0.0)
additional_mpeak = {'Sagittarius': 8.5}
additional_orbits = {'Sagittarius': sag_orbit_init}

realization.add_dwarf_galaxies(add_orbit_uncertainties=False,
                               additional_mpeak=additional_mpeak,
                               additional_orbits=additional_orbits)

print('realization contains '+str(len(realization.subhalo_orbits))+' subhalos')
print('realization contains '+str(len(realization.dwarf_galaxy_potentials))+' dwarf galaxies')

force_list = []
impulse_list_subhalos = []
force_list_dwarfs = []
impulse_list_dwarfs = []
mass_list = []
mass_list_dwarfs = []
deltaJ_list_subhalos = []
deltaJ_list_dwarfs = []

xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, phase_space_resolution), np.linspace(-1.0, 1.0, phase_space_resolution))
inds = np.where(xx ** 2 + yy ** 2 < np.sqrt(2))

j0 = np.mean(disc.action)

for subhalo_mass, subhalo_orbit, subhalo_potential in zip(realization.subhalo_masses,
                                                          realization.subhalo_orbits,
                                                          realization.subhalo_potentials):
    f = subhalo_orbit.force_exerted(disc, subhalo_potential, physical_units=True)
    dp = simps(np.absolute(f), time_Gyr.value) / 2.4
    impulse_list_subhalos.append(dp)
    mass_list.append(subhalo_mass)
    force_list.append(np.max(np.absolute(f)))
    dj = subhalo_orbit.deltaJ(disc, subhalo_potential, physical_units=False) / j0
    deltaJ_list_subhalos.append(np.mean(np.absolute(dj[inds])))

for dwarf_galaxy_mass, dwarf_galaxy_orbit, dwarf_galaxy_potential in zip(realization.dwarf_galaxy_masses,
                                                                         realization.dwarf_galaxy_orbits,
                                                                         realization.dwarf_galaxy_potentials):
    f = dwarf_galaxy_orbit.force_exerted(disc, dwarf_galaxy_potential, physical_units=True)
    mass_list_dwarfs.append(dwarf_galaxy_mass)
    force_list_dwarfs.append(np.max(np.absolute(f)))
    dp = simps(np.absolute(f), time_Gyr.value) / 2.4
    impulse_list_dwarfs.append(dp)
    dj = dwarf_galaxy_orbit.deltaJ(disc, dwarf_galaxy_potential, physical_units=False) / j0
    deltaJ_list_dwarfs.append(np.mean(np.absolute(dj[inds])))

plt.rcParams['axes.linewidth'] = 2.
plt.rcParams['xtick.major.width'] = 2.
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.minor.width'] = 1.0
plt.rcParams['xtick.minor.size'] = 3.5
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.minor.width'] = 1.0
plt.rcParams['ytick.minor.size'] = 3.5
plt.rcParams.update({'font.size': 16})
cmap_dwarfs = plt.colormaps['jet']
dwarf_galaxy_colors = [cmap_dwarfs(i / len(realization.dwarf_galaxy_orbits)) for i in
                       range(0, len(realization.dwarf_galaxy_orbits))]

j0 = np.mean(disc.action)
omega0 = np.mean(disc.frequency)
print(j0, omega0)
print(disc.action[50,50], disc.frequency[50,50])

f_array_list_subhalos = []
f_array_list_dwarfs = []

for subhalo_mass, subhalo_orbit, subhalo_potential in zip(realization.subhalo_masses,
                                                          realization.subhalo_orbits,
                                                          realization.subhalo_potentials):
    f_vs_t = subhalo_orbit.force_exerted(disc, subhalo_potential, physical_units=True)
    f_array_list_subhalos.append(f_vs_t)

for dwarf_galaxy_mass, dwarf_galaxy_orbit, dwarf_galaxy_potential in zip(realization.dwarf_galaxy_masses,
                                                                         realization.dwarf_galaxy_orbits,
                                                                         realization.dwarf_galaxy_potentials):
    f_vs_t = dwarf_galaxy_orbit.force_exerted(disc, dwarf_galaxy_potential, physical_units=True)
    f_array_list_dwarfs.append(f_vs_t)

f_net_subhalos = 0
f_net_subhalos_below_6 = 0
f_net_subhalos_6 = 0
f_net_subhalos_7 = 0
f_net_dwarfs = 0

minimum_subhalo_mass = 10 ** 6
for i in range(0, len(f_array_list_subhalos)):
    f_net_subhalos += f_array_list_subhalos[i]
    if mass_list[i] < 10 ** 6:
        f_net_subhalos_below_6 += f_array_list_subhalos[i]
        continue
    if mass_list[i] > 10 ** 6:
        f_net_subhalos_6 += f_array_list_subhalos[i]
    if mass_list[i] > 10 ** 7:
        f_net_subhalos_7 += f_array_list_subhalos[i]

i_max = len(f_array_list_dwarfs) - 1
for i in range(0, len(f_array_list_dwarfs)):
    if i == i_max:
        col = 'm'
    else:
        col = color = dwarf_galaxy_colors[i]
    f_net_dwarfs += f_array_list_dwarfs[i]
    # plt.plot(time_Gyr.value, f_array_list_dwarfs[i], color=col, alpha=1.)

fig = plt.figure(1)
fig.set_size_inches(11, 5)
ax = plt.subplot(111)
ax.plot(time_Gyr.value, f_net_subhalos_below_6, color='c', lw=3.0, label=r'$m_{\rm{sub}} < 10^6 M_{\odot}$')
ax.plot(time_Gyr.value, f_net_subhalos_7, color='b', lw=3.0, label=r'$m_{\rm{sub}} > 10^7 M_{\odot}$')
ax.plot(time_Gyr.value, f_net_subhalos_6, color='y', lw=3.0, label=r'$m_{\rm{sub}} > 10^6 M_{\odot}$')
ax.plot(time_Gyr.value, f_net_subhalos, color='k', lw=1.5, label=r'$m_{\rm{sub}} > 10^{5.5} M_{\odot}$')
leg = ax.legend(fontsize=12, loc=2, framealpha=1.)

ax.set_xlabel('time ' + r'$\left[\rm{Gyr}\right]$')
ax.set_ylabel(r'$\ F_{z} \ \left[\rm{2\pi G M_{\odot}} \rm{pc^{-2}}\right]$', fontsize=16)

ax.set_ylim(-0.4, 0.4)
ax.set_xlim(-2.4, 0.0)
plt.tight_layout()
plt.savefig('subhalo_force_figure_split.png')

f_net_subhalos = 0
f_net_dwarfs = 0

for i in range(0, len(f_array_list_subhalos)):
    plt.plot(time_Gyr.value, f_array_list_subhalos[i], color='k', alpha=0.5)
    f_net_subhalos += f_array_list_subhalos[i]

i_max = len(f_array_list_dwarfs) - 1
for i in range(0, len(f_array_list_dwarfs)):
    if i == i_max:
        col = 'm'
    else:
        col = color = dwarf_galaxy_colors[i]
    f_net_dwarfs += f_array_list_dwarfs[i]
    plt.plot(time_Gyr.value, f_array_list_dwarfs[i], color=col, alpha=1.)

fig = plt.figure(1)
fig.set_size_inches(11, 5)
ax = plt.subplot(111)
ax.plot(time_Gyr.value, f_net_subhalos, color='k',lw=2.2, label='all dark subhalos')
ax.plot(time_Gyr.value, f_net_dwarfs, color='r', lw=2.2, label='all dwarf galaxies')
ax.plot(time_Gyr.value, f_net_dwarfs + f_net_subhalos, color='g', lw=1.8, label='combined')
ax.legend(fontsize=14)
ax.set_xlabel('time ' + r'$\left[\rm{Gyr}\right]$')
ax.set_ylabel(r'$\ F_{z} \ \left[\rm{2\pi G M_{\odot}} \rm{pc^{-2}}\right]$', fontsize=16)

ax.set_ylim(-0.4, 0.4)
ax.set_xlim(-2.4, 0.0)
ax.set_xticks([-2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0.0])
ax.set_xticklabels([-2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0.0])

plt.tight_layout()
plt.savefig('force_figure_combined.png')
