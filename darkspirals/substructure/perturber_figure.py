from darkspirals.disc import Disc
from darkspirals.orbit_util import sample_sag_orbit, integrate_single_orbit
from darkspirals.substructure.realization import SubstructureRealization
from galpy.potential import MWPotential2014
import numpy as np
import astropy.units as apu
import matplotlib.pyplot as plt

np.random.seed(10)

z_min_max = 1.5
vz_min_max = 150
phase_space_resolution = 80
z = np.linspace(-z_min_max,z_min_max,phase_space_resolution)
vz = np.linspace(-vz_min_max, vz_min_max, phase_space_resolution)
galactic_potential = MWPotential2014
time_Gyr = np.linspace(-2.4, 0.0, 1200) * apu.Gyr

disc = Disc(galactic_potential, galactic_potential, z_min_max, vz_min_max, phase_space_resolution,
            time_Gyr, parallelize_action_angle_computation=True, compute_upfront=False)

r_min = 20
realization = SubstructureRealization.withDistanceCut(disc, norm=1200 * 1.0, r_min=r_min,
                                                      num_halos_scale=1.0, m_low=10**6.5, m_high=10**8)
realization.add_dwarf_galaxies(add_orbit_uncertainties=False,
                               )

print('realization contains '+str(len(realization.subhalo_orbits))+' subhalos')
print('realization contains '+str(len(realization.dwarf_galaxy_potentials))+' dwarf galaxies')

from palettable.cartocolors.qualitative import Vivid_10 as cmap_dwarfs
#from palettable.scientific.sequential import Oslo_20_r as cmap_dwarfs
#from palettable.mycarta import LinearL_10 as cmap_dwarfs

# print(cmap_dwarfs)
# plt.clf()
cmap_dwarfs = cmap_dwarfs.get_mpl_colormap()
cmap_dwarfs = plt.colormaps['jet']

co_rotating_frame = False
axis_size = 12.

fig = plt.figure(1)
fig.set_size_inches(9, 9)
ax = fig.add_subplot(111, projection='3d')
ax = realization.plot(lw_norm=3.0, grid_size=axis_size, lw_exp=0.8, solar_circle_color='g',
                      co_rotating_frame=co_rotating_frame, fig=fig, ax=ax, subhalo_alpha=0.6,
                      solar_circle_lw=6.5)

# for dwarf galaxies
lw_norm = 3.0
lw_exp = 0.5
dwarf_galaxy_alpha = 0.8
dwarf_galaxy_colors = [cmap_dwarfs(i / len(realization.dwarf_galaxy_orbits)) for i in
                       range(0, len(realization.dwarf_galaxy_orbits))]

for i, orbit in enumerate(realization.dwarf_galaxy_orbits):

    x_solar, y_solar = disc.solar_circle
    x1 = np.squeeze(orbit.x(disc.time_internal_eval))
    y1 = np.squeeze(orbit.y(disc.time_internal_eval))
    z1 = np.squeeze(orbit.z(disc.time_internal_eval))

    x1_present = np.squeeze(orbit.x(0.0))
    y1_present = np.squeeze(orbit.y(0.0))
    z1_present = np.squeeze(orbit.z(0.0))

    line_width = lw_norm * (realization.dwarf_galaxy_masses[i] / 10 ** 8.0) ** lw_exp
    alpha = 0.5
    label = realization._dwarf_galaxy_names[i]

    ax.plot(x1 * disc.units['ro'], y1 * disc.units['ro'], z1 * disc.units['ro'],
            alpha=dwarf_galaxy_alpha, lw=line_width, label=label, color=dwarf_galaxy_colors[i])
    ax.scatter(disc.units['ro'] * x1_present, disc.units['ro'] * y1_present, disc.units['ro'] * z1_present,
                   color=dwarf_galaxy_colors[i], s=32 * line_width * (80 / axis_size) ** 1.0, alpha=1.0)

sag_orbit_init, _ = sample_sag_orbit(0.0)
sag_orbit = integrate_single_orbit(sag_orbit_init, disc)
x_sag = sag_orbit.x(time_Gyr, **disc.units)
y_sag = sag_orbit.y(time_Gyr, **disc.units)
z_sag = sag_orbit.z(time_Gyr, **disc.units)
xf_sag = sag_orbit.x(0.0, **disc.units)
yf_sag = sag_orbit.y(0.0, **disc.units)
zf_sag = sag_orbit.z(0.0, **disc.units)
lw_sag = lw_norm * (10**8.5 / 10 ** 8.0) ** lw_exp
ax.plot(x_sag, y_sag, z_sag, color='m', lw=lw_sag, alpha=dwarf_galaxy_alpha,label='Sagittarius')
ax.scatter(xf_sag, yf_sag, zf_sag, color='m', s=65 * lw_sag, alpha=1.0)

plt.plot([-200,-250],[-200,-250],[-200,-250],color='k',alpha=0.9,label='dark subhalos',lw=4)
# leg = ax.legend(fontsize=14,loc=(0, 0.2))
# for line in leg.get_lines():
#     line.set_linewidth(4.0)
# leg.get_frame().set_alpha(1.0)

# ax.set_xlabel(r'$R \ \left[\rm{kpc}\right]$',fontsize=14)
# ax.set_ylabel(r'$R \ \left[\rm{kpc}\right]$',fontsize=14)
# ax.set_zlabel(r'$R \ \left[\rm{kpc}\right]$',fontsize=14)
# ticks = [-80,-40,0.0,40,80]
# ax.set_xticks(ticks)
# ax.set_xticklabels(ticks, fontsize=12)
# ax.set_yticks(ticks)
# ax.set_yticklabels(ticks, fontsize=12)
# ax.set_zticks(ticks)
# ax.set_zticklabels(ticks, fontsize=12)

# ticks = [-16,-8,0.0,8,16]
# ax.set_xticks(ticks)
# ax.set_xticklabels(ticks, fontsize=12)
# ax.set_yticks(ticks)
# ax.set_yticklabels(ticks, fontsize=12)
# ax.set_zticks(ticks)
# ax.set_zticklabels(ticks, fontsize=12)

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

ax.axis('off')

ax.axes.set_xlim3d(-axis_size, axis_size)
ax.axes.set_ylim3d(-axis_size, axis_size)
ax.axes.set_zlim3d(-axis_size, axis_size)
plt.tight_layout()
#plt.savefig('perturber_fig_zoomed_12.pdf')
#plt.show()
