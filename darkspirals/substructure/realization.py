import numpy as np
from scipy.stats.kde import gaussian_kde
from darkspirals.substructure.galacticus_subhalo_data import galacticus_output, number_of_realizations
from galpy.util.coords import rect_to_cyl, rect_to_cyl_vec
from darkspirals.orbit_util import integrate_single_orbit, sample_dwarf_orbit_init
from galpy.potential import NFWPotential
from darkspirals.substructure.halo_util import sample_concentration_nfw, sample_mass_function
from darkspirals.orbit_util import orbit_closest_approach
import astropy.units as apu
import matplotlib.pyplot as plt

class SubstructureRealization(object):

    def __init__(self, disc, orbits, potentials, subhalo_masses, dwarf_galaxies_added=False):

        self._disc = disc
        self._subhalo_orbits = orbits
        self._subhalo_potentials = potentials
        self._dwarf_galaxy_orbits = []
        self._dwarf_galaxy_potentials = []
        self._dwarf_galaxies_added = dwarf_galaxies_added
        self._subhalo_masses = subhalo_masses
        self._dwarf_galaxy_masses = []
        self._dwarf_galaxies_added = False

    @classmethod
    def join(cls, realization1, realization2):

        return SubstructureRealization(realization1._disc,
                                realization1.subhalo_orbits + realization2.subhalo_orbits,
                                realization1.subhalo_potentials + realization2.subhalo_potentials,
                                np.append(realization1.subhalo_masses, realization2.subhalo_masses),
                                dwarf_galaxies_added=False)

    @property
    def dwarf_galaxy_names(self):
        if self._dwarf_galaxies_added:
            return self._dwarf_galaxy_names
        else:
            raise Exception('population of dwarf galaxies not yet specified for the class')

    @property
    def dwarf_galaxy_masses(self):
        return self._dwarf_galaxy_masses

    @property
    def subhalo_masses(self):
        return np.array(self._subhalo_masses)

    @property
    def orbits(self):
        return self._subhalo_orbits + self._dwarf_galaxy_orbits

    @property
    def potentials(self):
        return self._subhalo_potentials + self._dwarf_galaxy_potentials

    @property
    def subhalo_orbits(self):
        return self._subhalo_orbits

    @property
    def subhalo_potentials(self):
        return self._subhalo_potentials

    @property
    def dwarf_galaxy_orbits(self):
        return self._dwarf_galaxy_orbits

    @property
    def dwarf_galaxy_potentials(self):
        return self._dwarf_galaxy_potentials

    def add_dwarf_galaxies(self, tidal_mass_loss=0.0, add_orbit_uncertainties=True,
                           additional_orbits=None, additional_mpeak=None,
                           include_dwarf_list='DEFAULT', log10_m_peak_dict={}, LMC_effect=False):

        if self._dwarf_galaxies_added is False:
            if include_dwarf_list == 'DEFAULT':
                include_dwarf_list = ['Sculptor', 'LeoI', 'LeoII', 'Fornax', 'UrsaMinor', 'UrsaMajorII', 'Draco',
                                          'Willman1', 'BootesI',
                                          'SegueI', 'SegueII', 'Hercules', 'TucanaIII']
                log10_m_peak_dict = {'Willman1': 7.711677918523403, 'SegueI': 7.489376352511657, 'SegueII': 7.5905868215901755,
                                   'Hercules': 8.308096754164671, 'LeoI': 9.383457988123926, 'LeoII': 9.014762707909323,
                                   'Draco': 8.82860773799705, 'BootesI': 8.342436020459168, 'UrsaMinor': 8.886442291756202,
                                   'UrsaMajorII': 8.022538644978852, 'Sculptor': 9.209954326846466, 'Fornax': 9.687089395359479,
                                     'TucanaIII': 7.7}
            else:
                assert len(include_dwarf_list) == len(log10_m_peak_dict.keys())

            dwarf_potentials = []
            orbits = []
            dwarf_galaxy_masses = []
            if LMC_effect:
                LMC_mass = 1.38 * 10 ** 11
                c = sample_concentration_nfw(LMC_mass)
                lmc_pot = NFWPotential(mvir=LMC_mass / 10 ** 12, conc=c)
                lmc_orbit = sample_dwarf_orbit_init('LMC', self._disc.units, False)
                lmc_orbit.integrate(self._disc.time_internal_eval, self._disc.galactic_potential)
                lmc_orbit.turn_physical_off()
            else:
                lmc_orbit = None
                lmc_pot = None
            for name in include_dwarf_list:
                m = 10**(log10_m_peak_dict[name] + tidal_mass_loss)
                c = sample_concentration_nfw(m)
                pot = NFWPotential(mvir=m / 10 ** 12, conc=c)
                dwarf_potentials.append(pot)
                orb_init = sample_dwarf_orbit_init(name, self._disc.units, add_orbit_uncertainties)
                orbit = integrate_single_orbit(orb_init,
                                               self._disc,
                                               lmc_orbit=lmc_orbit,
                                               lmc_potential=lmc_pot)
                impact_distance, impact_time = orbit.closest_approach, orbit.impact_time
                orbit.set_closest_approach(impact_distance, impact_time)
                orbits.append(orbit)
                dwarf_galaxy_masses.append(m)

            if additional_orbits is not None:
                for name in additional_orbits:
                    include_dwarf_list.append(name)
                    log10_m_peak_dict[name] = additional_mpeak[name]
                    m = 10 ** (log10_m_peak_dict[name] + tidal_mass_loss)
                    c = sample_concentration_nfw(m)
                    pot = NFWPotential(mvir=m / 10 ** 12, conc=c)
                    dwarf_potentials.append(pot)
                    if additional_orbits[name] is None:
                        orb_init = sample_dwarf_orbit_init(name, self._disc.units, add_orbit_uncertainties)
                    else:
                        orb_init = additional_orbits[name]
                    orbit = integrate_single_orbit(orb_init, self._disc)
                    impact_distance, impact_time = orbit.closest_approach, orbit.impact_time
                    orbit.set_closest_approach(impact_distance, impact_time)
                    orbits.append(orbit)
                    dwarf_galaxy_masses.append(m)

            self._dwarf_galaxy_orbits = orbits
            self._dwarf_galaxy_names = include_dwarf_list
            self._dwarf_galaxy_potentials = dwarf_potentials
            self._dwarf_galaxy_masses = dwarf_galaxy_masses
            self._dwarf_galaxies_added = True
        return

    @classmethod
    def withDistanceCut(cls, disc, r_min, num_halos_scale=1.0,
                        norm=1200.0, alpha=-1.9, m_high=10**8, m_low=10**6,
                        num_halos=None, max_impact_time=1.8, verbose=False):
        """

        :param disc: an instance of the Disc class
        :param r_min: the minimum distance a subhalo comes from the galactic center
        :param num_halos_scale: scales the number of halos/normalization
        :param norm: the normalization of the SHMF; this parameter is chosen by sampling orbits from the kde, integrating them,
        and then selecting objects that pass within 40 kpc of the solar position. Galacticus predicts ~ 370 objects that pass within
        40 kpc of the galactic center with mass between 10^6.7 and 10^9. Setting norm = 1000 reproduces this number of objects from the method used to sample orbits
        :param alpha: the logarithmic slope of the differential subhalo mass function
        :param m_high: the upper mass limit for subhalos
        :param m_low: the lower mass limit for subhalos
        :param num_halos: number of halos to generate, overrides calculation based on norm parameter
        :param max_impact_time: discard subhalos with impact times greater than this
        :return: an instance of Realization
        """
        _subhalo_masses = sample_mass_function(num_halos_scale * norm,
                                              alpha,
                                              m_high,
                                              m_low,
                                              num_halos)
        num_halos = len(_subhalo_masses)
        kde = gaussian_kde(galacticus_output.T)
        _, _, x, y, z, vx, vy, vz = kde.resample(num_halos)
        potentials = []
        orbits = []
        subhalo_masses = []
        for counter, (x_i, y_i, z_i, vx_i, vy_i, vz_i, m) in enumerate(zip(x, y, z, vx, vy, vz, _subhalo_masses)):
            vR, vT, vz = rect_to_cyl_vec(vx_i, vy_i, vz_i, x_i, y_i, z_i)
            R, phi, z = rect_to_cyl(x_i, y_i, z_i)
            vxvv = [R * apu.kpc,
                    vR * apu.km/apu.s,
                    vT * apu.km/apu.s,
                    z_i * apu.kpc,
                    vz_i * apu.km/apu.s,
                    phi * 180/np.pi * apu.deg]
            orb = integrate_single_orbit(vxvv, disc,
                                         radec=False,
                                         )
            impact_distance, impact_time = orbit_closest_approach(disc, orb)
            if impact_distance <= r_min and abs(impact_time) <= max_impact_time:
                orbits.append(orb)
                c = sample_concentration_nfw(m)
                potentials.append(NFWPotential(mvir=m / 10 ** 12, conc=c))
                subhalo_masses.append(m)
            if verbose and counter%25==0:
                percent_done = int(100*counter/len(_subhalo_masses))
                print('completed '+str(percent_done)+'% of force calculations... ')
        return SubstructureRealization(disc, orbits, potentials, np.array(subhalo_masses))

    def plot(self, grid_size=14, co_rotating_frame=False,
             subhalo_orbit_color='k', solar_circle_color='0.6', subhalo_alpha=0.5, lw_norm=4.0, lw_exp=1.0,
             axis_size=40, fig_size=8, ax=None, fig=None, label_impact_location=False, solar_circle_lw=2.5):

        if fig is None:
            fig = plt.figure(1)
            fig.set_size_inches(fig_size, fig_size)
            ax = fig.add_subplot(111, projection='3d')

        x_solar, y_solar = self._disc.solar_circle
        dr_list = []
        r_solar = np.array([self._disc.units['ro']]*15)
        theta_solar = np.linspace(0, 2*np.pi, len(r_solar))

        for i, orbit in enumerate(self._subhalo_orbits):

            x1 = np.squeeze(orbit.x(self._disc.time_internal_eval))
            y1 = np.squeeze(orbit.y(self._disc.time_internal_eval))
            z1 = np.squeeze(orbit.z(self._disc.time_internal_eval))

            x1_present = np.squeeze(orbit.x(0.0))
            y1_present = np.squeeze(orbit.y(0.0))
            z1_present = np.squeeze(orbit.z(0.0))

            dr = np.sqrt((x1 - x_solar) ** 2 + (y1 - y_solar) ** 2 + z1**2)
            idxmin = np.argsort(dr)[0]
            dr_min = dr[idxmin]
            dr_list.append(dr_min)
            line_width = lw_norm * (self.subhalo_masses[i] / 10 ** 8.0) ** lw_exp
            label = None
            if co_rotating_frame:
                x_solar_present = x_solar[-1]
                y_solar_present = y_solar[-1]
                z_solar_present = 0.0
                ax.scatter(self._disc.units['ro'] * (x1[idxmin] - x_solar[idxmin]),
                           self._disc.units['ro'] * (y1[idxmin] - y_solar[idxmin]),
                           self._disc.units['ro'] * z1[idxmin], color='r',
                           marker='+')
                ax.annotate(str(dr),
                            xy=(self._disc.units['ro'] * (x1[idxmin] - x_solar[idxmin]),
                                self._disc.units['ro'] * (y1[idxmin] - y_solar[idxmin])),
                            fontsize=12)
                ax.plot((x1 - x_solar) * self._disc.units['ro'], (y1 - y_solar) * self._disc.units['ro'], z1 * self._disc.units['ro'],
                            alpha=subhalo_alpha, lw=line_width, label=label, color=subhalo_orbit_color)
                ax.scatter(self._disc.units['ro'] * (x1_present-x_solar_present), self._disc.units['ro'] * (y1_present-y_solar_present), self._disc.units['ro'] * (z1_present-z_solar_present),
                           color=subhalo_orbit_color, s=15 * line_width, alpha=subhalo_alpha)
                ax.scatter(0., 0., 0., color=subhalo_orbit_color)

            else:

                ax.plot(x1 * self._disc.units['ro'], y1 * self._disc.units['ro'], z1 * self._disc.units['ro'],
                        alpha=subhalo_alpha, lw=line_width, label=label, color=subhalo_orbit_color)
                if self._disc.units['ro'] * x1_present < -axis_size and self._disc.units['ro'] * z1_present > axis_size:
                    pass
                else:
                    ax.scatter(self._disc.units['ro'] * x1_present, self._disc.units['ro'] * y1_present, self._disc.units['ro'] * z1_present,
                           color=subhalo_orbit_color, s=15 * line_width * (80 / grid_size) ** 1.5, alpha=0.8)
                if label_impact_location:
                    ax.scatter(self._disc.units['ro'] * x1[idxmin],
                               self._disc.units['ro'] * y1[idxmin],
                               self._disc.units['ro'] * z1[idxmin],
                               color='r',
                               s=6.5,
                               alpha=1.0,
                               marker='+')


        ax.axes.set_xlim3d(-grid_size, grid_size)
        ax.axes.set_ylim3d(-grid_size, grid_size)
        ax.axes.set_zlim3d(-grid_size, grid_size)
        if co_rotating_frame:
            ax.scatter(0.0, 0.0, 0.0,
                       color=solar_circle_color, label='solar circle', s=80)
        else:
            ax.plot(r_solar * np.cos(theta_solar), r_solar * np.sin(theta_solar), 0.,
                    color=solar_circle_color, label='solar circle', lw=solar_circle_lw, linestyle='-')

        return ax
