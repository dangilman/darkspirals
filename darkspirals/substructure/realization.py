import numpy as np
from scipy.stats.kde import gaussian_kde
from darkspirals.substructure.galacticus_subhalo_data import galacticus_output
from galpy.util.coords import rect_to_cyl, rect_to_cyl_vec
from darkspirals.orbit_util import integrate_single_orbit
from galpy.potential import NFWPotential
from darkspirals.substructure.halo_util import sample_concentration_nfw, sample_mass_function
import astropy.units as apu
import matplotlib.pyplot as plt
from darkspirals.substructure.dphr import PopulationdSphr

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
        self.pop_dsphr = None

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

    def add_dwarf_galaxies(self, add_orbit_uncertainties=True,
                           additional_orbits=None, additional_mpeak=None, LMC_effect=False,
                           t_max=None):
        """

        :param tidal_mass_loss:
        :param add_orbit_uncertainties:
        :param additional_orbits:
        :param additional_mpeak:
        :param LMC_effect:
        :param t_max:
        :return:
        """

        pop_dsphr = PopulationdSphr()
        self.pop_dsphr = pop_dsphr
        dwarf_potentials = []
        dwarf_orbits = []
        dwarf_galaxy_masses = []
        dwarf_galaxy_names = []

        for name in pop_dsphr.dwarf_galaxy_names:

            m_peak = 10 ** pop_dsphr.log10_mpeak_from_name(name)
            potential = pop_dsphr.dsphr_potential_from_name(name)
            dwarf_potentials.append(potential)
            orbit_init = pop_dsphr.orbit_init_from_name(name, uncertainties=add_orbit_uncertainties)
            orbit = integrate_single_orbit(orbit_init,
                                           self._disc,
                                           pot=potential,
                                           radec=True)
            _, _ = orbit.orbit_parameters(self._disc, t_max)
            dwarf_orbits.append(orbit)
            dwarf_galaxy_masses.append(m_peak)
            dwarf_galaxy_names.append(name)

        if additional_orbits is not None:
            for name in additional_orbits:
                m_peak = 10 ** additional_mpeak[name]
                c = sample_concentration_nfw(m_peak)
                potential = NFWPotential(mvir=m_peak / 10 ** 12, conc=c)
                dwarf_potentials.append(potential)
                orb_init = additional_orbits[name]
                orbit = integrate_single_orbit(orb_init, self._disc, pot=potential, radec=True)
                _, _ = orbit.orbit_parameters(self._disc, t_max)
                dwarf_orbits.append(orbit)
                dwarf_galaxy_masses.append(m_peak)
                dwarf_galaxy_names.append(name)

        self._dwarf_galaxy_orbits = dwarf_orbits
        self._dwarf_galaxy_names = dwarf_galaxy_names
        self._dwarf_galaxy_potentials = dwarf_potentials
        self._dwarf_galaxy_masses = dwarf_galaxy_masses
        self._dwarf_galaxies_added = True
        return

    @classmethod
    def withDistanceCut(cls, disc, r_min, num_halos_scale=1.0,
                        norm=1200.0, alpha=-1.9, m_high=10**8, m_low=10**6,
                        num_halos=None, t_max=-1.2, verbose=False):
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
        :param t_max: discard subhalos with impact times greater than this
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
            c = sample_concentration_nfw(m)
            pot = NFWPotential(mvir=m / 10 ** 12, conc=c)
            orb = integrate_single_orbit(vxvv,
                                         disc,
                                         pot=pot,
                                         radec=False)
            impact_distance, impact_time = orb.orbit_parameters(disc, t_max)
            if impact_distance <= r_min and abs(impact_time) <= abs(t_max):
                orbits.append(orb)
                potentials.append(pot)
                subhalo_masses.append(m)
                orb.set_closest_approach(impact_distance, impact_time)
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
