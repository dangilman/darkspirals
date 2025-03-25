import numpy as np
from scipy.stats.kde import gaussian_kde
from darkspirals.substructure.galacticus_subhalo_data import galacticus_output
from galpy.util.coords import rect_to_cyl, rect_to_cyl_vec
from darkspirals.orbit_util import integrate_single_orbit
from galpy.potential import NFWPotential, TwoPowerSphericalPotential
from darkspirals.substructure.halo_util import sample_concentration_nfw, sample_mass_function
import astropy.units as apu
import matplotlib.pyplot as plt
from darkspirals.substructure.dsphr import PopulationdSphr
from darkspirals.substructure.halo_util import mass_twopower
from astropy.cosmology import FlatLambdaCDM
import astropy.units as un

class SubstructureRealization(object):
    """
    This class is used to create and store the orbits of perturbers, both luminous and dark satellites.

    The main class methods include the class creation method "withDistanceCut" and the method "add_dwarf_galaxies".
    """
    def __init__(self, disc, orbits, potentials, subhalo_masses=None, dwarf_galaxies_added=False):
        """
        Instantiate the class from an instance of Disc, and orbits/potentials/masses of perturbers
        :param disc: an instance of Disc class
        :param orbits: a list of perturber orbits; can be either galpy Orbit classes or the OrbitExtension class in
        darkspirals (see orbit_util.py)
        :param potentials: a list of galpy potential instances, should be the same length as orbits
        :param subhalo_masses: the masses of the perturbers corresponding to potentials
        :param dwarf_galaxies_added: bool; keeps track of whether the class has had dwarf galaxies included upon creation
        """
        self._disc = disc
        self._subhalo_orbits = orbits
        self._subhalo_potentials = potentials
        self._dwarf_galaxy_orbits = []
        self._dwarf_galaxy_potentials = []
        self._subhalo_masses = subhalo_masses
        self._dwarf_galaxy_masses = []
        self.pop_dsphr = None
        self._dwarf_galaxies_added = dwarf_galaxies_added
        self._cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    @classmethod
    def join(cls, realization1, realization2):
        """
        Combines two SubstructureRealization into one
        :param realization1: one SubstructureRealization
        :param realization2: another SubstructureRealization class
        :return: one SubstructureRealization that includes perturbers from both realization1 and realization2
        """
        return SubstructureRealization(realization1._disc,
                                realization1.subhalo_orbits + realization2.subhalo_orbits,
                                realization1.subhalo_potentials + realization2.subhalo_potentials,
                                np.append(realization1.subhalo_masses, realization2.subhalo_masses),
                                dwarf_galaxies_added=False)

    @classmethod
    def withDistanceCut(cls, disc, r_min, num_halos_scale=1.0,
                        norm=1200.0, alpha=-1.9, m_high=10 ** 8, m_low=10 ** 6,
                        num_halos=None, t_max=None, model_disc_disruption=False,
                        density_profile='NFW', alpha_profile=None, beta_profile=None, verbose=False):
        """

        :param disc: an instance of the Disc class
        :param r_min: the minimum distance a subhalo comes from the galactic center
        :param num_halos_scale: linearly scales the number of halos, the same as increaing the normlaization
        :param norm: sets the overall normalization of the differential subhalo mass function
        :param alpha: the logarithmic slope of the differential subhalo mass function
        :param m_high: the upper mass limit for subhalos
        :param m_low: the lower mass limit for subhalos
        :param num_halos: number of halos to generate, overrides calculation based on norm parameter
        :param t_max: ignore perturbation by subhalos with impact times greater than this; should be a negative number
        specifying a time in the past in Gyr
        :param model_disc_disruption: bool; if True, will discard subhalos if they cross the disk midplane before their
        impact time
        :param density_profile: halo density profile model
        :param alpha: the inner slope of the halo density profile when using TWOPOWER density profile
        :param beta: the outer slope of the halo density profile when using TWOPOWER density profile
        :return: an instance of Realization that includes subhalos
        """
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
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
                    vR * apu.km / apu.s,
                    vT * apu.km / apu.s,
                    z_i * apu.kpc,
                    vz_i * apu.km / apu.s,
                    phi * 180 / np.pi * apu.deg]
            c = sample_concentration_nfw(m)
            if density_profile == 'NFW':
                pot = NFWPotential(mvir=m / 10 ** 12, conc=c)
            elif density_profile == 'TWOPOWER':
                amp = m / mass_twopower(rmax, 1.0, rs, alpha_profile, beta_profile)
                z_eval_rho_crit = 0
                rho_crit = un.Quantity(self.cosmo.astropy.critical_density(z_eval_rho_crit),
                                       unit=un.Msun / un.kpc ** 3).value
                r200_h = (3 * m * cosmo.h / (4 * np.pi * rho_crit * 200)) ** (1.0 / 3.0)
                r200 = r200_h / cosmo.h
                rs = r200/c
                a = float(rs / self._disc.units['ro'])
                pot = TwoPowerSphericalPotential(amp, a)
            else:
                raise Exception('density profile must be etiher NFW or GNFW')
            orb = integrate_single_orbit(vxvv,
                                         disc,
                                         pot=pot,
                                         radec=False)
            impact_distance, impact_time = orb.orbit_parameters(disc, t_max)
            if model_disc_disruption:
                crossing_times = orb.disc_crossing_times(disc)
                num_crossing = len(crossing_times)
                if num_crossing > 0:
                    t_cross = abs(crossing_times[-1])
                    if t_cross > impact_time + 0.2:
                        disrupted = True
                    else:
                        disrupted = False
                else:
                    disrupted = False
            else:
                disrupted = False
            if t_max is None:
                t_max = -10000
            if r_min is None:
                r_min = 1e9
            if impact_distance <= r_min and abs(impact_time) <= abs(t_max):
                if disrupted is False:
                    orbits.append(orb)
                    potentials.append(pot)
                    subhalo_masses.append(m)
                    orb.set_closest_approach(impact_distance, impact_time)
            if verbose and counter % 25 == 0:
                percent_done = int(100 * counter / len(_subhalo_masses))
                print('completed ' + str(percent_done) + '% of force calculations... ')
        return SubstructureRealization(disc, orbits, potentials, np.array(subhalo_masses))


    def add_dwarf_galaxies(self, add_orbit_uncertainties=True,
                           additional_orbits=None, additional_mpeak=None, LMC_effect=False,
                           t_max=None, log10_dsphr_masses={}, tidal_stripping=False, stripping_factor=10,
                           include_dwarf_list=None):
        """
        Add a population of dwarf galaxies; the default masses and kinematics are stored in the PopulationdSphr class
        (see dsphr.py)

        :param add_orbit_uncertainties: bool; add orbit uncertainties to the dwarf galaxies
        :param additional_orbits: an additional list of objects to include; each object name in additional_orbits should
        have a corresponding mass specified in additional_mpeak
        :param additional_mpeak: a dictionary that contains the log10(masses) corresponding to each object name in additional_orbits
        For example:
        additional_orbits = ['thing1', 'thing2']
        additinal_mpeak = {'thing1': 8.0, 'thing2': 8.7}
        :param LMC_effect: turn on/off LMC effects; note that this argument is not currently in use
        :param t_max: ignores any perturbation at t>|t_max| Gyr in the past; t_max should be a negative number in units of
        Gyr
        :param log10_dsphr_masses: overrides the default dSphr. masses stored in the PopulationdSphr class (see dsphr.py)
        :param tidal_stripping: bool; if True, adds random rescaling factors to dwarf galaxies between (1/stripping_factor - 1)
        :param stripping_factor: see above
        :param include_dwarf_list: a list of dwarf galaxies to include, overrides the default list in PopulationdSphr
        """
        pop_dsphr = PopulationdSphr()
        self.pop_dsphr = pop_dsphr
        dwarf_potentials = []
        dwarf_orbits = []
        dwarf_galaxy_masses = []
        dwarf_galaxy_names = []
        if include_dwarf_list is None:
            include_dwarf_list = pop_dsphr.dwarf_galaxy_names
        for name in include_dwarf_list:

            if name in list(log10_dsphr_masses.keys()):
                m_peak = 10 ** log10_dsphr_masses[name]
            else:
                m_peak = 10 ** pop_dsphr.log10_mpeak_from_name(name)
            if tidal_stripping:
                if name in list(log10_dsphr_masses.keys()):
                    print(name, np.log10(m_peak))
                    pass
                else:
                    tidal_stripping_factor = np.random.uniform(1/stripping_factor, 1.0)
                    m_peak *= tidal_stripping_factor
                    print(name, np.log10(m_peak), tidal_stripping_factor)
            potential = pop_dsphr.dsphr_potential_from_mass(np.log10(m_peak))
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


    def plot(self, grid_size=14, co_rotating_frame=False,
             subhalo_orbit_color='k', solar_circle_color='0.6', subhalo_alpha=0.5, lw_norm=4.0, lw_exp=1.0,
             axis_size=40, fig_size=8, ax=None, fig=None, label_impact_location=False, solar_circle_lw=2.5):
        """
        Makes a plot of the realization
        :param grid_size:
        :param co_rotating_frame:
        :param subhalo_orbit_color:
        :param solar_circle_color:
        :param subhalo_alpha:
        :param lw_norm:
        :param lw_exp:
        :param axis_size:
        :param fig_size:
        :param ax:
        :param fig:
        :param label_impact_location:
        :param solar_circle_lw:
        :return:
        """
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

