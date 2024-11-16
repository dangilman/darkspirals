import astropy.table as table
import numpy as np
from darkspirals.substructure.halo_util import sample_concentration_nfw
from darkspirals.orbit_util import integrate_single_orbit
from galpy.potential import NFWPotential
from galpy.potential import turn_physical_off

dsph_mw = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/main/data/dwarf_mw.csv')

class PopulationdSphr(object):

    _dsph_mw = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/main/data/dwarf_mw.csv')
    dwarf_galaxy_names = [dsph_mw['key'][i] for i in range(0, 65)]
    dwarf_galaxy_ra = [dsph_mw['ra'][i] for i in range(0, 65)]
    dwarf_galaxy_dec = [dsph_mw['dec'][i] for i in range(0, 65)]
    dwarf_galaxy_pmra = [dsph_mw['pmra'][i] for i in range(0, 65)]
    dwarf_galaxy_pmdec = [dsph_mw['pmdec'][i] for i in range(0, 65)]
    dwarf_galaxy_R = [dsph_mw['distance'][i] for i in range(0, 65)]
    dwarf_galaxy_vlos = [dsph_mw['vlos_systemic'][i] for i in range(0, 65)]
    dwarf_galaxy_pmra_sigma = [dsph_mw['pmra_em'][i] for i in range(0, 65)]
    dwarf_galaxy_pmdec_sigma = [dsph_mw['pmdec_em'][i] for i in range(0, 65)]
    dwarf_galaxy_R_sigma = [dsph_mw['distance_em'][i] for i in range(0, 65)]
    dwarf_galaxy_vlos_sigma = [dsph_mw['vlos_systemic_em'][i] for i in range(0, 65)]
    # pmra_em, pmdec_em

    ra = {}
    dec = {}
    pmra = {}
    pmdec = {}
    R = {}
    vlos = {}
    ra_sigma = {}
    dec_sigma = {}
    pmra_sigma = {}
    pmdec_sigma = {}
    R_sigma = {}
    vlos_sigma = {}
    for i, name in enumerate(dwarf_galaxy_names):
        ra[name] = dwarf_galaxy_ra[i]
        dec[name] = dwarf_galaxy_dec[i]
        pmra[name] = dwarf_galaxy_pmra[i]
        pmdec[name] = dwarf_galaxy_pmdec[i]
        R[name] = dwarf_galaxy_R[i]
        vlos[name] = dwarf_galaxy_vlos[i]
        ra_sigma[name] = abs(0.001 * dwarf_galaxy_ra[i])
        dec_sigma[name] = abs(0.001 * dwarf_galaxy_dec[i])
        pmra_sigma[name] = dwarf_galaxy_pmra_sigma[i]
        pmdec_sigma[name] = dwarf_galaxy_pmdec_sigma[i]
        R_sigma[name] = dwarf_galaxy_R_sigma[i]
        vlos_sigma[name] = dwarf_galaxy_vlos_sigma[i]

    # my names are different from the names in the database
    name_matching = {'Willman I': 'willman_1',
           'Segue I': 'segue_1',
           'Segue II': 'segue_2',
           'Hercules': 'hercules_1',
           'Leo I': 'leo_1',
           'Leo II': 'leo_2',
           'Draco': 'draco_1',
           'Bootes I': 'bootes_1',
           'UrsaMinor': 'ursa_major_1',
           'UrsaMajor II': 'ursa_major_2',
          'Fornax':  'fornax_1',
          'Sculptor':  'sculptor_1',
           'Tucana III': 'tucana_3',
           'Tucana IV': 'tucana_4',
           'CanesVenatici I': 'canes_venatici_1',
           'CanesVenatici II': 'canes_venatici_2',
           'Reticulum III': 'reticulum_3',
          'Sagittarius': 'sagittarius_1'}

    M_V = {'Willman I': -2.53,
           'Segue I': -1.30,
           'Segue II': -1.86,
           'Leo I': -11.78,
           'Leo II': -9.74,
           'Draco': -8.71,
           'Bootes I': -6.02,
           'UrsaMinor': -9.03,
           'UrsaMajor II': -4.25,
           'Fornax': -13.46,
           'Sculptor': -10.82,
           'Hercules': -5.83,
           'Tucana III': -2.4,
           'Tucana IV': -3.5,
           'CanesVenatici I': -8.8,
           'CanesVenatici II': -5.17,
           'Reticulum III': -3.31,
           'Sagittarius': -13.5
           }
    m_peak = {
        'Willman I': 7.711677918523403, 'Segue I': 7.489376352511657, 'Segue II': 7.5905868215901755,
        'Hercules': 8.308096754164671, 'Leo I': 9.383457988123926, 'Leo II': 9.014762707909323,
        'Draco': 8.82860773799705, 'Bootes I': 8.342436020459168, 'UrsaMinor': 8.886442291756202,
        'UrsaMajor II': 8.022538644978852, 'Fornax': 9.687089395359479, 'Sculptor': 9.209954326846466,
        'Tucana III': 7.688182631058748, 'Tucana IV': 7.8869889096058365, 'CanesVenatici I': 8.84487370624181,
        'CanesVenatici II': 8.188812987036417, 'Reticulum III': 7.852649643311338,
        'Sagittarius (peak)': 9.694318714579373, 'Sagittarius': 8.7
    }

    @property
    def dwarf_galaxy_names(self):
        """

        :return: a list of dwarf galaxy names
        """
        return list(self.name_matching.keys())

    @staticmethod
    def _mv_to_mstar(mv):
        mstar = (mv - 6.758) / -2.7586
        return mstar

    @staticmethod
    def _mpeak_from_mstar(mstar):
        return 0.49857 * mstar + 6.732

    def log10_mpeak_from_mv(self, mv):
        """
        Compute the peak mass from absolve V-band mag

        Thanks to Ethan Nadler for providing these routines
        see Nadler et al. (2020)
        :param mv: absolute V-band magnitude
        :return: peak mass in log10(M_solar)
        """
        y = self._mpeak_from_mstar(self._mv_to_mstar(mv))
        return np.log10(0.2 * (10 ** y))

    def log10_mpeak_from_name(self, name):
        """
        Compute the peak mass from absolve V-band mag

        Thanks to Ethan Nadler for providing these routines
        see Nadler et al. (2020)
        :param name: dwarf galaxy name
        :return: peak mass in log10(M_solar)
        """
        return self.m_peak[name]

    def orbit_init_from_name(self, name, uncertainties=True):
        """
        Compute the initial conditions for the dwarf galaxy orbit at t=0 (today)
        :param name: dwarf galaxy name
        :param uncertainties: bool; add a random application of orbital uncertainties when initializing
        orbit parameters
        :return: a list of [ra, dec, R, pmra, pmdec, v_los] to create the Orbit class in galpy
        """
        catalog_name = self.name_matching[name]

        if uncertainties:
            orbit_init = [np.random.normal(self.ra[catalog_name], self.ra_sigma[catalog_name]),
                          np.random.normal(self.dec[catalog_name], self.dec_sigma[catalog_name]),
                          np.random.normal(self.R[catalog_name], self.R_sigma[catalog_name]),
                          np.random.normal(self.pmra[catalog_name], self.pmra_sigma[catalog_name]),
                          np.random.normal(self.pmdec[catalog_name], self.pmdec_sigma[catalog_name]),
                          np.random.normal(self.vlos[catalog_name], self.vlos_sigma[catalog_name])]
        else:
            orbit_init = [self.ra[catalog_name],
                      self.dec[catalog_name],
                      self.R[catalog_name],
                      self.pmra[catalog_name],
                      self.pmdec[catalog_name],
                      self.vlos[catalog_name]]
        return orbit_init

    def orbit_instances(self, disc, override_mpeak={}, orbit_uncertainties=True):
        """
        :param disc: an instance of Disc class
        :param names: dwarf galaxy names
        :param masses: dictionary of log10(mass), overrides the default m_peak
        :return: an instance of OrbitExtension
        """
        orbit_list = []
        for name in self.dwarf_galaxy_names:
            if name in list(override_mpeak.keys()):
                log10mass = override_mpeak[name]
            else:
                log10mass = self.m_peak[name]
            orbit_init = self.orbit_init_from_name(name, orbit_uncertainties)
            pot = self.dsphr_potential_from_mass(log10mass)
            orb = integrate_single_orbit(orbit_init, disc, pot)
            orbit_list.append(orb)
        return orbit_list

    def dsphr_potential_from_mass(self, log10_m):
        """

        :param m: mass in log10(M_solar)
        :return: galpy potential instance
        """
        m = 10 ** log10_m
        c = sample_concentration_nfw(m)
        pot = NFWPotential(mvir=m / 10 ** 12, conc=c)
        turn_physical_off(pot)
        return pot

    def dsphr_potential_from_name(self, name, log10mass_sagittarius=8.5):
        """
        Compute the gravitational potential from the name of the dsphr
        :param name: name of dwarf galaxy
        :return: galpy instance of NFW potential class
        """
        log10_mpeak = self.m_peak[name]
        return self.dsphr_potential_from_mass(log10_mpeak)
