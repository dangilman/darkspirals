import astropy.table as table
import numpy as np
from darkspirals.substructure.halo_util import sample_concentration_nfw
from darkspirals.orbit_util import integrate_single_orbit
from galpy.potential import NFWPotential
from galpy.potential import turn_physical_off

# dsph_mw = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/main/data/dwarf_mw.csv')

class PopulationdSphr(object):
    """
    This class is compiled from the local volume database (Pace 2024): https://github.com/apace7/local_volume_database
    """
    dwarf_galaxy_names = [
        'antlia_2', 'aquarius_2', 'aquarius_3', 'bootes_1', 'bootes_2', 'bootes_3', 'bootes_4', 'bootes_5',
         'canes_venatici_1', 'canes_venatici_2', 'carina_1', 'carina_2', 'carina_3', 'centaurus_1', 'cetus_2',
         'cetus_3', 'columba_1', 'coma_berenices_1', 'crater_2', 'draco_1', 'draco_2', 'eridanus_2', 'eridanus_4',
         'fornax_1', 'grus_1', 'grus_2', 'hercules_1', 'horologium_1', 'horologium_2', 'hydra_2', 'hydrus_1', 'leo_1',
         'leo_2', 'leo_4', 'leo_5', 'leo_6', 'leo_minor_1', 'lmc', 'pegasus_3', 'pegasus_4', 'phoenix_2', 'pictor_1',
         'pictor_2', 'pisces_2', 'reticulum_2', 'reticulum_3', 'sagittarius_1', 'sculptor_1', 'segue_1', 'segue_2',
         'sextans_1', 'sextans_2', 'smc', 'triangulum_2', 'tucana_2', 'tucana_3', 'tucana_4', 'tucana_5',
         'ursa_major_1', 'ursa_major_2', 'ursa_minor_1', 'virgo_1', 'virgo_2', 'virgo_3', 'willman_1'
    ]
    dwarf_galaxy_ra = [
        143.8079, 338.4813, 357.218, 210.02, 209.5141, 209.3, 233.689, 213.909, 202.0091, 194.2927, 100.4065, 114.1066,
         114.6298, 189.585, 19.47, 31.331, 82.85696, 186.7454, 177.31, 260.0684, 238.1983333, 56.0925, 76.438, 39.9583,
         344.166, 331.025, 247.7722, 43.8755, 49.131, 185.4286, 37.389, 152.1146, 168.3627, 173.2405, 172.7857, 171.077,
         164.261, 80.894, 336.1074167, 328.539, 354.9919, 70.9475, 101.18, 344.6364583, 53.9203, 56.36,
         284.095116792971, 15.0183, 151.7504, 34.8226, 153.2628, 156.437, 13.158, 33.3155, 342.9796, 359.1075, 0.717,
         354.347, 158.7706, 132.8726, 227.242, 180.038, 225.059, 186.348, 162.3436
    ]
    dwarf_galaxy_dec = [
        -36.6991, -9.3274, -3.489, 14.5135, 12.8553, 26.8, 43.726, 32.914, 33.5521, 34.3226, -50.9593, -57.9991,
         -57.8997, -40.902, -17.42, -4.27, -28.04253, 23.9069, -18.413, 57.9185, 64.56527778, -43.5329, -9.515,
         -34.4997, -50.168, -46.442, 12.7852, -54.1174, -50.009, -31.9728, -79.3089, 12.3059, 22.1529, -0.5453, 2.2194,
         24.874, 28.875, -69.7561, 5.4150472, 26.62, -54.4019, -50.28305556, -59.897, 5.9555444, -54.0513, -60.45,
         -30.549906, -33.7186, 16.0756, 20.1624, -1.6133, -0.631, -72.8003, 36.1691, -58.5689, -59.58332, -60.83,
         -63.266, 51.9479, 63.1335, 67.2221, -0.681, 5.909, 4.441, 51.0501
    ]
    dwarf_galaxy_pmra = [
        -0.093, -0.179, 1.01, -0.385, -2.426, -1.176, 0.469, -0.22, -0.096, -0.124, 0.532, 1.885, 3.095, -0.14, 2.844,
         None, 0.169, 0.423, -0.072, 0.044, 1.027, 0.125, 0.25, 0.381, 0.069, 0.384, -0.035, 0.847, 0.967, -0.394,
         3.781, -0.111, -0.13, -0.1921, 0.1186, -0.05, -0.01, 1.91, -0.03, 0.33, 0.507, 0.153, 1.15, 0.681, 2.377, 0.26,
         -2.679, 0.1, -2.102, 1.446, -0.409, None, 0.83, 0.575, 0.911, -0.048, 0.534, -0.14, -0.401, 1.731, -0.12,
         None, None, None, 0.255
    ]
    dwarf_galaxy_pmdec = [
        0.1, -0.466, -0.11, -1.068, -0.414, -0.89, 0.489, -0.28, -0.116, -0.254, 0.127, 0.133, 1.395, -0.19, 0.474,
         None, -0.4, -1.721, -0.112, -0.188, 0.887, 0.013, -0.1, -0.359, -0.248, -1.484, -0.339, -0.607, -0.771, 0.0,
         -1.496, -0.063, -0.143, -0.0686, -0.1183, -0.22, -1.29, 0.229, -0.58, -0.21, -1.199, 0.096, 1.14, -0.645,
         -1.379, -0.502, -1.394, -0.158, -3.375, -0.322, 0.037, None, -1.21, 0.112, -1.28, -1.638, -1.707, -1.18,
         -0.613, -1.906, 0.071, None, None, None, -1.11
    ]
    dwarf_galaxy_R = [
        124.16523075924093, 107.64652136298349, 85.50667128846841, 66.37430704019089, 41.68693834703355,
         46.55860935229591, 208.9296130854041, 101.8591388054117, 210.86281499332887, 159.9558028614668,
         105.58445924300804, 37.39383436431516, 27.797132677592884, 117.70637907805695, 29.922646366081892,
         251.1886431509582, 182.8100216142741, 42.26686142656025, 116.5735440441268, 81.54550000611157,
         21.57744409152669, 369.8281797802659, 76.5596606911257, 142.5607593602188, 126.47363474711523,
         55.20774392807573, 130.61708881318404, 79.43282347242821, 77.98301105232593, 151.35612484362073,
         27.542287033381633, 258.22601906345955, 233.34580622810043, 151.35612484362073, 169.04409316432634,
         111.1731727281592, 81.65823713585922, 49.59067275049043, 214.7830474130533, 89.94975815300347,
         83.17637711026708, 114.8153621496884, 45.708818961487516, 182.8100216142741, 31.622776601683793,
         91.62204901219992, 26.302679918953814, 83.94599865193982, 22.90867652767775, 36.475394692560734,
         85.90135215053961, 125.89254117941661, 62.805835881331795, 28.44461107447914, 56.23413251903491,
         22.90867652767775, 46.98941086052151, 54.954087385762485, 97.27472237769662, 34.673685045253166,
         70.14552984199713, 91.20108393559097, 72.44359600749905, 151.35612484362073, 38.01893963205613
    ]
    dwarf_galaxy_vlos = [
        288.8, -65.3, -13.1, 101.8, -130.4, 197.5, None, 5.1, 30.9, -128.9, 222.9, 477.2, 284.6, 44.8, None,
         None, 153.7, 98.1, 89.3, -290.7, -342.5, 75.6, -31.5, 55.2, -143.5, -110.0, 45.0, 112.8, None, 303.1, 80.4,
         282.9, 78.5, 131.6, 173.1, 170.75, None, 262.2, -222.9, -273.6, 32.4, None, None, -226.5, 64.3, 274.2,
         140.0, 111.4, 208.5, -40.2, 224.3, None, 145.6, -381.7, -124.7, -102.3, 15.9, -34.7, -55.3, -116.5, -247.0,
         None, None, None, -14.1
    ]
    dwarf_galaxy_pmra_sigma = [
        0.008, 0.113, 0.25, 0.017, 0.077, 0.019, 0.244, 0.05, 0.031, 0.115, 0.006, 0.019, 0.041, 0.05, 0.059, None,
         0.073, 0.027, 0.02, 0.006, 0.065, 0.1, 0.06, 0.001, 0.05, 0.033, 0.042, 0.035, 0.171, 0.14, 0.016, 0.004,
         0.009, 0.0514, 0.1943, 0.19, 0.4, 0.02, 0.21, 0.07, 0.048, 0.088, 0.06, 0.307, 0.024, 0.144, 0.001, 0.002,
         0.051, 0.059, 0.008, None, 0.02, 0.06, 0.026, 0.036, 0.053, 0.05, 0.036, 0.021, 0.005, None, None,
         None, 0.087
    ]
    dwarf_galaxy_pmdec_sigma = [
        0.009, 0.095, 0.19, 0.013, 0.061, 0.015, 0.255, 0.07, 0.02, 0.08, 0.006, 0.019, 0.045, 0.04, 0.063, None,
         0.079, 0.024, 0.013, 0.006, 0.072, 0.127, 0.05, 0.002, 0.072, 0.04, 0.036, 0.035, 0.23, 0.104, 0.015, 0.005,
         0.009, 0.0523, 0.1704, 0.17, 0.4, 0.047, 0.208, 0.08, 0.057, 0.114, 0.05, 0.209, 0.025, 0.226, 0.001, 0.002,
         0.046, 0.05, 0.009, None, 0.01, 0.067, 0.029, 0.039, 0.055, 0.06, 0.042, 0.025, 0.005, None, None,
         None, 0.091
    ]
    dwarf_galaxy_R_sigma = [
        5.04102995650355, 3.4147784236531322, 4.223619672058462, 2.4008234919260687, 1.1360848021951213,
         0.42685189625800035, 18.38354128907926, 6.798659439783606, 5.7465971107722, 4.359239700959279,
         5.261577185208139, 0.39398124702920256, 1.25107705783752, 3.7338986760820916, 2.5069046468030827,
         22.10187787418073, 9.029938739336473, 1.5288336461489394, 3.6979627203032237, 1.4883731125754025,
         0.49116259219380254, 8.418317140652732, 2.7692376782155463, 3.1167054730727273, 5.692251236437258,
         2.484757785793512, 5.878737388889974, 3.5750659695098506, 6.861659686992979, 7.476267018036367,
         0.5027033922931707, 9.3402872352205, 13.559818949617664, 4.1248746003487895, 4.606920842833091,
         3.5266513651757094, 6.841287084743854, 0.5450786059793984, 11.547346319431199, 1.2341569492074882,
         7.318619607348566, 5.16754253536989, 3.050867081328171, 13.76592844994775, 1.4232593976636636,
         12.189225539771712, 1.755590762103477, 1.5321871506395155, 2.0157152191373378, 2.434575722460643,
         3.8661977175578386, 11.077179029728214, 2.826728255780914, 1.4050274333906785, 4.947994119898432,
         1.031060288182232, 3.7380277570206246, 5.065638639761318, 5.652673365496511, 2.0149018280195676,
         2.847864213565316, 8.02470682532389, 6.979978600771446, 13.317698383332356, 6.396163030372339
    ]
    dwarf_galaxy_vlos_sigma = [
        0.4, 1.8, 1.0, 0.7, 1.1, 3.6, None, 13.4, 0.6, 1.2, 0.1, 1.2, 3.1, 0.8, None, None, 4.9, 0.9, 0.3, 0.75,
         1.2, 1.3, 1.2, 0.1, 1.2, 0.5, 1.1, 2.6, None, 1.4, 0.6, 0.5, 0.6, 1.2, 0.8, 1.65, None, 3.4, 2.6, 1.5,
         3.75, None, None, 2.7, 1.2, 7.45, 2.0, 0.1, 0.9, 0.9, 0.1, None, 0.6, 1.1, 1.0, 0.4, 1.7, 0.8, 1.4, 1.9,
         0.4, None, None, None, 1.0
    ]
    # _dsph_mw = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/main/data/dwarf_mw.csv')
    # dwarf_galaxy_names = [dsph_mw['key'][i] for i in range(0, 65)]
    # dwarf_galaxy_ra = [dsph_mw['ra'][i] for i in range(0, 65)]
    # dwarf_galaxy_dec = [dsph_mw['dec'][i] for i in range(0, 65)]
    # dwarf_galaxy_pmra = [dsph_mw['pmra'][i] for i in range(0, 65)]
    # dwarf_galaxy_pmdec = [dsph_mw['pmdec'][i] for i in range(0, 65)]
    # dwarf_galaxy_R = [dsph_mw['distance'][i] for i in range(0, 65)]
    # dwarf_galaxy_vlos = [dsph_mw['vlos_systemic'][i] for i in range(0, 65)]
    # dwarf_galaxy_pmra_sigma = [dsph_mw['pmra_em'][i] for i in range(0, 65)]
    # dwarf_galaxy_pmdec_sigma = [dsph_mw['pmdec_em'][i] for i in range(0, 65)]
    # dwarf_galaxy_R_sigma = [dsph_mw['distance_em'][i] for i in range(0, 65)]
    # dwarf_galaxy_vlos_sigma = [dsph_mw['vlos_systemic_em'][i] for i in range(0, 65)]
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
