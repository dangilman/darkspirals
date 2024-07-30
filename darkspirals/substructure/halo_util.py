import astropy.units as apu
from galpy.orbit import Orbit
from galpy.potential import HernquistPotential
from galpy.util.coords import rect_to_cyl, rect_to_cyl_vec
from darkspirals.orbit_util import integrate_single_orbit
from galpy.potential import turn_physical_off


import numpy as np

def sample_orbit_with_uncertainties(name, units, add_uncertainties=True):
    # uncertainities from Fritz et al.
    # https://ui.adsabs.harvard.edu/abs/2018A%26A...619A.103F/abstract
    # and Pace et al.
    # https://ui.adsabs.harvard.edu/abs/2022ApJ...940..136P/abstract
    _unit = {'ra': apu.deg,
             'dec': apu.deg,
             'dist': apu.kpc,
             'pmra': apu.mas / apu.yr,
             'pmdec': apu.mas / apu.yr,
             'vlos': apu.km / apu.s}
    if name == 'SegueI':
        orb = [Orbit.from_name('Segue1')]
    elif name == 'SegueII':
        orb = [Orbit.from_name('Segue2')]
    elif name == 'WillmanI':
        orb = [Orbit.from_name('Willman1')]
    else:
        orb = [Orbit.from_name(name)]
    init = {'ra': orb[0].ra(**units) * apu.deg,
            'dec': orb[0].dec(**units) * apu.deg,
            'dist': orb[0].dist(**units) * apu.kpc,
            'pmra': orb[0].pmra(**units) * apu.mas / apu.yr,
            'pmdec': orb[0].pmdec(**units) * apu.mas / apu.yr,
            'vlos': orb[0].vlos(**units) * apu.km / apu.s}
    if add_uncertainties:
        uncertainty = sample_orbit_uncertainty(name)
    else:
        uncertainty = None
    neworb = []
    for orb_param in init.keys():
        if add_uncertainties:
            neworb.append(np.random.normal(init[orb_param].value, uncertainty[orb_param]) * _unit[orb_param])
        else:
            neworb.append(init[orb_param].value)
    return neworb

def dwarf_galaxy_render(disc,
                        time_internal,
                        units,
                        log10_mass_loss=-1,
                        include_dwarf_list=None,
                        log10_m_peak_dict=None):
    """

    :param galactic_potential: the large-scale potential of the galaxy in which to integrate orbits
    :param time_internal: the orbit integration times in internal galpy units
    :param units: the internal units used by galpy
    :param log10_mass_loss: the logarithmic decrease in subhalo mass due to tidal stripping
    :param include_dwarf_list: a list of dwarf galaxies to include
    :param log10_m_peak_dict: a list of peak masses for specified satellites
    :return:
    """
    # these from Errani et al. 2018
    # dwarf_halo_masses = {'Fornax': 8.9, 'LeoI': 9.3, 'Sculptor': 9.3, 'LeoII': 8.5, 'Sextans': 8.5, 'Carina': 8.6,
    #            'UrsaMinor': 8.9, 'Draco': 9.5, 'CanesVenaticiI': 8.5, 'CraterII': 7., 'LeoT': 9.8, 'Hercules': 7.1,
    #            'BootesI': 6.4, 'LeoIV': 7.2, 'UrsaMajorI': 8.5, 'UrsaMajorII': 9.1,
    #           'CanesVenaticiII': 8.7, 'ComaBerenices': 8.6,
    #            'BootesII': 10.4, 'Willman1': 10.4, 'Segue2': 9., 'Segue1': 9.8, 'LeoV': 7.5,
    #           'LMC': np.log10(2 * 10 ** 11), 'Sagittarius dSph': 10.0}

    if include_dwarf_list is None:
        include_dwarf_list = ['Sculptor', 'LeoI', 'LeoII', 'Fornax', 'UrsaMinor', 'UrsaMajorII', 'Draco', 'Willman1', 'BootesI',
                              'SegueI', 'SegueII', 'Hercules'
                              ]
    if log10_m_peak_dict is None:
        # these calculated from the m_peak / M_V relation from Nadler et al. 2020
        log10_m_peak_dict = {'Willman1': 7.711677918523403, 'SegueI': 7.489376352511657, 'SegueII': 7.5905868215901755,
                       'Hercules': 8.308096754164671, 'LeoI': 9.383457988123926, 'LeoII': 9.014762707909323,
                       'Draco': 8.82860773799705, 'BootesI': 8.342436020459168, 'UrsaMinor': 8.886442291756202,
                       'UrsaMajorII': 8.022538644978852, 'Sculptor': 9.209954326846466, 'Fornax': 9.687089395359479}

    dwarf_names = []
    dwarf_potentials = []
    dwarf_orbits = []
    for name in include_dwarf_list:
        dwarf_names.append(name)
        log10_dwarf_halo_mass = log10_m_peak_dict[name] + log10_mass_loss

        # get the halo potentials
        m = 10 ** log10_dwarf_halo_mass
        c = sample_concentration_herquist(m, 17.5)
        pot = HernquistPotential(amp=0.5 * m * apu.solMass, a=c * apu.kpc)
        dwarf_potentials.append(pot)

        # get the halo orbits
        orb_init = sample_orbit_with_uncertainties(name, units)
        orb = integrate_single_orbit(orb_init, disc, **units)
        dwarf_orbits.append(orb)

    return dwarf_potentials, dwarf_orbits

def sample_orbit_uncertainty(name):
    if name == 'Willman1':
        uncertainty = {'ra': 0.001,
                       'dec': 0.001,
                       'dist': 0.001,
                       'pmra': 0.087,
                       'pmdec': 0.095,
                       'vlos': 2.5}
    elif name == 'Segue1':
        uncertainty = {'ra': 0.001,
                       'dec': 0.001,
                       'dist': 0.001,
                       'pmra': 0.051,
                       'pmdec': 0.046,
                       'vlos': 0.9}
    elif name == 'Segue2':
        uncertainty = {'ra': 0.001,
                       'dec': 0.001,
                       'dist': 0.001,
                       'pmra': 0.059,
                       'pmdec': 0.05,
                       'vlos': 2.5}
    elif name == 'Hercules':
        uncertainty = {'ra': 0.001,
                       'dec': 0.001,
                       'dist': 0.001,
                       'pmra': 0.042,
                       'pmdec': 0.036,
                       'vlos': 1.1}
    elif name == 'LeoI':
        uncertainty = {'ra': 0.001,
                       'dec': 0.001,
                       'dist': 0.001,
                       'pmra': 0.014,
                       'pmdec': 0.01,
                       'vlos': 0.5}
    elif name == 'LeoII':
        uncertainty = {'ra': 0.001,
                       'dec': 0.001,
                       'dist': 0.001,
                       'pmra': 0.028,
                       'pmdec': 0.026,
                       'vlos': 0.1}
    elif name == 'Draco':
        uncertainty = {'ra': 0.001,
                       'dec': 0.001,
                       'dist': 0.001,
                       'pmra': 0.006,
                       'pmdec': 0.006,
                       'vlos': 0.1}
    elif name == 'UrsaMinor':
        uncertainty = {'ra': 0.001,
                       'dec': 0.001,
                       'dist': 0.001,
                       'pmra': 0.021,
                       'pmdec': 0.025,
                       'vlos': 1.4}
    elif name == 'Sculptor':
        uncertainty = {'ra': 0.001,
                       'dec': 0.001,
                       'dist': 0.001,
                       'pmra': 0.002,
                       'pmdec': 0.002,
                       'vlos': 0.1}
    elif name == 'LMC':
        uncertainty = {'ra': 0.001,
                       'dec': 0.001,
                       'dist': 0.001,
                       'pmra': 0.02,
                       'pmdec': 0.047,
                       'vlos': 24.0}
    else:
        print('adding default orbit uncertainties for '+str(name))
        uncertainty = {'ra': 0.001,
                       'dec': 0.001,
                       'dist': 0.001,
                       'pmra': 0.002,
                       'pmdec': 0.002,
                       'vlos': 0.1}

    return uncertainty

def sample_concentration_herquist(m, norm=17.5):
    """

    :param m:
    :param norm:
    :return:
    """
    a = 1.05 * (m / 10 ** 8) ** 0.5
    rescale = 17.5/norm
    return rescale * a

def sample_concentration_nfw(m, norm=17.5):
    # close to the CDM mass concentration relation
    return norm * (m / 10 ** 8) ** -0.06

def sample_dwarf_galaxy_potential(log10_mass):
    """
    Computes the potential of a dwarf galaxy
    :param log10_mass: halo mass
    :return: a potential instance from galpy
    """
    m = 10 ** log10_mass
    a_scale = (m/10**10)**0.33
    pot = HernquistPotential(amp=m * apu.M_sun, a=a_scale * 3. * apu.kpc)
    turn_physical_off(pot)
    return pot

def sample_mass_function(norm, alpha, mH, mL, num_halos=None):
    """

    :param norm:
    :param alpha:
    :param mH:
    :param mL:
    :param num_halos:
    :return:
    """
    m0 = 10 ** 8
    one_plus_alpha = 1 + alpha
    m1 = mH / m0
    m2 = mL / m0
    if num_halos is None:
        n_mean = int(norm / one_plus_alpha * (m1 ** one_plus_alpha - m2 ** one_plus_alpha))
        ndraw = np.random.poisson(n_mean)
    else:
        ndraw = int(num_halos)
    x = np.random.uniform(0, 1, int(ndraw))
    X = (x * (mH ** (1 + alpha) - mL ** (1 + alpha)) + mL ** (
        1 + alpha)) ** (
            (1 + alpha) ** -1)
    return np.array(X)
