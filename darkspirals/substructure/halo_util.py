import astropy.units as apu
from galpy.orbit import Orbit
from galpy.potential import HernquistPotential, NFWPotential
from galpy.potential import turn_physical_off


import numpy as np

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

def sample_dwarf_galaxy_potential_hernquist(log10_mass):
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

def sample_dwarf_galaxy_potential_nfw(log10_mass):
    """

    :param log10_mass:
    :return:
    """
    m = 10 ** log10_mass
    c = sample_concentration_nfw(m)
    pot = NFWPotential(mvir=m / 10 ** 12, conc=c)
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

def rho_twopower(r, M0, rs, alpha, beta):
    norm = M0 / (4*np.pi*rs**3)
    p1 = alpha
    p2 = beta - alpha
    x = r/rs
    denom = x ** p1 * (1+x)**p2
    return norm / denom

def mass_twopower(rmax, M0, rs, alpha, beta, x_min=1e-3, num=2000):
    r = np.linspace(x_min * rs, rmax, num)
    rho = rho_twopower(r, M0, rs, alpha, beta)
    return np.trapz(4*np.pi*r**2 * rho, r)
