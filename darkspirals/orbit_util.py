from galpy.orbit import Orbit
import numpy as np
import astropy.units as apu
from galpy.potential import evaluatezforces
from galpy.util.conversion import time_in_Gyr
from galpy.util.conversion import force_in_2piGmsolpc2
from galpy.potential import turn_physical_off, MovingObjectPotential


class OrbitExtension(Orbit):
    """
    Endow galpy's orbit class with some new properties
    """
    force_array_computed = False

    def minimum_distance_galactic_center(self, time_gyr, ro=8.0, vo=220.0):
        """
        Compute the minimum distance from the galactic center
        """
        t_internal = time_gyr / time_in_Gyr(vo, ro)
        x, z, y = self.x(t_internal), self.y(t_internal), self.z(t_internal)
        r_min = np.min(np.sqrt(x**2 + y**2 + z**2))
        return r_min * ro

    def set_closest_approach(self, r_min, t_min):
        """
        Set the minimum distance from the solar neighborhood of an orbit
        :param r_min: the distance from the solar neighborhood in kpc
        :param t_min: the time of minimum distance from the solar neighborhood in Gyr
        """
        self._closest_approach = r_min
        self._impact_time = t_min

    def set_force_array(self, force):
        """

        :param force:
        :return:
        """
        self.force_array_computed = True
        self._force_array = force

    @property
    def closest_approach(self):
        if hasattr(self, '_closest_approach'):
            return self._closest_approach
        else:
            return None

    @property
    def impact_time(self):
        if hasattr(self, '_impact_time'):
            return self._impact_time
        else:
            return None

    @property
    def force_exerted_array(self):
        if hasattr(self, '_force_array'):
            return self._force_array
        else:
            return None

    def impact_velocity(self, ro=8.0, vo=220.0):
        """
        Compute the velocity at the minimum separation from the solar position
        :param ro:
        :param vo:
        :return:
        """
        t_min = self.impact_time
        if t_min is None:
            raise Exception('must integrate orbits before computing impact velocity')
        t_min_internal = t_min / time_in_Gyr(vo, ro)
        v = np.sqrt(self.vx(-t_min_internal) ** 2 + self.vy(-t_min_internal) ** 2  + self.vz(-t_min_internal)**2)
        return v * vo

    def force_exerted(self, disc, satellite_potential, physical_units=False, t_max=None):
        """

        :param disc: an instance of Disc
        :param satellite_potential: a potential instance in galpy for the satellite
        :param physical_units: bool; give output in physical units rather than internal galpy units
        :param t_max: force is set to zero for t < t_max (t_max is < 0)
        :return: the force exerted by the satellite as a function of time at the solar position
        """
        f = satellite_vertical_force(disc,
                                     self,
                                     satellite_potential)
        if physical_units:
            f *= force_in_2piGmsolpc2(**disc.units)
        if t_max is None:
            return f
        else:
            if t_max > 0.0:
                t_max *= -1
                print('you specified a time t_max that is positive, suggesting you want to know the orbit in the future. '
                      'Orbit info. is only stored in the past. Assuming you meant t_max = -t_max... ')
            t_max_internal = disc.time_to_internal_time(t_max)
            indexes = np.where(disc.time_internal_eval<t_max_internal)[0]
            f[indexes] = 0.0
            return f

    def deltaJ(self, disc, satellite_potential, physical_units=False, t_max=None):
        """
        Compute the perturbation to the action at the solar position
        :param disc:
        :param satellite_potential:
        :param physical_units:
        :param t_max: force is set to zero for t < t_max (t_max is < 0)
        :return:
        """
        f, _, _ = disc.compute_satellite_forces(satellite_orbit_list=[self],
                                          satellite_potentials_list=[satellite_potential],
                                          t_max=t_max)
        dj = disc.compute_deltaJ_from_forces(f)[0]
        if physical_units:
            dj *= disc.units['ro'] * disc.units['vo']
        return dj

def integrate_single_orbit(orbit_init, disc, ro=8., vo=220., radec=True, lmc_orbit=None, lmc_potential=None,
                           t_max=None):

    """
    This function integrates an orbit in a potential with initial conditions orbit_init over a time
    time_Gyr

    """
    satellite_orbit = OrbitExtension(vxvv=orbit_init, radec=radec, ro=ro, vo=vo)
    if lmc_orbit is None:
        satellite_orbit.integrate(disc.time_internal_eval, disc.galactic_potential)  # Integrate orbit
    else:
        lmc_pot = MovingObjectPotential(lmc_orbit,
                                        lmc_potential)
        pot = disc.galactic_potential + lmc_pot
        satellite_orbit.integrate(disc.time_internal_eval, pot)
    satellite_orbit.turn_physical_off()
    dr_min, t_min = orbit_closest_approach(disc, satellite_orbit, t_max)
    satellite_orbit.set_closest_approach(dr_min, t_min)
    return satellite_orbit

def orbit_closest_approach(disc, orb, t_max=None):
    """
    Calculate the minimum distance between a halo and the solar neighborhood, and the time of minimum distance
    :param disc:
    :param orb:
    :param units:
    :return:
    """
    _x_solar, _y_solar = disc.solar_circle
    orb.turn_physical_off()
    if t_max is None:
        t = disc.time_internal_eval
        x_solar = _x_solar
        y_solar = _y_solar
    else:
        assert t_max < 0
        t_max_interal = disc.time_to_internal_time(t_max)
        indexes = np.where(disc.time_internal_eval > t_max_interal)[0]
        t = disc.time_internal_eval[indexes]
        x_solar = _x_solar[indexes]
        y_solar = _y_solar[indexes]
    x_orb, y_orb, z_orb = np.squeeze(orb.x(t)), np.squeeze(orb.y(t)), np.squeeze(
        orb.z(t))
    dx, dy, dz = x_orb - x_solar, y_orb - y_solar, z_orb - 0.0
    dr = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) * disc.units['ro']
    idx_min = np.argsort(dr)[0]
    return dr[idx_min], abs(disc.internal_time_to_time(t[idx_min]))

def sample_sag_orbit(scale_uncertainties=1.0):
    """

    :param scale_uncertainties:
    :return:
    """
    # orbit_init = [280. * apu.deg, -30. * apu.deg, 27. * apu.kpc,
    #               -2.6 * apu.mas / apu.yr, -1.3 * apu.mas / apu.yr,
    #               140. * apu.km / apu.s]  # Initial conditions of the satellite
    alpha_0, delta_0 = 283.8313, -30.5453

    standard_dev = np.sqrt(0.001)
    d_alpha = np.random.normal(0, standard_dev) * scale_uncertainties
    d_delta = np.random.normal(0, standard_dev) * scale_uncertainties
    alpha, delta = alpha_0 + d_alpha, delta_0 + d_delta

    z_0 = 26
    delta_z = np.random.normal(0, 2.0) * scale_uncertainties
    z = z_0 + delta_z

    delta_mu_alpha = np.random.normal(0, standard_dev) * scale_uncertainties
    delta_mu_delta = np.random.normal(0, standard_dev) * scale_uncertainties
    mu_alpha = -2.692 + delta_mu_alpha
    mu_delta = -1.359 + delta_mu_delta

    vr_0 = 140
    delta_vr = np.random.normal(0, 2.) * scale_uncertainties
    vr = vr_0 + delta_vr

    uncertainties = [d_alpha, d_delta, delta_z, delta_mu_alpha, delta_mu_delta, delta_vr]

    #[283. * apu.deg, -30. * apu.deg, 26. * apu.kpc,
    # -2.6 * apu.mas / apu.yr, -1.3 * apu.mas / apu.yr, 140. * apu.km / apu.s]
    orbit_init_sag = [alpha * apu.deg, delta * apu.deg, z * apu.kpc,
                      mu_alpha * apu.mas / apu.yr, mu_delta * apu.mas / apu.yr,
                      vr * apu.km / apu.s]  # Initial conditions of the satellite

    return orbit_init_sag, uncertainties

def sample_dwarf_orbit_init(name, units, add_uncertainties=True):
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
        uncertainty = sample_dwarf_orbit_uncertainty(name)
    else:
        uncertainty = None
    neworb_init = []
    for orb_param in init.keys():
        if add_uncertainties:
            neworb_init.append(np.random.normal(init[orb_param].value, uncertainty[orb_param]) * _unit[orb_param])
        else:
            neworb_init.append(init[orb_param].value)
    return neworb_init

def sample_dwarf_orbit_uncertainty(name):
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

def satellite_vertical_force(disc, satellite_orbit, satellite_potential,
                             z_coord=0.0, record_force_array=False, t_max=None):
    """
    Calculates the force from a passing satellite on the solar neighborhood as a function of time
    :param disc:
    :param satellite_orbit:
    :param satellite_potential:
    :param record_force_array: bool; if True, then the Orbit instance will aquire a 'force_exerted' attribute for use
    in the calculation of perturbations to the full distribution function. This is intended to avoid multiple expensive
     calculations of the same quantity
    :param t_max: the lookback time over which to return the force exerted, should specify a time in the past (<0)
    :return:
    """
    x_solar, y_solar = disc.solar_circle
    t_eval_internal = disc.time_internal_eval
    satellite_orbit.turn_physical_off()
    dx = x_solar - satellite_orbit.x(t_eval_internal)
    dy = y_solar - satellite_orbit.y(t_eval_internal)
    dz = z_coord - satellite_orbit.z(t_eval_internal)
    dR = np.sqrt(dx ** 2. + dy ** 2.)
    turn_physical_off(satellite_potential)
    force = evaluatezforces(satellite_potential, R=dR, z=dz)
    if t_max is not None:
        assert t_max < 0
        t_max_interal = disc.time_to_internal_time(t_max)
        indexes_before_tmax = np.where(t_eval_internal < t_max_interal)
        if isinstance(z_coord, float) or isinstance(z_coord, int):
            force[indexes_before_tmax] = 0
        else:
            # we have an array of forces
            force[:, :, indexes_before_tmax] = 0
    if record_force_array:
        satellite_orbit.set_force_array(force)
    return force
