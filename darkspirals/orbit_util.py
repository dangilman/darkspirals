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

    def __init__(self, vxvv=None,
        ro=None,
        vo=None,
        zo=None,
        solarmotion=None,
        radec=False,
        uvw=False,
        lb=False,
        pot=None):
        """
        This class implements an Orbit class in galpy, but the potential of the satellite associated with the orbit
        is also made into a class attribute in order to calculate various aspects of the perturbation associated with
        the satellite

        See the dcoumentation in galpy for the initialization of orbits
        :param pot: an instance of potential in galpy for the satellite
        """
        self.potential = pot
        turn_physical_off(self.potential)
        super(OrbitExtension, self).__init__(vxvv, ro, vo, zo, solarmotion, radec, uvw, lb)

    def trajectory(self, disc, relative=False):
        """
        Calculate the 6D phase space of the orbit as a function of time
        :param disc: an instance of Disc
        :param relative: bool; calculate motion relative to the position and velocity of the sun
        :return: position and velocity in internal units
        """
        t = disc.time_internal_eval
        x = self.x(t) * disc.units['ro']
        y = self.y(t) * disc.units['ro']
        z = self.z(t) * disc.units['ro']
        vx = self.vx(t) * disc.units['vo']
        vy = self.vy(t) * disc.units['vo']
        vz = self.vz(t) * disc.units['vo']
        if relative:
            xref, yref = disc.solar_circle
            vxref, vyref = disc.solar_circle_velocity(t)
            zref, vzref = 0.0, 0.0
        else:
            xref, yref = 0.0, 0.0
            vxref, vyref = 0.0, 0.0
            zref, vzref = 0.0, 0.0
        xref *= disc.units['ro']
        yref *= disc.units['ro']
        zref *= disc.units['ro']
        vxref *= disc.units['vo']
        vyref *= disc.units['vo']
        vzref *= disc.units['vo']
        x = (x - xref, y - yref, z - zref)
        v = (vx - vxref, vy - vyref, vz - vzref)
        return x, v

    def disc_crossing_times(self, disc):
        """
        Compute the time or times when the subhalo cross the galactic disk (z=0)
        :param disc: an instance of Disc
        :return: a list of times in the past in Gyr when a subhalo crosses z=0
        """
        t = disc.time_internal_eval
        z = np.squeeze(self.z(t))
        index_cross = np.where(np.sign(z[:-1]) != np.sign(z[1:]))[0]
        if len(index_cross) == 0:
            return []
        else:
            crossing_times = np.array(t[index_cross])
            return disc.internal_time_to_time(crossing_times)

    def orbit_parameters(self, disc, t_max=None):
        """
        Calculate the distance from the solar neighborhood at the time when a satellite exerts the largest force on the
        solar neighborhood. Return both
        :param disc: an instance of Disc
        :param t_max: ignore properties of the orbit before t_max; t_max should be negative, i.e. a time in the past
        :return: distance in 3D from the solar position when the perturbed exerted the strongest vertical force, and
        the (absolute value of the) time when this maximum force occured in kpc and Gyr, resp.
        """
        r_min, t_min = self.closest_approach, self.impact_time
        if r_min is None or t_min is None:
            _x_solar, _y_solar = disc.solar_circle
            self.turn_physical_off()
            if t_max is None:
                t = disc.time_internal_eval
                x_solar = _x_solar
                y_solar = _y_solar
            else:
                assert t_max < 0
                t_max_interal = disc.time_to_internal_time(t_max)
                indexes = np.where(disc.time_internal_eval >= t_max_interal)[0]
                t = disc.time_internal_eval[indexes]
                x_solar = _x_solar[indexes]
                y_solar = _y_solar[indexes]
            x_orb, y_orb, z_orb = np.squeeze(self.x(t)), np.squeeze(self.y(t)), np.squeeze(
                self.z(t))
            dx, dy, dz = x_orb - x_solar, y_orb - y_solar, z_orb - 0.0
            dR = np.sqrt(dx ** 2. + dy ** 2.)
            dr3d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            f = evaluatezforces(self.potential, R=dR, z=dz)
            idx_max = np.argmax(np.absolute(f))
            r_min = dr3d[idx_max] * disc.units['ro']
            t_min = abs(disc.internal_time_to_time(t[idx_max]))
            self.set_closest_approach(r_min, t_min)
        return r_min, t_min

    def vertical_distance_from_solar_position(self, disc, t_max=None):
        """
        Calculate the vertical distance of the perturber from the solar position as a function of time
        :param disc: an instance of Disc
        :param t_max: ignore positions of the halo at times > |t_max|; t_max should be a negative number that specifies
        a time in the past in Gyr
        :return: vertical distance as a function of time in kpc
        """
        _x_solar, _y_solar = disc.solar_circle
        self.turn_physical_off()
        if t_max is None:
            t = disc.time_internal_eval
        else:
            assert t_max < 0
            t_max_interal = disc.time_to_internal_time(t_max)
            indexes = np.where(disc.time_internal_eval >= t_max_interal)[0]
            t = disc.time_internal_eval[indexes]
        z_orb = np.squeeze(self.z(t))
        return z_orb * disc.units['ro']

    def minimum_distance_galactic_center(self, disc):
        """
        Compute the minimum distance from the galactic center of the perturber orbit in kpc
        """
        t_internal = disc.time_internal_eval
        x, z, y = self.x(t_internal), self.y(t_internal), self.z(t_internal)
        r_min = np.min(np.sqrt(x**2 + y**2 + z**2))
        return r_min * disc.units['ro']

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

    def impact_velocity(self, disc, relative=False):
        """
        The speed of a satellite relative to the sun at the impact time
        :param disc: instance of Disc class
        :return: speed in km/sec
        """
        ro, vo = disc.units['ro'], disc.units['vo']
        t_min = self.impact_time
        if t_min is None:
            raise Exception('must integrate orbits before computing impact velocity')
        t_min_internal = t_min / time_in_Gyr(vo, ro)
        if relative:
            vx_ref, vy_ref = disc.solar_circle_velocity(-t_min_internal)
        else:
            vx_ref = 0.0
            vy_ref = 0.0
        vx_orb = self.vx(-t_min_internal)
        vy_orb = self.vy(-t_min_internal)
        vz_orb = self.vz(-t_min_internal)
        v = np.sqrt((vx_orb - vx_ref) ** 2 + (vy_orb - vy_ref) ** 2 + vz_orb**2)
        return v * vo

    def force_exerted(self, disc, physical_units=False, t_max=None):
        """

        :param disc: an instance of Disc
        :param physical_units: bool; give output in physical units rather than internal galpy units
        :param t_max: force is set to zero for t < t_max (t_max is < 0)
        :return: the force exerted by the satellite as a function of time at the solar position
        """
        f = satellite_vertical_force(disc,
                                     self,
                                     self.potential)
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

    def deltaJ(self, disc, physical_units=False, t_max=None):
        """
        Compute the perturbation to the action at the solar position
        :param disc: an instance of Disc
        :param physical_units: if True, returns action in units kpc * km / sec
        :param t_max: force is set to zero for t < t_max (t_max is negative number in Gyr)
        :return: the change to the vertical action at the solar position caused by the satellite
        """
        f, _, _ = disc.compute_satellite_forces(satellite_orbit_list=[self],
                                          satellite_potentials_list=[self.potential],
                                          t_max=t_max)
        dj = disc.compute_deltaJ_from_forces(f)[0]
        if physical_units:
            dj *= disc.units['ro'] * disc.units['vo']
        return dj

def integrate_single_orbit(orbit_init, disc, pot,
                           ro=8., vo=220., radec=True, lmc_orbit=None, lmc_potential=None):
    """
    Integrate the orbit of a satellite in the Galactic potential
    :param orbit_init: initialization of the orbit as described in galpy documentation
    :param disc: an instance of Disc (see darkspirals.disc.py)
    :param pot: potential of the satellite
    :param ro: length scale used in galpy's internal units
    :param vo: velocity scale used in galpy's internal units
    :param radec: bool; specifies the orbit_init used by galpy, as described in galpy's orbit class documentation
    :param lmc_orbit: bool; include the effects of the LMC
    :param lmc_potential: a galpy potential class for the LMC
    :return: an instance of OrbitExtension that stores the satellite orbit
    """
    satellite_orbit = OrbitExtension(vxvv=orbit_init, radec=radec, ro=ro, vo=vo, pot=pot)
    if lmc_orbit is None:
        satellite_orbit.integrate(disc.time_internal_eval, disc.galactic_potential)  # Integrate orbit
    else:
        lmc_pot = MovingObjectPotential(lmc_orbit,
                                        lmc_potential)
        pot = disc.galactic_potential + lmc_pot
        satellite_orbit.integrate(disc.time_internal_eval, pot)
    satellite_orbit.turn_physical_off()
    return satellite_orbit

def satellite_vertical_force(disc, satellite_orbit, satellite_potential,
                             z_coord=0.0, record_force_array=False, t_max=None):
    """
    Calculates the force from a passing satellite on the solar neighborhood as a function of time

    :param disc: an instance of Disc (see darkspirals.disc)
    :param satellite_orbit: an instance of OrbitExtension
    :param satellite_potential: an instance of a galpy potential class that corresponds to the perturber
    :param z_coord: z-coordinates at which to compute the vertical force; z_coord = 0 corresponds to the Galactic
    midplane. The use of this method in the Disc class passes an array of z-coordinates in phase-space
    :param record_force_array: bool; if True, then the Orbit instance will receive a 'force_exerted' attribute for use
    in the calculation of perturbations to the full distribution function. This is intended to avoid multiple expensive
     calculations of the same quantity
    :param t_max: the lookback time over which to return the force exerted, should specify a time in the past (<0)
    :return: The vertical force exerted by the satellite at the specified z-coordinate
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
