import numpy as np
import galpy
from galpy.potential import toVerticalPotential
from galpy.orbit import Orbit
from galpy.util.conversion import time_in_Gyr
from galpy.util.conversion import force_in_2piGmsolpc2
from scipy.interpolate import RegularGridInterpolator
from galpy.actionAngle.actionAngleVertical import actionAngleVertical
from multiprocessing.pool import Pool
from galpy.potential import verticalfreq
from darkspirals.orbit_util import satellite_vertical_force
from scipy.integrate import simps
from galpy.potential import turn_physical_off
import os
from tqdm import tqdm
import multiprocessing as mp


class Disc(object):

    def __init__(self, local_potential, galactic_potential, z_min_max_kpc, vz_min_max_kpc, phase_space_pixels,
                 time_Gyr_eval, units_ro=8.0, units_vo=220.0, compute_action_angle=True,
                 parallelize_action_angle_computation=True, compute_upfront=True, r_over_r0=1.178):
        """

        :param local_potential:
        :param galactic_potential:
        :param z_min_max_kpc:
        :param vz_min_max_kpc:
        :param phase_space_pixels:
        :param time_Gyr_eval:
        :param units_ro:
        :param units_vo:
        :param compute_action_angle:
        :param parallelize_action_angle_computation:
        :param compute_upfront:
        """
        turn_physical_off(local_potential)
        turn_physical_off(galactic_potential)
        self.local_potential = local_potential
        self.local_vertical_potential = [toVerticalPotential(local_potential,
                                                            1.0, # the spatial coordinate where we evaluate the 3D potential
                                                            # in units of ro
                                                            phi=0.)]
        self.galactic_potential = galactic_potential
        self.time_Gyr_eval = time_Gyr_eval
        self.units = {'ro': units_ro, 'vo': units_vo}
        self._r_over_r0 = r_over_r0
        try:
            self.time_internal_eval = self.time_to_internal_time(time_Gyr_eval.value)
        except:
            self.time_internal_eval = self.time_to_internal_time(time_Gyr_eval)
        self.z_units_internal = np.linspace(-z_min_max_kpc, z_min_max_kpc, phase_space_pixels) / self.units['ro']
        self.vz_units_internal = np.linspace(-vz_min_max_kpc, vz_min_max_kpc, phase_space_pixels) / self.units['vo']
        self._parallelize = parallelize_action_angle_computation
        self._phase_space_dim = int(len(self.z_units_internal))
        self._time_step = self.time_internal_eval[1] - self.time_internal_eval[0]
        if compute_upfront:
            if compute_action_angle:
                _ = self.action_angle_frequency
                _ = self.action_angle_interp
            _ = self.orbits_in_phase_space

    def compute_deltaJ_from_forces(self, forces, second_order=False, verbose=False, parallel=False, n_cpu=6):
        """
        Compute the change to the vertical action from an external force
        :param forces: a list of forces acting on the phase space, each force must have shape (n, n, len(time_eval_internal)
        :param second_order: bool; include the (dz / dJ) * f_z term in the denominator
        :param verbose: make print statements
        :return: a list of action perturbations of the same length as forces
        """
        v_z = self._phase_space_orbits.vx(self.time_internal_eval)
        time_step = self.time_internal_eval[1] - self.time_internal_eval[0]
        delta_J_list = []
        freq_0 = self.frequency.reshape(self._phase_space_dim, self._phase_space_dim, 1)
        for counter, f in enumerate(forces):
            if second_order:
                dj_dz = self.dJdz.reshape(self._phase_space_dim, self._phase_space_dim, 1)
                omega = freq_0 + f / dj_dz
            else:
                omega = freq_0
            dJ = np.squeeze(simps(v_z * f / omega, dx=time_step))
            delta_J_list.append(dJ)
            if verbose and counter%25==0:
                percent_done = int(100*counter/len(forces))
                print('completed '+str(percent_done)+'% of action calculations... ')
        return delta_J_list

    def compute_satellite_forces(self, satellite_orbit_list=None,
                                 satellite_potentials_list=None,
                                 verbose=False,
                                 parallel=False, n_cpu=6):

        """
        Computes the vertical forces from a list of passing satellites
        :param satellite_orbit_list: list of Orbit instances for each satellite
        :param satellite_potentials_list: list of potentials corresponding to each satellite
        :param verbose: make print statements
        :return: a list of vertical forces, impact times (time of maximum force), and impact parameters (minimum distance)
         for satellites

         All quantities returned in internal units
        """
        assert len(satellite_orbit_list) == len(satellite_potentials_list)
        force_list = []
        impact_times = []
        impact_parameters = []
        z_coord = self.orbits_in_phase_space.x(self.time_internal_eval)
        if parallel:
            arg_list = []
            for orbit, potential in zip(satellite_orbit_list, satellite_potentials_list):
                arg = (self, orbit, potential, z_coord, True)
                arg_list.append(arg)
            with mp.Pool(processes=n_cpu) as pool:
                force_list = tqdm(
                    pool.starmap(satellite_vertical_force, arg_list)
                )
            impact_times = [orbit.impact_time for orbit in satellite_orbit_list]
            impact_parameters = [orbit.closest_approach for orbit in satellite_orbit_list]
        else:
            for i in range(0, len(satellite_orbit_list)):
                satellite_force = satellite_orbit_list[i].force_exerted_array
                if satellite_force is None:
                    satellite_force = satellite_vertical_force(self,
                                                           satellite_orbit_list[i],
                                                           satellite_potentials_list[i],
                                                           z_coord=z_coord,
                                                            record_force_array=True)
                force_list.append(satellite_force)
                impact_times.append(satellite_orbit_list[i].impact_time)
                impact_parameters.append(satellite_orbit_list[i].closest_approach)
                if verbose and i%25==0:
                    percent_done = int(100*i/len(satellite_orbit_list))
                    print('completed '+str(percent_done)+'% of force calculations... ')
        return force_list, np.array(impact_times), np.array(impact_parameters)

    @property
    def orbits_in_phase_space(self):
        """
        Compute the orbits of test particles in the phase-space coordinates that initial the class
        :param time_in_Gyr: time in Gyr with astropy units
        :return: instances of orbits of test particles
        """
        if not hasattr(self, '_phase_space_orbits'):
            vxvv = np.array(np.meshgrid(self.z_units_internal, self.vz_units_internal)).T
            orbits = Orbit(vxvv, ro=self.units['ro'], vo=self.units['vo'])
            orbits.turn_physical_off()
            orbits.integrate(self.time_internal_eval, self.local_vertical_potential)
            self._phase_space_orbits = orbits
        return self._phase_space_orbits

    @property
    def action_angle_interp(self):
        if not hasattr(self, '_action_angle_interp'):
            nz, nv = len(self.z_units_internal), len(self.vz_units_internal)
            action, angle, frequency = self.action_angle_frequency
            self._action = action.reshape(nz, nv)
            self._angle = angle.reshape(nz, nv)
            self._frequency = frequency.reshape(nz, nv)
            z_phys = self.z_units_internal * self.units['ro']
            vz_phys = self.vz_units_internal * self.units['vo']
            points = (z_phys, vz_phys)
            shape = (self._action.shape[0], self._action.shape[1], 3)
            values = np.empty(shape)
            values[:, :, 0] = self._action
            values[:, :, 1] = self._angle
            values[:, :, 2] = self._frequency
            self._action_angle_interp = RegularGridInterpolator(points, values)
        return self._action_angle_interp

    @property
    def action(self):
        """
        Get the vertical action
        :return: vertical action in internal units
        """
        action, _, _ = self.action_angle_frequency
        return action

    @property
    def angle(self):
        """
        Get the vertical angle
        :return: vertical angle in internal units
        """
        _, angle, _ = self.action_angle_frequency
        return angle

    @property
    def frequency(self):
        """
        Get the vertical frequency
        :return: vertical frequency in internal units
        """
        _, _, frequency = self.action_angle_frequency
        return frequency

    @property
    def action_angle_frequency(self):

        """
        Computes action/angle/frequency coordinates at each point in phase space
        :param parallelize: bool, do the calculation across mulitpole CPUs (good as long as it's not on a cluster)
        :return: actions, angles, and frequencies in internal units
        """

        if not hasattr(self, '_action'):
            nz, nv = len(self.z_units_internal), len(self.vz_units_internal)
            z_step = abs(self.z_units_internal[1] - self.z_units_internal[0])
            v_step = abs(self.vz_units_internal[1] - self.vz_units_internal[0])

            action = np.empty([nz, nv])
            frequency = np.empty([nz, nv])
            angle = np.empty([nz, nz])
            tol = 1e-3
            aAV = actionAngleVertical(pot=self.local_vertical_potential)

            if self._parallelize:
                action = action.ravel()
                frequency = frequency.ravel()
                angle = angle.ravel()
                arg_list = []
                for i, zval in enumerate(self.z_units_internal):
                    for j, vval in enumerate(self.vz_units_internal):
                        arg_list.append((zval, vval))

                pool = Pool(int(os.cpu_count()))
                out = pool.starmap(aAV.actionsFreqsAngles, arg_list)
                pool.close()
                for i in range(0, len(arg_list)):
                    J, freq, theta = out[i]
                    action[i] = J
                    angle[i] = theta
                    frequency[i] = freq
                action = action.reshape(nz, nv)
                angle = angle.reshape(nz, nv)
                frequency = frequency.reshape(nz, nv)
            else:
                for i, zval in enumerate(self.z_units_internal):
                    for j, vval in enumerate(self.vz_units_internal):
                        J, freq, theta = aAV.actionsFreqsAngles(zval, vval)
                        action[i, j] = J
                        angle[i, j] = theta
                        frequency[i, j] = freq
                        try:
                            assert np.isfinite(J)
                            assert np.isfinite(freq)
                            assert np.isfinite(theta)
                        except:
                            print('undefined result with ', (zval, vval))
                            exit(1)
            self._action = action
            self._angle = angle
            self._frequency = frequency
        return self._action, self._angle, self._frequency

    @property
    def solar_circle(self):
        """
        computes the location of the sun as a function of time
        :param time_Gyr: the time in Gyr
        :return: the x, y coordinate the solar position in internal units
        """
        freq = self.circular_velocity / self._r_over_r0
        x_solar = self._r_over_r0 * np.cos(freq * self.time_internal_eval)
        y_solar = self._r_over_r0 * np.sin(freq * self.time_internal_eval)
        return x_solar, y_solar

    @property
    def rho_midplane(self):

        """
        :return: the density evaluated at R_over_R0 in internal units.
        """
        if not hasattr(self, '_rho_midplane'):
            self._rho_midplane = galpy.potential.evaluateDensities(self.galactic_potential, 1.0, 0., phi=0.)
        return self._rho_midplane

    @property
    def circular_velocity(self):

        """
        :return: the circular velocity of the galactic potential evaluated at R_over_R0=1.0 in internal units
        """
        if not hasattr(self, '_vcirc'):
            self._vcirc = galpy.potential.vcirc(self.galactic_potential, 1.0)
        return self._vcirc

    @property
    def dJdz(self):
        if not hasattr(self, '_djdz'):
            z_step = (self.z_units_internal[1] - self.z_units_internal[0])
            self._djdz = np.gradient(self.action, z_step, axis=0)
        return self._djdz

    def Omega(self, t=None, force=None, second_order=False):
        """
        """
        freq_0 = self.frequency.reshape(self._phase_space_dim, self._phase_space_dim, 1)
        if second_order:
            dj_dz = self.dJdz.reshape(self._phase_space_dim, self._phase_space_dim, 1)
            idx = np.argmin(np.absolute(t - self.time_internal_eval))
            omega = freq_0 + force[:, :, 0:idx] / dj_dz
            return omega
        else:
            return freq_0

    @property
    def vertical_frequency(self):

        """
        :return: the second derivative of the local potential at z = 0
        """
        if not hasattr(self, '_d2psi_dz2'):
            self._d2psi_dz2 = verticalfreq(self.local_potential, 1.0)
        return self._d2psi_dz2

    def internal_time_to_time(self, time_internal):
        """

        :param time_internal:
        :return:
        """
        time_Gyr = time_internal * time_in_Gyr(self.units['vo'], self.units['ro'])
        return time_Gyr

    def time_to_internal_time(self, time_Gyr):

        """

        :param time: time in Gyr
        :return: time expressed in internal time units
        """
        try:
            t = time_Gyr.value / time_in_Gyr(self.units['vo'], self.units['ro'])
        except:
            t = time_Gyr / time_in_Gyr(self.units['vo'], self.units['ro'])
        return t
