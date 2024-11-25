import numpy as np
import galpy
from galpy.potential import toVerticalPotential
from galpy.orbit import Orbit
from galpy.util.conversion import time_in_Gyr
from scipy.interpolate import RegularGridInterpolator
from galpy.actionAngle.actionAngleVertical import actionAngleVertical
from galpy.actionAngle import actionAngleVerticalInverse
from multiprocessing.pool import Pool
from galpy.potential import verticalfreq
from darkspirals.orbit_util import satellite_vertical_force
from scipy.integrate import simpson
from galpy.potential import turn_physical_off
import os
from tqdm import tqdm
import multiprocessing as mp


class Disc(object):

    def __init__(self, local_potential, galactic_potential, z_min_max_kpc, vz_min_max_kpc, phase_space_pixels,
                 time_Gyr_eval, units_ro=8.0, units_vo=220.0, compute_action_angle=True,
                 parallelize_action_angle_computation=True, compute_upfront=True, r_over_r0=1.0225):
        """
        The main class in the module; calculates actions, angles, frequencies for a local gravitational potential
        :param local_potential: an instance of a galpy potential that specifies the gravitational potential at the solar position
        :param galactic_potential: an instance of a galpy potential that specifies the gravitational potential of the galaxy
        in which to integrate orbits
        :param z_min_max_kpc: the maximum/minimum vertical distance from the midplane in which to calculate
        actions/angles/frequencies [kpc]
        :param vz_min_max_kpc: the maximum/minimum vertical velocity for which to calculate actions/angles/frequencies [kpc]
        :param phase_space_pixels: the resolution of the phase space; will calculate dynamics on an (NxN) grid with
        N = phase_space_pixels
        :param time_Gyr_eval: time over which to integrate orbits and calculate the effects of a perturbation; should be
        a quantity in Gyr with astropy units that goes backwards in time (i.e. starts at t=0 and ends at t = -T)
        :param units_ro: default galpy internal units [kpc]
        :param units_vo: default galpy internal units [km/sec]
        :param compute_action_angle: bool; perform the calculation of action/angle variables
        :param parallelize_action_angle_computation: bool; do the calculation of action/angle variables with multi-threading
        :param compute_upfront: bool; calculate action/angle variables and orbits in the equilibrium potential on the grid
        immediately when creating the class
        :param r_over_r0: the distance from the galactic center of the solar position in galpy internal units
        """
        turn_physical_off(local_potential)
        turn_physical_off(galactic_potential)
        self.local_potential = local_potential
        self.local_vertical_potential = [toVerticalPotential(local_potential,
                                                            r_over_r0, # the spatial coordinate where we evaluate the 3D potential
                                                            # in units of ro
                                                            phi=0.)]
        self.galactic_potential = galactic_potential
        self.time_Gyr_eval = time_Gyr_eval
        self.units = {'ro': units_ro, 'vo': units_vo}
        self.r_over_r0 = r_over_r0
        try:
            self.time_internal_eval = self.time_to_internal_time(time_Gyr_eval.value)
        except:
            self.time_internal_eval = self.time_to_internal_time(time_Gyr_eval)
        self.z_units_internal = np.linspace(-z_min_max_kpc, z_min_max_kpc, phase_space_pixels) / self.units['ro']
        self.vz_units_internal = np.linspace(-vz_min_max_kpc, vz_min_max_kpc, phase_space_pixels) / self.units['vo']
        self._parallelize = parallelize_action_angle_computation
        self._phase_space_dim = int(len(self.z_units_internal))
        self._time_step = self.time_internal_eval[1] - self.time_internal_eval[0]
        self.aAV = actionAngleVertical(pot=self.local_vertical_potential)

        if compute_upfront:
            _ = self.orbits_in_phase_space
            if compute_action_angle:
                _ = self.action_angle_frequency
                _ = self.action_angle_interp

    def _deltaJ_integral(self, v_z, f, omega, t, delta_v):
        """
        Performs the integral to compute deltaJ
        :param v_z:
        :param f:
        :param omega:
        :param t:
        :return:
        """
        integrand = (v_z + delta_v) * f / omega
        I = np.squeeze(simpson(integrand, x=t))
        return I

    def compute_deltaJ_from_forces(self, forces, parallel=False, n_cpu=None, delta_v_kick=False,
                                   verbose=False):
        """
        Compute the change to the vertical action from an external force
        :param forces: a list of external forces, should be a numpy array with shape(N, N, N_T), where N is the resolution
        of the (z, vz) grid and N_T is the number of time steps in the orbit integration
        :param parallel: bool; do the calculation with multi-threading
        :param n_cpu: number of CPUs for multi-threading
        :param delta_v_kick: True, False, or None; This removes a singularity for orbits with J_z = 0.0 and performs
        deltaJ calculation with v_z -> v_z + delta_vz
        if False: ignores this term
        if True: calculates the impulse, delta_vz, from each perturbing force
        if numpy array or float: use the specified delta_vz
        :return: a list of numpy arrays of shape (N, N) that contain deltaJs corresponding to each perturbation
        """
        v_z = self._phase_space_orbits.vx(self.time_internal_eval)
        delta_J_list = []
        freq_0 = self.frequency[:, :, np.newaxis]
        if parallel:
            arg_list = []
            for counter, f in enumerate(forces):
                arg = (v_z, f, freq_0, self.time_internal_eval)
                arg_list.append(arg)
            with mp.Pool(processes=n_cpu) as pool:
                delta_J_list = pool.starmap(self._deltaJ_integral, arg_list)
        else:
            for counter, f in enumerate(forces):
                dJ = self._deltaJ_integral(v_z, f, freq_0, self.time_internal_eval)
                if verbose and counter%25==0:
                    percent_done = int(100*counter/len(forces))
                    print('completed '+str(percent_done)+'% of action calculations... ')
                delta_J_list.append(dJ)
        return delta_J_list

    def compute_satellite_forces(self, satellite_orbit_list=None,
                                 satellite_potentials_list=None,
                                 verbose=False,
                                 parallel=False,
                                 n_cpu=6,
                                 t_max=None):

        """
        Computes the vertical forces from a list of passing satellites
        :param satellite_orbit_list: list of Orbit instances for each satellite
        :param satellite_potentials_list: list of potentials corresponding to each satellite
        :param verbose: make print statements
        :param t_max: the lookback time over which to return the force exerted, should specify a time in the past (<0)
        :return: a list of vertical forces, impact times (time of maximum force), and impact parameters (minimum distance)
         for satellites
         All quantities returned in internal units
        """
        assert len(satellite_orbit_list) == len(satellite_potentials_list)
        force_list = []
        impact_times = []
        impact_parameters = []
        record_force_array = True
        z_coord = self.orbits_in_phase_space.x(self.time_internal_eval)
        if parallel:
            arg_list = []
            for orbit, potential in zip(satellite_orbit_list, satellite_potentials_list):
                arg = (self, orbit, potential, z_coord, record_force_array, t_max)
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
                                                            record_force_array=record_force_array,
                                                               t_max=t_max)
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
    def action_angle_interp_inverse(self):
        """
        Create the inverse mapping to z, vz as a function of (action, angle, frequency). The action/angle/frequency
        are expected in internal galpy units, and the outputs are given in galpy internal units
        :return:
        """
        if not hasattr(self, '_action_angle_interp_inverse'):
            max_energy = 0
            for i, orbit in enumerate(self.orbits_in_phase_space):
                max_energy = max(np.max(orbit.E()), max_energy)
            energies = np.linspace(1e-2, max_energy, 100)
            self._action_angle_interp_inverse = actionAngleVerticalInverse(pot=self.local_vertical_potential,
                                                                           Es=energies, setup_interp=True)
        return self._action_angle_interp_inverse

    @property
    def action_angle_interp(self):
        """
        Create interpolation of action, angle frequency as a function of (z, vz). z and vz are expected to be
        specified in physical units
        :return:
        """
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
        :return: actions, angles, and frequencies in internal units
        """

        if not hasattr(self, '_action'):

            nz, nv = len(self.z_units_internal), len(self.vz_units_internal)
            action = np.empty([nz, nv])
            frequency = np.empty([nz, nv])
            angle = np.empty([nz, nz])
            aAV = actionAngleVertical(pot=self.local_vertical_potential)
            # r in dimension 0, v in dimension 1
            vxvv = np.array(np.meshgrid(self.z_units_internal, self.vz_units_internal)).T
            z = vxvv[:, :, 0].ravel()
            vz = vxvv[:, :, 1].ravel()
            if self._parallelize:
                action = action.ravel()
                frequency = frequency.ravel()
                angle = angle.ravel()
                arg_list = []
                for i in range(0, len(z)):
                    arg_list.append((z[i], vz[i]))
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
                action = np.empty([nz, nv]).ravel()
                frequency = np.empty([nz, nv]).ravel()
                angle = np.empty([nz, nz]).ravel()
                for i in range(0, len(z)):
                    J, freq, theta = aAV.actionsFreqsAngles(z[i], vz[i])
                    action[i] = J
                    angle[i] = theta
                    frequency[i] = freq
                    try:
                        assert np.isfinite(J)
                        assert np.isfinite(freq)
                        assert np.isfinite(theta)
                    except:
                        print('undefined result with ', (z[i], vz[i]))
                        exit(1)
                action = action.reshape(nz, nv)
                frequency = frequency.reshape(nz, nv)
                angle = angle.reshape(nz, nv)
            self._action = action
            self._angle = angle
            self._frequency = frequency
        return self._action, self._angle, self._frequency

    @property
    def rotation_frequency(self):
        """
        The rotation frequency of the solar neighborhood around galactic center
        :return: the frequency f = 2 pi / T
        """
        return self.circular_velocity / self.r_over_r0

    @property
    def solar_circle(self):
        """
        computes the location of the sun as a function of time
        :param time_Gyr: the time in Gyr
        :return: the x, y coordinate the solar position in internal units
        """
        freq = self.rotation_frequency
        x_solar = self.r_over_r0 * np.cos(freq * self.time_internal_eval)
        y_solar = self.r_over_r0 * np.sin(freq * self.time_internal_eval)
        return x_solar, y_solar

    def solar_circle_velocity(self, t=None):
        """
        computes the location of the sun as a function of time
        :param time_Gyr: the time in Gyr
        :return: the x, y coordinate the solar position in internal units
        """
        if t is None:
            t = self.time_internal_eval
        freq = self.rotation_frequency
        vx_solar = -self.r_over_r0 * freq * np.sin(freq * t)
        vy_solar = self.r_over_r0 * freq * np.cos(freq * t)
        return vx_solar, vy_solar

    @property
    def rho_midplane(self):

        """
        :return: the density evaluated at R_over_R0 in internal units.
        """
        if not hasattr(self, '_rho_midplane'):
            self._rho_midplane = galpy.potential.evaluateDensities(self.galactic_potential, self.r_over_r0, 0., phi=0.)
        return self._rho_midplane

    @property
    def circular_velocity(self):

        """
        :return: the circular velocity at the solar position in internal units
        """
        if not hasattr(self, '_vcirc'):
            self._vcirc = galpy.potential.vcirc(self.galactic_potential, self.r_over_r0)
        return self._vcirc

    @property
    def dJdz(self):
        """
        The derivative of the vertical action with respect to z in internal units
        :return:
        """
        if not hasattr(self, '_djdz'):
            z_step = (self.z_units_internal[1] - self.z_units_internal[0])
            self._djdz = np.gradient(self.action, z_step, axis=0)
        return self._djdz

    @property
    def vertical_frequency(self):

        """
        :return: the second derivative of the local potential at z = 0
        """
        if not hasattr(self, '_d2psi_dz2'):
            self._d2psi_dz2 = verticalfreq(self.local_potential, self.r_over_r0)
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
