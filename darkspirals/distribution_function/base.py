from scipy.interpolate import RegularGridInterpolator
import numpy as np
from darkspirals.distribution_function.df_util import fit_sec_squared
from scipy.interpolate import interp1d
from scipy.integrate import simps

class DistributionFunctionBase(object):

    def __init__(self, z_coords, vz_coords, units, fit_midplane):
        """

        :param z_coords: z coordintes in physical units
        :param vz_coords: vz coordinates in physical units
        :param units: the internal units used by galpy
        :param fit_midplane: bool; compute the z-coordinate of the midplane
        """
        n = len(z_coords)
        assert len(vz_coords) == n
        self._n = n
        self._v = np.array(vz_coords) / units['vo']  # velocity domain in km/sec
        self._z = np.array(z_coords) / units['ro']
        self._units = units
        points_interp = (z_coords, vz_coords)
        if fit_midplane:
            self._z_midplane = np.round(fit_sec_squared(self.density, self._z)[1], 2)
        else:
            self._z_midplane = 0.0
        df = self.function
        # interp done for physical units
        self.interp_df = RegularGridInterpolator(points_interp, df)
        self.interp_df_normalized = RegularGridInterpolator(points_interp, df / np.max(df))

    def sample(self, N_samples=1000):
        """
        Draw samples of (z, v_z) from the distribution function with rejection sampling
        :return: (z, v_z) in physical units
        """
        z = []
        vz = []
        while True:
            _z = np.random.uniform(min(self._z), max(self._z)) * self._units['ro']
            _vz = np.random.uniform(min(self._v), max(self._v)) * self._units['vo']
            p = float(self.interp_df_normalized((_z, _vz)))
            u = np.random.rand()
            if p > u:
                z.append(_z)
                vz.append(_vz)
            if len(z) == N_samples:
                break
        return np.array(z), np.array(vz)

    def sample_action_angle_freq(self, disc, N_samples=10000, physical_output=False, z_max=None, vz_max=None):
        """
        Sample (J(z, vz), theta(z, vz)) weighted by df(z, vz)
        :return:
        """
        N_samples = int(N_samples)
        zmin_max = float(np.max(np.absolute(self._z * self._units['ro'])))
        if z_max is not None:
            zmin_max = min(z_max, zmin_max)
        vmin_max = float(np.max(np.absolute(self._v * self._units['vo'])))
        if vz_max is not None:
            vmin_max = min(vmin_max, vz_max)
        samples_z_phys = np.random.uniform(-zmin_max, zmin_max, N_samples)
        samples_vz_phys = np.random.uniform(-vmin_max, vmin_max, N_samples)
        J = []
        weights = []
        theta = []
        frequencies = []
        for (zi, vzi) in zip(samples_z_phys.ravel(), samples_vz_phys.ravel()):
            j, angle, freq = disc.action_angle_interp((zi, vzi))
            w = float(self.interp_df_normalized((zi,vzi)))
            J.append(j)
            theta.append(angle)
            weights.append(w)
            frequencies.append(freq)
        if physical_output:
            jphys = self._units['ro'] * self._units['vo']
            freq_phys = self._units['vo'] / self._units['ro']
            return np.array(J) * jphys, \
                   np.array(theta), \
                   np.array(frequencies) * freq_phys, \
                   np.array(weights)
        else:
            return np.array(J), np.array(theta), np.array(frequencies), np.array(weights)

    def update_params(self, *args, **kwargs):
        """
        Changes the parameters describing the distribution function, this is model-specific so it should be defined
        separately for each distribution function class
        :return:
        """
        raise Exception('update params not defined for this class')

    def frequency_angle_representation_grid(self, disc, N_samples=10**8, nbins=50):
        """
        Derives the distribution function in frequency-angle coordinates on a grid
        This is essentially the same calculation as in frequency_angle_representation, however, this allows for much
        higher sampling because the histogram is iteratively updated during the sampling, rather than doing all of the
        sampling up front, which consumes a lot of RAM
        :param disc: an instance of Disc used to compute the frequency angle coordinates
        :param N_samples: the number of samples to draw from the phase space distribution
        :return: frequency coordinates, angle coordinates, and importance weights
        """
        N_samples = int(N_samples)
        blocks = int(1e8)
        n_blocks = int(N_samples / blocks)
        h = 0
        for i in range(0, n_blocks):
            freq, angle, weights = self.frequency_angle_representation(disc, blocks)
            if i==0:
                phys_gyr = 28.05
                max_freq = np.max(freq)
                min_freq = 50.0 / phys_gyr
                freq_range = (min_freq, max_freq)
                angle_range = (0, 2 * np.pi)
                hist_range = (freq_range, angle_range)
            _h, angle_vals, freq_vals = np.histogram2d(freq, angle, weights=weights, range=hist_range, bins=nbins)
            h += _h
        return h/n_blocks, angle_range, freq_range

    def frequency_angle_representation(self, disc, N_samples=10**7):
        """
        Derives the distribution function in frequency-angle coordinates
        :param disc: an instance of Disc used to compute the frequency angle coordinates
        :param N_samples: the number of samples to draw from the phase space distribution
        :return: frequency coordinates, angle coordinates, and importance weights
        """
        N_samples = int(N_samples)
        zmin_max = float(np.max(np.absolute(self._z * self._units['ro'])))
        vmin_max = float(np.max(np.absolute(self._v * self._units['vo'])))
        samples_z = np.random.uniform(-zmin_max, zmin_max, N_samples)
        samples_vz = np.random.uniform(-vmin_max, vmin_max, N_samples)
        samples_z_vz = np.column_stack((samples_z, samples_vz))
        weights = np.squeeze(self.interp_df_normalized(samples_z_vz))
        out = disc.action_angle_interp(samples_z_vz)
        action, angle, freq = out[:, 0], out[:, 1], out[:, 2]
        return freq, angle, weights

    def mean_vertical_velocity(self, z=None, subtract_mean=False):
        """

        :param z:
        :return:
        """
        domain = self._z * self._units['ro']
        vz_mean = self.mean_v_relative
        if z is None:
            if subtract_mean:
                vz_out = np.array(vz_mean) - np.mean(vz_mean)
                return np.array(domain), vz_out
            else:
                return np.array(domain), np.array(vz_mean)
        else:
            vz_mean_interp = interp1d(domain, vz_mean)
            vz_mean = []
            domain = []
            for i in range(0, len(z) - 1):
                z_array = np.linspace(z[i], z[i + 1], 20)
                meanv = np.mean(np.squeeze(vz_mean_interp(z_array)))
                vz_mean.append(meanv)
                z_midpoint = (z[i + 1] + z[i]) / 2
                domain.append(z_midpoint)
            domain = np.array(domain)
            vz_mean = np.array(vz_mean)
            if subtract_mean:
                vz_mean -= np.mean(vz_mean)
            return np.array(domain), np.array(vz_mean)

    def vertical_asymmetry(self, z=None):
        """
        Calculate the vertical asymmetry
        :return:
        """
        kwargs_interp = {'fill_value': 'extrapolate', 'kind': 'cubic'}
        func = interp1d(self._z - self._z_midplane, np.log(self.density), **kwargs_interp)
        z_eval = np.linspace(-self._z[-1] - self._z_midplane, self._z[-1] - self._z_midplane, int(len(self._z)))
        p = np.exp(func(z_eval))
        A = (p - p[::-1]) / (p + p[::-1])
        idx_mid = int(len(self._z) / 2)
        domain = self._z[idx_mid:] * self._units['ro']
        asymmetry = A[idx_mid:]
        if z is not None:
            asym_interp = interp1d(domain, asymmetry)
            asymmetry = []
            domain = []
            for i in range(0, len(z)-1):
                z_array = np.linspace(z[i], z[i+1], 20)
                asym = np.mean(np.squeeze(asym_interp(z_array)))
                asymmetry.append(asym)
                z_midpoint = (z[i+1] + z[i]) / 2
                domain.append(z_midpoint)
        return np.array(domain), np.array(asymmetry)

    @property
    def function(self):
        """
        :return: the numerical value of the distribution function
        """
        raise Exception("this method needs to be specified in a sub-class for a specific distribution function model")

    def density_z(self, z):
        """
        returns the density at a height z above or below the midplane
        :param z: height in kpc
        :return: density at z
        """
        z /= self._units['ro']
        return self._interpolated_density(z)

    @property
    def density(self):
        """
        calculates the density from the zeroth moment of the df
        :return: the density in units of solar masses per pc^2
        """
        return self._moment(0.0)

    @property
    def _interpolated_density(self):
        """
        Calculate a spline interpolation of the density
        :return: function that returns density given a height z in internal units
        """
        if not hasattr(self, '_rho_interp'):
            self._rho_interp = interp1d(self._z, self.density, fill_value='extrapolate')
        return self._rho_interp

    @property
    def mean_v(self):
        """
        calculate the mean vertical velocity, <v_z>, as a function of height in km/sec
        :return: <v_z(z)>
        """
        return self._moment(1) * self._units['vo'] / self._moment(0.0)

    @property
    def mean_v_relative(self):
        """
        Computes the mean vertical velocity as a function of height minus the value at the midplane (z=0)
        :return: the mean vertical velocity as a function of height minus <v_z(0)>
        """
        mean_vz = self.mean_v
        idx_mid = int(len(self._z) / 2)
        return mean_vz - mean_vz[idx_mid]

    @property
    def velocity_dispersion(self):
        """
        Computes the velocity dispersion <v_z^2> - <v_z>^2 from the distribution function
        :return: the velocity dispersion as a function of height in km/sec
        """
        v2 = self._moment(2) / self._moment(0) - (self._moment(1) / self._moment(0)) ** 2
        return np.sqrt(v2) * self._units['vo']

    def _moment(self, n):
        """
        Calculates the n-th moment of the distribution function along the second axis; when the phase space is specified
        as (z, vz), this will integrate over velocity
        :param power:
        :return:
        """
        v_array = self._v[None, :] ** n
        return simps(self.function * v_array, self._v, axis=1)
