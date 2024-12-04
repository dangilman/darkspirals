import numpy as np
from scipy.ndimage.filters import gaussian_filter
from multiprocessing.pool import Pool
import multiprocessing as mp
from tqdm import tqdm

class DiffusionBase(object):
    """
    Base class for any diffusion model
    """
    def __init__(self, disc_model):
        """

        :param disc_model: an instance of Disc class (see darkspirals.disc.py)
        """
        self._disc_model = disc_model

    def compute_parallel(self, deltaJ_list, impact_time_gyr_list, df_model, diffusion_coefficients=None,
                         diffusion_timescale=0.6, n_cpu=6):
        """
        Parallelize the computation across multiple cpus
        :param deltaJ_list: a list of perturbations to the vertical action; should be list of numpy arrays each with
        shape (N, N) where N is the resolution of the phase-space (number of pixels along each dimension)
        :param impact_time_gyr_list: a list of impact times in Gyr; note each time should be a positive number specifying
        how many Gyr ago a perturbation occured
        :param df_model: a string that specifies the distribution function model
        :param diffusion_coefficients: parameters for the diffusion model
        :param diffusion_timescale: a characteristic timescale for the diffusion process
        :param nproc: number of CPUs
        :return: a list of deltaJ with diffusion applied
        """
        arg_list = []
        for dj, t_gyr in zip(deltaJ_list, impact_time_gyr_list):
            arg = (dj, t_gyr, df_model, diffusion_coefficients, diffusion_timescale)
            arg_list.append(arg)
        with mp.Pool(processes=n_cpu) as pool:
            results = pool.map(self._call_parallel, arg_list)
        return results

    def _call_parallel(self, x):
        """

        :param x:
        :return:
        """
        return self(*x)

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        raise Exception('call method not specified!')

class DiffusionConvolutionSpatiallyVarying(DiffusionBase):
    """
    This implements Gaussian convolutions with a spatially-varying kernel
    """
    def __call__(self, deltaJ, impact_time_gyr, df_model, diffusion_coefficients=None,
                 diffusion_timescale=0.6, verbose=False):
        """
        This class implements diffusion for the change in a distribution function by applying a series of Gaussian
        convolutions to vertical action. The convolutions have spatially-varying kernels. This method applies the diffusion
        model to ONE deltaJ array

        :param deltaJ: a numpy array giving the perturbation to the vertical action
        :param impact_time_gyr: time since a perturbation event [Gyr]; should be a positive number
        :param df_model: a string specifying the distribution function; currently this is not used
        :param diffusion_coefficients: coefficients that normalize the diffusion kernels; will take on default values
        if not specified
        :param diffusion_timescale: the timescale for the diffusion process, defined as the time when a perturbation is
        damped by 1/e
        :param verbose: make print statements
        :return: a numpy array with the same shape as deltaJ that has diffusion applied to it
        """

        if diffusion_coefficients is None:
            diffusion_coefficients = (0.24, 1.0)
        tau = abs(impact_time_gyr) / diffusion_timescale
        prefactor = diffusion_coefficients[0] * tau ** diffusion_coefficients[1]
        num_pixel_per_kpc = len(self._disc_model.z_units_internal) / \
                            (np.max(self._disc_model.z_units_internal) - np.min(self._disc_model.z_units_internal))
        num_pixel_per_vz = len(self._disc_model.vz_units_internal) / \
                           (np.max(self._disc_model.vz_units_internal) - np.min(self._disc_model.vz_units_internal))
        zz, vzvz = np.meshgrid(self._disc_model.z_units_internal,
                               self._disc_model.vz_units_internal)
        z_min, z_max = np.min(self._disc_model.z_units_internal), np.max(self._disc_model.z_units_internal)
        vz_min, vz_max = np.min(self._disc_model.vz_units_internal), np.max(self._disc_model.vz_units_internal)
        n = int(len(self._disc_model.z_units_internal) / 2)
        z_coords = np.linspace(z_min, z_max, n)
        vz_coords = np.linspace(vz_min, vz_max, n)
        z_step = z_coords[1] - z_coords[0]
        vz_step = vz_coords[1] - vz_coords[0]
        dJ = np.ones_like(deltaJ).ravel()
        (N, N) = self._disc_model.action.shape
        counter = 0
        for z_i in z_coords:
            for vz_i in vz_coords:
                if verbose and counter % 25 == 0:
                    percent_done = int(100 * counter / N**2)
                    print('performed ' + str(percent_done) + '% of convolutions... ')
                x = (z_i * self._disc_model.units['ro'], vz_i * self._disc_model.units['vo'])
                j_i, _, omega_i = self._disc_model.action_angle_interp(x)
                kernel_z = np.sqrt(j_i / omega_i)
                kernel_vz = np.sqrt(j_i * omega_i)
                kernel_pixel_z = prefactor * kernel_z * num_pixel_per_kpc
                kernel_pixel_vz = prefactor * kernel_vz * num_pixel_per_vz
                dj_temp = np.squeeze(gaussian_filter(deltaJ, (kernel_pixel_z, kernel_pixel_vz))).ravel()
                dz = np.absolute(z_i - zz).ravel()
                dvz = np.absolute(vz_i - vzvz).ravel()
                cond1 = dz <= z_step / 2
                cond2 = dvz <= vz_step / 2
                inds = np.where(np.logical_and(cond1, cond2))[0]
                dJ[inds] = dj_temp[inds]
                counter += 1
        return np.squeeze(dJ.reshape(N, N))
