import numpy as np
from scipy.ndimage.filters import gaussian_filter
from multiprocessing.pool import Pool
import multiprocessing as mp
from tqdm import tqdm

class DiffusionBase(object):

    def __init__(self, disc_model):
        """

        :param disc_model:
        """
        self._disc_model = disc_model

    def compute_parallel(self, deltaJ_list, impact_time_gyr_list, df_model, diffusion_coefficients=None,
                         diffusion_timescale=0.6, nproc=6):
        """
        Parallelize the computation across multiple cpus
        :param deltaJ_list:
        :param impact_time_gyr_list:
        :param df_model:
        :param diffusion_coefficients:
        :param diffusion_timescale:
        :param nproc: number of CPUs
        :return:
        """
        arg_list = []
        for dj, t_gyr in zip(deltaJ_list, impact_time_gyr_list):
            arg = (dj, t_gyr, df_model, diffusion_coefficients, diffusion_timescale)
            arg_list.append(arg)
        # pool = Pool(nproc)
        # result = pool.starmap(self, arg_list)
        # pool.close()
        with mp.Pool(processes=nproc) as pool:
            results = tqdm(
                pool.map(self._call_parallel, arg_list)
            )
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

class DiffusionConvolution(DiffusionBase):

    def __call__(self, deltaJ, impact_time_gyr, df_model, diffusion_coefficients=None,
                 diffusion_timescale=0.6, verbose=False):
        """

        :param deltaJ:
        :param impact_time_gyr:
        :param df_model:
        :param diffusion_coefficients:
        :param diffusion_timescale:
        :return:
        """
        if diffusion_coefficients is None:
            if df_model == 'ISOTHERMAL':
                diffusion_coefficients = (0.021, 1.47)
            else:
                diffusion_coefficients = (0.028, 1.47)
        j0 = np.mean(self._disc_model.action)
        omega0 = np.mean(self._disc_model.frequency)
        tau = abs(impact_time_gyr) / diffusion_timescale
        prefactor = diffusion_coefficients[0] * tau ** diffusion_coefficients[1]
        kernel_z = np.sqrt(j0 / omega0) * self._disc_model.units['ro']
        kernel_vz = np.sqrt(j0 * omega0) * self._disc_model.units['vo']  # internal units now
        num_pixel_per_kpc = len(self._disc_model.z_units_internal) / \
                            (np.max(self._disc_model.z_units_internal) - np.min(self._disc_model.z_units_internal)) / self._disc_model.units['ro']
        num_pixel_per_vz = len(self._disc_model.vz_units_internal) / \
                           (np.max(self._disc_model.vz_units_internal) - np.min(self._disc_model.vz_units_internal)) / self._disc_model.units['vo']
        kernel_pixel_z = prefactor * kernel_z * num_pixel_per_kpc
        kernel_pixel_vz = prefactor * kernel_vz * num_pixel_per_vz
        if verbose:
            print('impact time: ', impact_time_gyr)
            print('kernel pixel sizes (z, vz): ', kernel_pixel_z, kernel_pixel_vz)
        if kernel_pixel_z < 0 or kernel_pixel_vz < 0:
            raise Exception('pixel sizes for convolution should be greater than or equal to zero pixels')
        else:
            if kernel_pixel_z == 0 or kernel_pixel_vz == 0:
                dJ = deltaJ
            else:
                dJ = gaussian_filter(deltaJ, (kernel_pixel_z, kernel_pixel_vz))
        return np.squeeze(dJ)

class DiffusionRescaling(DiffusionBase):

    def __call__(self, deltaJ, impact_time_gyr, df_model, diffusion_coefficients=None,
                 diffusion_timescale=0.6, verbose=False):
        """

        :param deltaJ:
        :param impact_time_gyr:
        :param df_model:
        :param diffusion_coefficients:
        :param diffusion_timescale:
        :return:
        """
        if diffusion_coefficients is None:
            if df_model == 'ISOTHERMAL':
                diffusion_coefficients = (2.2, 4.0)
            else:
                diffusion_coefficients = (2.2, 4.0)
        tau = abs(impact_time_gyr / diffusion_timescale)
        rescaling = 1.0/(1+diffusion_coefficients[0]*tau**diffusion_coefficients[1])
        dJ = np.ones_like(deltaJ) * deltaJ * rescaling
        return np.squeeze(dJ)

