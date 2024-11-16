import numpy as np
from darkspirals.distribution_function.base import DistributionFunctionBase
from galpy.actionAngle.actionAngleInverse import actionAngleInverse
from galpy.potential import evaluatelinearPotentials

class DistributionFunctionIsothermal(DistributionFunctionBase):

    def __init__(self, velocity_dispersion, vertical_frequency, action, z_coords, vz_coords, units, fit_midplane=False):
        """

        :param velocity_dispersion:
        :param vertical_frequency:
        :param action:
        :param z_coords: z coordintes in physical units
        :param vz_coords: vz coordinates in physical units
        :param units:
        :param fit_midplane:
        """
        self._action = action
        self._velocity_dispersion = velocity_dispersion
        self._vertical_frequency = vertical_frequency
        super(DistributionFunctionIsothermal, self).__init__(z_coords,
                                                             vz_coords,
                                                             units,
                                                             fit_midplane)

    def update_params(self, velocity_dispersion=None, vertical_frequency=None):
        """
        Update the velocity dispersion of the df
        :return:
        """
        if velocity_dispersion is not None:
            self._velocity_dispersion = velocity_dispersion
        if vertical_frequency is not None:
            self._vertical_frequency = vertical_frequency

    @property
    def function(self):
        """
        Calculates an isothermal distribution function given the action, vertical frequency, and velocity dispersion
        :return: the numerical value of the distribution function
        """
        exp_argument = -self._action * self._vertical_frequency / self._velocity_dispersion ** 2
        df = 1.0 / np.sqrt(2 * np.pi) / self._velocity_dispersion * np.exp(exp_argument)
        num_pixels = len(exp_argument.ravel())
        normalization = np.sum(df) / num_pixels
        return 1.0 / np.sqrt(2 * np.pi) / self._velocity_dispersion * np.exp(exp_argument) / normalization

class DistributionFunctionLiandWidrow2021(DistributionFunctionBase):

    def __init__(self, velocity_dispersion, vertical_frequency, alpha,
                 action, z_coords, vz_coords, units, fit_midplane=False,
                 solve_Ez=False, vertical_potential=None):
        """

        :param velocity_dispersion:
        :param vertical_frequency:
        :param alpha:
        :param action:
        :param z_coords: z coordintes in physical units
        :param vz_coords: vz coordinates in physical units
        :param units:
        :param fit_midplane:
        :param solve_Ez:
        :param vertical_potential:
        """
        self._velocity_dispersion = velocity_dispersion
        self._vertical_frequency = vertical_frequency
        self._alpha = alpha
        self._solve_Ez = solve_Ez
        self._action = action
        if solve_Ez:
            aaV_inverse = actionAngleInverse(pot=vertical_potential, nta=2 * 128,
                                                     Es=np.linspace(0., 2.5, 1501),
                                                     setup_interp=True,
                                                     use_pointtransform=True, pt_deg=7)
            angles = np.array([np.pi/2] * len(action.ravel()))
            [z_out, vz_out] = aaV_inverse(action.ravel(), angles)
            Ez = 0.5 * vz_out ** 2 + evaluatelinearPotentials(vertical_potential, z_out)
            self._Ez = Ez.reshape(action.shape)
        else:
            self._Ez = action * vertical_frequency
        super(DistributionFunctionLiandWidrow2021, self).__init__(
                                                         z_coords,
                                                         vz_coords,
                                                         units,
                                                         fit_midplane)

    def update_params(self, velocity_dispersion=None, alpha=None, vertical_frequency=None):
        """
        Update the parameters of the vdf; velocity dispersion should be set in physical units (km/sec), while
        vertical frequency is expected to be in internal galpy units
        :return:
        """
        if velocity_dispersion is not None:
            self._velocity_dispersion = velocity_dispersion / self._units['vo']
        if alpha is not None:
            self._alpha = alpha
        if vertical_frequency is not None:
            if self._solve_Ez:
                raise Exception('cannot update the vertical frequency if energies are being explicitely solved '
                                'for (solve_Ez=True')
            self._vertical_frequency = vertical_frequency
            self._Ez = self._J * vertical_frequency

    @property
    def function(self):
        """
        Calculates the distribution function model presented by Li & Widrow (2021)
        :return: the numerical value of the distribution function
        """
        df = (1 + self._Ez / (self._alpha * self._velocity_dispersion**2)) ** -self._alpha
        num_pixels = len(self._Ez.ravel())
        normalization = np.sum(df) / num_pixels
        return df / normalization
