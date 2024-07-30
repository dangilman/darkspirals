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
        :param z_coords:
        :param vz_coords:
        :param units:
        :param fit_midplane:
        """
        self._velocity_dispersion = velocity_dispersion
        self._vertical_frequency = vertical_frequency
        super(DistributionFunctionIsothermal, self).__init__(action,
                                                             vertical_frequency,
                                                             z_coords,
                                                             vz_coords,
                                                             units,
                                                             fit_midplane)

    @property
    def function(self):
        """
        Calculates an isothermal distribution function given the action, vertical frequency, and velocity dispersion
        :return: the numerical value of the distribution function
        """
        exp_argument = -self._J * self._vertical_freq / self._velocity_dispersion ** 2
        return 1.0 / np.sqrt(2 * np.pi) / self._velocity_dispersion * np.exp(exp_argument)

class DistributionFunctionLiandWidrow2021(DistributionFunctionBase):

    def __init__(self, velocity_dispersion, vertical_frequency, alpha,
                 action, z_coords, vz_coords, units, fit_midplane=False,
                 solve_Ez=False, vertical_potential=None):
        """

        :param velocity_dispersion:
        :param vertical_frequency:
        :param alpha:
        :param action:
        :param z_coords:
        :param vz_coords:
        :param units:
        :param fit_midplane:
        :param solve_Ez:
        :param vertical_potential:
        """
        self._velocity_dispersion = velocity_dispersion
        self._vertical_frequency = vertical_frequency
        self._alpha = alpha
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
        super(DistributionFunctionLiandWidrow2021, self).__init__(action,
                                                         vertical_frequency,
                                                         z_coords,
                                                         vz_coords,
                                                         units,
                                                         fit_midplane)

    @property
    def function(self):
        """
        Calculates the distribution function model presented by Li & Widrow (2021)
        :return: the numerical value of the distribution function
        """
        df = (1 + self._Ez / (self._alpha * self._velocity_dispersion**2)) ** -self._alpha
        return df
