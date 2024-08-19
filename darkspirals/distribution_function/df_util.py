import numpy as np
from scipy.optimize import minimize

def _sec_square_func_to_min(x, zdata, rho_data):
    n0, zsun, H1 = x[0], x[1], x[2]
    if abs(zsun) > 2:
        return np.inf
    if (H1 < 0. or H1 > 5.):
        return np.inf
    z = zdata + zsun
    model = n0 * (1. / np.cosh(z / (2. * H1)) ** 2)
    return np.sum((rho_data - model) ** 2)

def fit_sec_squared(rho, z):
    """
    This routine fits a curve given by rho(z) with a secant^2 function
    :param rho: a curve
    :param z: points at which rho is evaluated
    :return: the normalization, centroid position (or maximum position), and scale height
    """
    guess = np.array([np.max(rho), 0., 0.02])
    opt = minimize(_sec_square_func_to_min, x0=guess, args=(z, rho))
    return opt['x']

def save_stack_deltaJ(delta_J_list, filename):
    """
    Create and save a numpy array with shape (phase_space_resolution, phase_space_resolution, N) where N is the length of
    delta_J_list, or the number of perturbations, and phase_space_resolution is the dimension along each axis of deltaJ

    :param delta_J_list: a list of perturbations to the action
    :return:
    """
    phase_space_resolution = int(delta_J_list[0].shape[0])
    output_shape = (phase_space_resolution, phase_space_resolution, len(delta_J_list))
    output = np.empty(output_shape)
    for i in range(0, len(delta_J_list)):
        output[:, :, i] = delta_J_list[i]
    np.savetxt(filename, X=output.ravel())
