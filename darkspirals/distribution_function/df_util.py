import numpy as np
from scipy.optimize import minimize
import multiprocessing as mp
import matplotlib.pyplot as plt


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

def _eval(points, disc, df):
    out = disc.action_angle_interp(points)
    w = np.squeeze(df.interp_df_normalized(points))
    return [out, w]

def _eval(points, disc, df_list):
    out = disc.action_angle_interp(points)
    weights = []
    for j, df in enumerate(df_list):
        w = np.squeeze(df.interp_df_normalized(points))
        weights.append(w)
    return [out, weights]

def action_angle_frequency_sample_parallel(df_list, disc, n_cpu=10, n_per_cpu=1000, zmin_max=1.5, vmin_max=100):
    """
    Calculates the actions/angles/frequencies for a given potential as defined in the Disc class, and returns them
    along with the weights calculated from a series of distribution functions specified with df_list
    :param df_list: a list of DistributionFunction classes to compute weights from action/angles/frequencies
    :param disc: an instance of Disc class
    :param n_cpu: number of cpu's for multi-threading
    :param n_per_cpu: number of samples of (z, v_z) to generate per cpu
    :param zmin_max: minimum and maximum vertical height
    :param vmin_max: minimum and maximum vertical speed
    :return: actions, angles, frequencies (all internal galpy units) and weights
    """
    args = []
    for i in range(0, n_cpu):
        samples_z = np.random.uniform(-zmin_max, zmin_max, n_per_cpu)
        samples_vz = np.random.uniform(-vmin_max, vmin_max, n_per_cpu)
        if i==0:
            samples_stacked = np.column_stack((samples_z, samples_vz))
        else:
            samples_stacked = np.vstack((samples_stacked, np.column_stack((samples_z, samples_vz))))
        points_eval = (np.column_stack((samples_z, samples_vz)), disc, df_list)
        args.append(points_eval)
    if n_cpu == 1:
        [out, weight_list] = _eval(*args[0])
        action, angle, freq = out[:, 0], out[:, 1], out[:, 2]
        return action, angle, freq, weight_list
    else:
        print('using multi-threading to calculate action/angle/frequency... ')
        with mp.Pool(processes=n_cpu) as pool:
            result = pool.starmap(_eval, args)
        for i, res in enumerate(result):
            [out, _weight_list] = res
            _action, _angle, _freq = out[:, 0], out[:, 1], out[:, 2]
            if i == 0:
                action = _action
                angle = _angle
                freq = _freq
                w0 = np.array(_weight_list[0])
                w1 = np.array(_weight_list[1])
                w2 = np.array(_weight_list[2])
            else:
                action = np.append(action, _action)
                angle = np.append(angle, _angle)
                freq = np.append(freq, _freq)
                w0 = np.append(w0, np.array(_weight_list[0]))
                w1 = np.append(w1, np.array(_weight_list[1]))
                w2 = np.append(w2, np.array(_weight_list[2]))
        return action, angle, freq, [w0, w1, w2]

def plot_df_perturbation(df, df_eq, filename=None, annotation='', show_plot=False):
    """
    Makes a plot of a perturbation to the distribution function
    :param df: perturbed distribution function class (see classes in df_models)
    :param df_eq: equilibrium distribution function class (see classes in df_models)
    :param filename: if a string, will save the figure with that path/filename
    """
    try:
        from palettable.scientific.diverging import Vik_20 as colormaps
        cmap = colormaps.get_mpl_colormap()
    except:
        print('could not use default color map because palettable is not installed, using bwr instead')
        cmap = 'bwr'
    cbar_label_pad = -12
    cbar_ticks = [0.9, 0.95, 1.0, 1.05, 1.1]
    cbar_tick_labels = [r'$-0.1$', r'$-0.05$', r'$0.0$', r'$0.05$', r'$0.1$']
    fig = plt.figure()
    ax = plt.subplot(111)
    vminmax = 0.1
    im = ax.imshow(np.rot90(df.function / df_eq.function), origin='upper', vmin=1 - vminmax,
                   vmax=1. + vminmax, cmap=cmap)
    ax.annotate(annotation, xy=(0.03, 0.82), xycoords='axes fraction',
                bbox=dict(facecolor='w', edgecolor='k', pad=5.0, alpha=0.8), fontsize=12)
    # cbar = plt.colorbar(im, ax=ax, pad=0.025)
    # cbar.set_ticks(cbar_ticks)
    # cbar.set_ticklabels(cbar_tick_labels)
    # cbar.set_label(r'$\delta f\left(z, v_z\right)$', labelpad=cbar_label_pad, fontsize=16)
    # cbar.ax.tick_params(labelsize=14)
    zticks = [r'$-1.5$', r'$-0.75$', r'$0.0$', r'$0.75$', r'$1.5$']
    vzticks = [r'$100$', r'$50$', r'$0$', r'$-50$', r'$-100$']
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(zticks, fontsize=15)
    ax.set_yticklabels(vzticks, fontsize=15)
    ax.set_xlabel(r'$z \ \left[\rm{kpc}\right]$', fontsize=16, labelpad=-0)
    ax.set_ylabel(r'$v_z \ \left[ \rm{km} \ \rm{s^{-1}}\right]$', fontsize=16, labelpad=-8)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    if show_plot:
        plt.show()
