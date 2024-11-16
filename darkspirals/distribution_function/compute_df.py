from darkspirals.distribution_function.df_models import  DistributionFunctionIsothermal, \
    DistributionFunctionLiandWidrow2021
from copy import deepcopy

def compute_df_from_orbits(disc_model,
                           velocity_dispersion,
                           satellite_orbit_list,
                           satellite_potential_list,
                           df_model='ISOTHERMAL',
                           alpha=2.34,
                           solve_Ez=False,
                           fit_midplane=True,
                           diffusion_model=None,
                           diffusion_coefficients=None,
                           diffusion_timescale=0.6,
                           overwrite_impact_times=None,
                           parallelize_diffusion_calculation=False,
                           t_max=None,
                           verbose=False
                           ):
    """
    Compute the distribution from a series of orbits and potentials
    :param disc_model: an instance of Disc class (see darkspirals.Disc)
    :param velocity_dispersion: the velocity dispersion that sets the distribution function in km/sec
    :param satellite_orbit_list: a list of Orbit instances for galactic satellites
    :param satellite_potential_list: a list of potentials for each satellite
    :param df_model: either 'ISOTHERMAL' or 'LI&WIDROW', see documentation in distribution function classes
    :param alpha: parameter for the LI&WIDROW distribution class
    :param solve_Ez: bool; solve for the vertical energies to use with the LI&WIDROW class
    :param fit_midplane: bool; fit distribution function symmetrically around the midplane of the stellar disc
    :param diffusion_model: a callable function or method that applies a diffusion model to a given perturbation to the
    action, see docs in diffusion_models
    :param diffusion_model: a function that transforms a perturbation to the action according to some prescription for
    diffusion in phase space
    :param diffusion_coefficients: parameters used in the diffusion model; defaults are chosen based on the specified
    distribution function
    :param diffusion_timescale: a characteristic timescale for the diffusion model; default is 0.6 Gyr
    :param overwrite_impact_times: a list of impact times that replaces the impact times calculated from the actual orbits
    :param parallelize_diffusion_calculation: bool, if True the the diffusion model will be split across multiple CPUs
    :param t_max: set perturbing vertical force = 0 for times in the past > t_max
    :return: distribution function, and the series of individual changes to the action (internal units)
    """
    if verbose:
        print('cacluating satellite forces... ')
    force_list, impact_times, _ = disc_model.compute_satellite_forces(satellite_orbit_list,
                                                                      satellite_potential_list,
                                                                      t_max=t_max)
    if verbose:
        print('done.')
    if overwrite_impact_times is not None:
        assert len(overwrite_impact_times) == len(force_list)
        impact_times = deepcopy(overwrite_impact_times)
    _deltaJ_list = disc_model.compute_deltaJ_from_forces(force_list)
    if diffusion_model is None:
        deltaJ_list = _deltaJ_list
    else:
        if parallelize_diffusion_calculation:
            if verbose:
                print('applying diffusion model in parallel... ')
            deltaJ_list = diffusion_model.compute_parallel(_deltaJ_list,
                                                           impact_times,
                                                           df_model,
                                                           diffusion_coefficients,
                                                           diffusion_timescale,
                                                           nproc=10)
            if verbose:
                print('done.')
        else:
            deltaJ_list = []
            if verbose:
                print('applying diffusion model... ')
            for dj, t in zip(_deltaJ_list, impact_times):
                dj_with_diffusion = diffusion_model(dj,
                                                   t,
                                                   df_model,
                                                   diffusion_coefficients,
                                                   diffusion_timescale)
                deltaJ_list.append(dj_with_diffusion)
            if verbose:
                print('done.')
    deltaJ_net = 0
    for dJ in deltaJ_list:
        deltaJ_net += dJ
    df = compute_df_from_actions(disc_model, velocity_dispersion, deltaJ_net, df_model,
                                     alpha, solve_Ez, fit_midplane)
    return df, deltaJ_list

def compute_df_from_actions(disc_model,
                            velocity_dispersion,
                            deltaJ_net,
                            df_model='ISOTHERMAL',
                            alpha=2.34,
                            solve_Ez=False,
                            fit_midplane=True):
    """
    Computes the distribution function
    :param disc_model: an instance of Disc class (see darkspirals.Disc)
    :param velocity_dispersion: the velocity dispersion that sets the distribution function in km/sec
    :param deltaJ_net: the net change in the action at phase-space coordinates
    :param df_model: either 'ISOTHERMAL' or 'LI&WIDROW', see documentation in distribution function classes
    :param alpha: parameter for the LI&WIDROW distribution class
    :param solve_Ez: bool; solve for the vertical energies to use with the LI&WIDROW class
    :param fit_midplane: bool; fit distribution function symmetrically around the midplane of the stellar disc
    :return: an instance of a distribution function class
    """

    if isinstance(deltaJ_net, list):
        deltaJ = 0
        for dji in deltaJ_net:
            deltaJ += dji
    else:
        deltaJ = deltaJ_net

    if df_model == 'LI&WIDROW':
        dF = DistributionFunctionLiandWidrow2021(velocity_dispersion / disc_model.units['vo'],
                                                 disc_model.vertical_frequency,
                                                 alpha,
                                                 disc_model.action + deltaJ,
                                                 disc_model.z_units_internal * disc_model.units['ro'],
                                                 disc_model.vz_units_internal * disc_model.units['vo'],
                                                 disc_model.units,
                                                 fit_midplane,
                                                 solve_Ez,
                                                 disc_model.local_vertical_potential
                                                 )
    elif df_model == 'ISOTHERMAL':
        dF = DistributionFunctionIsothermal(velocity_dispersion / disc_model.units['vo'],
                                            disc_model.vertical_frequency,
                                            disc_model.action + deltaJ,
                                            disc_model.z_units_internal * disc_model.units['ro'],
                                            disc_model.vz_units_internal * disc_model.units['vo'],
                                            disc_model.units,
                                            fit_midplane)
    else:
        raise Exception('df model '+str(df_model)+' not recognized')

    return dF
