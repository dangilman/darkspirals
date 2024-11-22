from darkspirals.disc import Disc
from darkspirals.substructure.realization import SubstructureRealization
from galpy.potential import MWPotential2014
import numpy as np
import astropy.units as apu
import os
from darkspirals.diffusion import DiffusionConvolutionSpatiallyVarying
import sys
from darkspirals.distribution_function.df_util import save_stack_deltaJ

def program():

    n_cpu = 10
    seed_start = 42
    log_mlow = 6.0
    log_mhigh = 9.0
    norm = 500.0
    z_min_max = 1.5
    vz_min_max = 100
    phase_space_resolution = 100
    galactic_potential = MWPotential2014
    time_Gyr = np.linspace(0.0, -1.2, 1000) * apu.Gyr
    print('setting up disk model... ')
    parallel = True
    disc = Disc(galactic_potential, galactic_potential, z_min_max, vz_min_max, phase_space_resolution,
                time_Gyr, parallelize_action_angle_computation=parallel, compute_upfront=True)

    for seed_increment in range(0, 1000):

        seed = seed_start + seed_increment
        print(seed)
        np.random.seed(seed)
        t_max = None
        diffusion_timescale = 0.9

        r_min = 30
        realization = SubstructureRealization.withDistanceCut(disc, norm=norm, r_min=r_min,
                                                              t_max=t_max, num_halos_scale=1.0, m_low=10**log_mlow,
                                                              m_high=10**log_mhigh)
        print('realization contains '+str(len(realization.subhalo_orbits))+' subhalos')
        print('computing perturbing forces... ')
        force_list, impact_times, _ = disc.compute_satellite_forces(realization.orbits, realization.potentials,
                                                                    parallel=True, n_cpu=n_cpu, t_max=t_max)
        print('computing action perturbations... ')
        deltaJ_list = disc.compute_deltaJ_from_forces(force_list, parallel=True, n_cpu=n_cpu)
        df_model='LI&WIDROW'
        diffusion_convolution = DiffusionConvolutionSpatiallyVarying(disc)
        print('evaluating diffusion model... ')
        deltaJ_list_with_diffusion_LW = diffusion_convolution.compute_parallel(deltaJ_list, impact_times,
                                                                            df_model, diffusion_timescale=diffusion_timescale,
                                                                               n_cpu=n_cpu)
        output_path = os.getcwd()
        #output_path = os.getenv('SCRATCH')
        folder = '/dfs_tmax12_tdiff9/'
        fname = output_path + folder + 'dj_seed'+str(seed)+'_nodiffusion.txt'
        save_stack_deltaJ(deltaJ_list, fname)
        fname = output_path + folder + 'dj_seed'+str(seed)+'_withdiffusion_LW.txt'
        save_stack_deltaJ(deltaJ_list_with_diffusion_LW, fname)
        fname = output_path + folder + 'subhalomass_seed'+str(seed)+'.txt'
        np.savetxt(fname, X=np.log10(realization.subhalo_masses))

if __name__ == '__main__':
    program()
