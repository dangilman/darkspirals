from darkspirals.disc import Disc
from darkspirals.workflow.cache_class import CachedSingleRealization
from darkspirals.distribution_function.compute_df import compute_df_from_actions
from galpy.potential import MWPotential2014
import pickle
import numpy as np
import astropy.units as apu
from darkspirals.diffusion import DiffusionConvolution
import matplotlib.pyplot as plt


f = open('local_potential_MWPot2014', 'rb')
disc = pickle.load(f)
f.close()

f = open('./cache/realization_1', 'rb')
realization_data = pickle.load(f)
f.close()

force_list, deltaJ_list, impact_times, deltaJ_diffusion_iso, deltaJ_diffusion_LW = realization_data.data

for idx_plot in [1,2,3]:
    print(impact_times[idx_plot])
    plt.imshow(deltaJ_list[idx_plot])
    plt.show()
    plt.imshow(deltaJ_diffusion_iso[idx_plot])
    plt.show()

velocity_dispersion = 20.0
df_iso = compute_df_from_actions(disc, velocity_dispersion, deltaJ_list, 'ISOTHERMAL')
df_iso_diffusion = compute_df_from_actions(disc, velocity_dispersion, deltaJ_diffusion_iso, 'ISOTHERMAL')
df_eq = compute_df_from_actions(disc, velocity_dispersion, [], 'ISOTHERMAL')
plt.imshow(df_iso.function.T / df_eq.function.T-1, origin='lower', vmin=-0.025, vmax=0.025)
plt.show()
plt.imshow(df_iso_diffusion.function.T / df_eq.function.T -1, origin='lower', vmin=-0.025, vmax=0.025)
plt.show()
