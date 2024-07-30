class CachedSingleRealization(object):

    def __init__(self, force_list, deltaJ_list, impact_times, deltaJ_diffusion_iso, deltaJ_diffusion_LW):

        self.force_list = force_list
        self.deltaJ_list = deltaJ_list
        self.impact_times = list(impact_times)
        self.deltaJ_diffusion_iso = deltaJ_diffusion_iso
        self.deltaJ_diffusion_LW = deltaJ_diffusion_LW

    @property
    def data(self):
        return self.force_list, self.deltaJ_list, self.impact_times, self.deltaJ_diffusion_iso, self.deltaJ_diffusion_LW
