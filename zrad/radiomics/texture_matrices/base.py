import numpy as np


class TextureMatrixBase:
    """Common utilities for texture-matrix based feature extractors."""

    def __init__(self, image, slice_weight=False, slice_median=False):
        self.image = image  # Import image as (x, y, z) array
        self.lvl = int(np.nanmax(self.image) + 1)
        self.tot_no_of_roi_voxels = np.sum(~np.isnan(image))
        self.slice_weight = slice_weight
        self.slice_median = slice_median

        x_indices, y_indices, z_indices = np.where(~np.isnan(self.image))

        self.range_x = np.unique(x_indices)
        self.range_y = np.unique(y_indices)
        self.range_z = np.unique(z_indices)

        self.reset_fields()

    def reset_fields(self):
        self.short_runs_emphasis = 0
        self.long_runs_emphasis = 0
        self.low_grey_level_run_emphasis = 0
        self.high_gr_lvl_emphasis = 0
        self.short_low_gr_lvl_emphasis = 0
        self.short_high_gr_lvl_emphasis = 0
        self.long_low_gr_lvl_emphasis = 0
        self.long_high_gr_lvl_emphasis = 0
        self.non_uniformity = 0
        self.norm_non_uniformity = 0
        self.length_non_uniformity = 0
        self.norm_length_non_uniformity = 0
        self.percentage = 0
        self.gr_lvl_var = 0
        self.length_var = 0
        self.entropy = 0
        self.energy = 0

        self.short_runs_emphasis_list = []
        self.long_runs_emphasis_list = []
        self.low_grey_level_run_emphasis_list = []
        self.high_gr_lvl_emphasis_list = []
        self.short_low_gr_lvl_emphasis_list = []
        self.short_high_gr_lvl_emphasis_list = []
        self.long_low_gr_lvl_emphasis_list = []
        self.long_high_gr_lvl_emphasis_list = []
        self.non_uniformity_list = []
        self.norm_non_uniformity_list = []
        self.length_non_uniformity_list = []
        self.norm_length_non_uniformity_list = []
        self.percentage_list = []
        self.gr_lvl_var_list = []
        self.length_var_list = []
        self.entropy_list = []
        self.energy_list = []

    # The following methods compute statistics that are reused across different
    # texture matrices.
    def calc_short_emphasis(self, m):
        Ns = np.sum(m)
        _, j = np.indices(m.shape)
        return np.sum(m / (j + 1) ** 2) / Ns

    def calc_long_emphasis(self, m):
        Ns = np.sum(m)
        _, j = np.indices(m.shape)
        return np.sum(m * (j + 1) ** 2) / Ns

    def calc_low_gr_lvl_emphasis(self, M):
        Ns = np.sum(M)
        i, _ = np.indices(M.shape)
        mask = i != 0
        return np.sum(M[mask] / (i[mask]) ** 2) / Ns

    def calc_high_gr_lvl_emphasis(self, m):
        Ns = np.sum(m)
        i, _ = np.indices(m.shape)
        return np.sum(m * i ** 2) / Ns

    def calc_short_low_gr_lvl_emphasis(self, M):
        Ns = np.sum(M)
        i, j = np.indices(M.shape)
        mask = i != 0
        M_j = M[mask] / (i[mask] ** 2)
        return np.sum(M_j / ((j[mask] + 1) ** 2)) / Ns

    def calc_short_high_gr_lvl_emphasis(self, M):
        Ns = np.sum(M)
        i, j = np.indices(M.shape)
        return np.sum((i ** 2 * M) / ((j + 1)) ** 2) / Ns

    def calc_long_low_gr_lvl_emphasis(self, M):
        Ns = np.sum(M)
        i, j = np.indices(M.shape)
        mask = i != 0
        return np.sum((M[mask] * (j[mask] + 1) ** 2) / (i[mask]) ** 2) / Ns

    def calc_long_high_gr_lvl_emphasis(self, M):
        n_s = np.sum(M)
        i, j = np.indices(M.shape)
        return np.sum(M * (j + 1) ** 2 * i ** 2) / n_s

    def calc_non_uniformity(self, M):
        Ns = np.sum(M)
        return np.sum(np.sum(M, axis=1) ** 2) / Ns

    def calc_norm_non_uniformity(self, M):
        Ns = np.sum(M)
        return np.sum(np.sum(M, axis=1) ** 2) / Ns ** 2

    def calc_length_non_uniformity(self, M):
        Ns = np.sum(M)
        return np.sum(np.sum(M, axis=0) ** 2) / Ns

    def calc_norm_length_non_uniformity(self, M):
        Ns = np.sum(M)
        return np.sum(np.sum(M, axis=0) ** 2) / Ns ** 2

    def calc_percentage(self, M, Nv):
        Ns = np.sum(M)
        return Ns / Nv

    def calc_gr_lvl_var(self, M):
        Ns = np.sum(M)
        i, _ = np.indices(M.shape)
        mu = np.sum(M * i / Ns)
        return np.sum((i - mu) ** 2 * (M / Ns))

    def calc_length_var(self, M):
        Ns = np.sum(M)
        _, j = np.indices(M.shape)
        mu = np.sum(M * j / Ns)
        return np.sum((j - mu) ** 2 * (M / Ns))

    def calc_entropy(self, M):
        Ns = np.sum(M)
        mask = M != 0
        return np.sum((M[mask] / Ns) * np.log2((M[mask] / Ns))) * (-1)

    def calc_energy(self, M):
        Ns = np.sum(M)
        mask = M != 0
        return np.sum((M[mask] / Ns) ** 2)
