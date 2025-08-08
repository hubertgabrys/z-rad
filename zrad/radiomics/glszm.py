import numpy as np

from ._texture_matrix_base import ZoneMatrixBase


class GLSZM(ZoneMatrixBase):
    """Gray Level Size Zone Matrix (GLSZM) feature extractor."""

    def calc_2d_glszm_features(self):
        number_of_slices = self.glszm_2D_matrices.shape[0]
        weights = []
        for i in range(number_of_slices):
            glszm_slice = self.glszm_2D_matrices[i]
            weight = 1
            if self.slice_weight:
                weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
            weights.append(weight)
            self.short_runs_emphasis_list.append(self.calc_short_emphasis(glszm_slice))
            self.long_runs_emphasis_list.append(self.calc_long_emphasis(glszm_slice))
            self.low_grey_level_run_emphasis_list.append(self.calc_low_gr_lvl_emphasis(glszm_slice))
            self.high_gr_lvl_emphasis_list.append(self.calc_high_gr_lvl_emphasis(glszm_slice))
            self.short_low_gr_lvl_emphasis_list.append(self.calc_short_low_gr_lvl_emphasis(glszm_slice))
            self.short_high_gr_lvl_emphasis_list.append(self.calc_short_high_gr_lvl_emphasis(glszm_slice))
            self.long_low_gr_lvl_emphasis_list.append(self.calc_long_low_gr_lvl_emphasis(glszm_slice))
            self.long_high_gr_lvl_emphasis_list.append(self.calc_long_high_gr_lvl_emphasis(glszm_slice))
            self.non_uniformity_list.append(self.calc_non_uniformity(glszm_slice))
            self.norm_non_uniformity_list.append(self.calc_norm_non_uniformity(glszm_slice))
            self.length_non_uniformity_list.append(self.calc_length_non_uniformity(glszm_slice))
            self.norm_length_non_uniformity_list.append(self.calc_norm_length_non_uniformity(glszm_slice))
            self.percentage_list.append(self.calc_percentage(glszm_slice, self.no_of_roi_voxels[i]))
            self.gr_lvl_var_list.append(self.calc_gr_lvl_var(glszm_slice))
            self.length_var_list.append(self.calc_length_var(glszm_slice))
            self.entropy_list.append(self.calc_entropy(glszm_slice))
        if self.slice_median and not self.slice_weight:
            self.short_runs_emphasis = np.median(self.short_runs_emphasis_list)
            self.long_runs_emphasis = np.median(self.long_runs_emphasis_list)
            self.low_grey_level_run_emphasis = np.median(self.low_grey_level_run_emphasis_list)
            self.high_gr_lvl_emphasis = np.median(self.high_gr_lvl_emphasis_list)
            self.short_low_gr_lvl_emphasis = np.median(self.short_low_gr_lvl_emphasis_list)
            self.short_high_gr_lvl_emphasis = np.median(self.short_high_gr_lvl_emphasis_list)
            self.long_low_gr_lvl_emphasis = np.median(self.long_low_gr_lvl_emphasis_list)
            self.long_high_gr_lvl_emphasis = np.median(self.long_high_gr_lvl_emphasis_list)
            self.non_uniformity = np.median(self.non_uniformity_list)
            self.norm_non_uniformity = np.median(self.norm_non_uniformity_list)
            self.length_non_uniformity = np.median(self.length_non_uniformity_list)
            self.norm_length_non_uniformity = np.median(self.norm_length_non_uniformity_list)
            self.percentage = np.median(self.percentage_list)
            self.gr_lvl_var = np.median(self.gr_lvl_var_list)
            self.length_var = np.median(self.length_var_list)
            self.entropy = np.median(self.entropy_list)
        elif not self.slice_median:
            self.short_runs_emphasis = np.average(self.short_runs_emphasis_list, weights=weights)
            self.long_runs_emphasis = np.average(self.long_runs_emphasis_list, weights=weights)
            self.low_grey_level_run_emphasis = np.average(self.low_grey_level_run_emphasis_list, weights=weights)
            self.high_gr_lvl_emphasis = np.average(self.high_gr_lvl_emphasis_list, weights=weights)
            self.short_low_gr_lvl_emphasis = np.average(self.short_low_gr_lvl_emphasis_list, weights=weights)
            self.short_high_gr_lvl_emphasis = np.average(self.short_high_gr_lvl_emphasis_list, weights=weights)
            self.long_low_gr_lvl_emphasis = np.average(self.long_low_gr_lvl_emphasis_list, weights=weights)
            self.long_high_gr_lvl_emphasis = np.average(self.long_high_gr_lvl_emphasis_list, weights=weights)
            self.non_uniformity = np.average(self.non_uniformity_list, weights=weights)
            self.norm_non_uniformity = np.average(self.norm_non_uniformity_list, weights=weights)
            self.length_non_uniformity = np.average(self.length_non_uniformity_list, weights=weights)
            self.norm_length_non_uniformity = np.average(self.norm_length_non_uniformity_list, weights=weights)
            self.percentage = np.average(self.percentage_list, weights=weights)
            self.gr_lvl_var = np.average(self.gr_lvl_var_list, weights=weights)
            self.length_var = np.average(self.length_var_list, weights=weights)
            self.entropy = np.average(self.entropy_list, weights=weights)

    def calc_2_5d_glszm_features(self):
        glszm = np.sum(self.glszm_2D_matrices, axis=0)
        self.short_runs_emphasis = self.calc_short_emphasis(glszm)
        self.long_runs_emphasis = self.calc_long_emphasis(glszm)
        self.low_grey_level_run_emphasis = self.calc_low_gr_lvl_emphasis(glszm)
        self.high_gr_lvl_emphasis = self.calc_high_gr_lvl_emphasis(glszm)
        self.short_low_gr_lvl_emphasis = self.calc_short_low_gr_lvl_emphasis(glszm)
        self.short_high_gr_lvl_emphasis = self.calc_short_high_gr_lvl_emphasis(glszm)
        self.long_low_gr_lvl_emphasis = self.calc_long_low_gr_lvl_emphasis(glszm)
        self.long_high_gr_lvl_emphasis = self.calc_long_high_gr_lvl_emphasis(glszm)
        self.non_uniformity = self.calc_non_uniformity(glszm)
        self.norm_non_uniformity = self.calc_norm_non_uniformity(glszm)
        self.length_non_uniformity = self.calc_length_non_uniformity(glszm)
        self.norm_length_non_uniformity = self.calc_norm_length_non_uniformity(glszm)
        self.percentage = self.calc_percentage(glszm, np.sum(self.no_of_roi_voxels))
        self.gr_lvl_var = self.calc_gr_lvl_var(glszm)
        self.length_var = self.calc_length_var(glszm)
        self.entropy = self.calc_entropy(glszm)

    def calc_3d_glszm_features(self):
        M = self.glszm_3D_matrix
        self.short_runs_emphasis = self.calc_short_emphasis(M)
        self.long_runs_emphasis = self.calc_long_emphasis(M)
        self.low_grey_level_run_emphasis = self.calc_low_gr_lvl_emphasis(M)
        self.high_gr_lvl_emphasis = self.calc_high_gr_lvl_emphasis(M)
        self.short_low_gr_lvl_emphasis = self.calc_short_low_gr_lvl_emphasis(M)
        self.short_high_gr_lvl_emphasis = self.calc_short_high_gr_lvl_emphasis(M)
        self.long_low_gr_lvl_emphasis = self.calc_long_low_gr_lvl_emphasis(M)
        self.long_high_gr_lvl_emphasis = self.calc_long_high_gr_lvl_emphasis(M)
        self.non_uniformity = self.calc_non_uniformity(M)
        self.norm_non_uniformity = self.calc_norm_non_uniformity(M)
        self.length_non_uniformity = self.calc_length_non_uniformity(M)
        self.norm_length_non_uniformity = self.calc_norm_length_non_uniformity(M)
        self.percentage = self.calc_percentage(M, self.tot_no_of_roi_voxels)
        self.gr_lvl_var = self.calc_gr_lvl_var(M)
        self.length_var = self.calc_length_var(M)
        self.entropy = self.calc_entropy(M)
