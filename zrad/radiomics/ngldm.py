import numpy as np
from scipy.ndimage import convolve

from ._texture_matrix_base import TextureMatrixBase


class NGLDM(TextureMatrixBase):
    """Neighboring Gray Level Dependence Matrix (NGLDM) feature extractor."""

    def calc_ngld_3d_matrix(self):
        x, y, z = self.image.shape
        ngldm = np.zeros((self.lvl, 27), dtype=np.int64)
        offsets = np.array([
            (dx, dy, dz) for dx in range(-1, 2)
            for dy in range(-1, 2)
            for dz in range(-1, 2)
            if not (dx == dy == dz == 0)
        ])
        valid_mask = ~np.isnan(self.image)
        for lvl in range(self.lvl):
            mask = (self.image == lvl) & valid_mask
            lvl_i, lvl_j, lvl_k = np.where(mask)
            if lvl_i.size == 0:
                continue
            neighbor_i = lvl_i[:, None] + offsets[:, 0]
            neighbor_j = lvl_j[:, None] + offsets[:, 1]
            neighbor_k = lvl_k[:, None] + offsets[:, 2]
            valid_neighbors = (
                (0 <= neighbor_i) & (neighbor_i < x) &
                (0 <= neighbor_j) & (neighbor_j < y) &
                (0 <= neighbor_k) & (neighbor_k < z)
            )
            neighbor_i = neighbor_i[valid_neighbors]
            neighbor_j = neighbor_j[valid_neighbors]
            neighbor_k = neighbor_k[valid_neighbors]
            valid_neighbor_mask = mask[neighbor_i, neighbor_j, neighbor_k]
            neighbor_counts = np.sum(valid_neighbor_mask.reshape(len(lvl_i), -1), axis=1)
            if neighbor_counts.size > 0:
                j_k = np.bincount(neighbor_counts, minlength=27)
                ngldm[lvl, :len(j_k)] += j_k
        self.ngldm_3D_matrix = ngldm

    def calc_ngld_2d_matrices(self):
        self.ngldm_2d_matrices = []
        self.no_of_roi_voxels = []
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1), (0, 1),
                   (1, -1), (1, 0), (1, 1)]

        def calc_ngldm_slice(array):
            padded = np.pad(array, pad_width=1, mode='constant', constant_values=np.nan)
            center = padded[1:-1, 1:-1]
            neighbor_count = np.zeros_like(center, dtype=int)
            for dx, dy in offsets:
                neighbor = padded[1 + dx: 1 + dx + center.shape[0],
                                   1 + dy: 1 + dy + center.shape[1]]
                neighbor_count += (neighbor == center)
            ngldm = np.zeros((self.lvl, 9), dtype=int)
            valid = ~np.isnan(center)
            intensities = center[valid].astype(int)
            counts = neighbor_count[valid]
            np.add.at(ngldm, (intensities, counts), 1)
            return ngldm

        for z in range(self.image.shape[2]):
            slice_ = self.image[:, :, z]
            if np.any(~np.isnan(slice_)):
                self.no_of_roi_voxels.append(np.count_nonzero(~np.isnan(slice_)))
                self.ngldm_2d_matrices.append(calc_ngldm_slice(slice_))
        self.ngldm_2d_matrices = np.array(self.ngldm_2d_matrices)

    def calc_2d_ngldm_features(self):
        number_of_slices = self.ngldm_2d_matrices.shape[0]
        weights = []
        for i in range(number_of_slices):
            ngldm_matrix = self.ngldm_2d_matrices[i]
            weight = 1
            if self.slice_weight:
                weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
            weights.append(weight)
            self.short_runs_emphasis_list.append(self.calc_short_emphasis(ngldm_matrix))
            self.long_runs_emphasis_list.append(self.calc_long_emphasis(ngldm_matrix))
            self.low_grey_level_run_emphasis_list.append(self.calc_low_gr_lvl_emphasis(ngldm_matrix))
            self.high_gr_lvl_emphasis_list.append(self.calc_high_gr_lvl_emphasis(ngldm_matrix))
            self.short_low_gr_lvl_emphasis_list.append(self.calc_short_low_gr_lvl_emphasis(ngldm_matrix))
            self.short_high_gr_lvl_emphasis_list.append(self.calc_short_high_gr_lvl_emphasis(ngldm_matrix))
            self.long_low_gr_lvl_emphasis_list.append(self.calc_long_low_gr_lvl_emphasis(ngldm_matrix))
            self.long_high_gr_lvl_emphasis_list.append(self.calc_long_high_gr_lvl_emphasis(ngldm_matrix))
            self.non_uniformity_list.append(self.calc_non_uniformity(ngldm_matrix))
            self.norm_non_uniformity_list.append(self.calc_norm_non_uniformity(ngldm_matrix))
            self.length_non_uniformity_list.append(self.calc_length_non_uniformity(ngldm_matrix))
            self.norm_length_non_uniformity_list.append(self.calc_norm_length_non_uniformity(ngldm_matrix))
            self.percentage_list.append(self.calc_percentage(ngldm_matrix, self.no_of_roi_voxels[i]))
            self.gr_lvl_var_list.append(self.calc_gr_lvl_var(ngldm_matrix))
            self.length_var_list.append(self.calc_length_var(ngldm_matrix))
            self.entropy_list.append(self.calc_entropy(ngldm_matrix))
            self.energy_list.append(self.calc_energy(ngldm_matrix))
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
            self.energy = np.median(self.energy_list)
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
            self.energy = np.average(self.energy_list, weights=weights)

    def calc_2_5d_ngldm_features(self):
        ngld_matrix = np.sum(self.ngldm_2d_matrices, axis=0)
        self.short_runs_emphasis = self.calc_short_emphasis(ngld_matrix)
        self.long_runs_emphasis = self.calc_long_emphasis(ngld_matrix)
        self.low_grey_level_run_emphasis = self.calc_low_gr_lvl_emphasis(ngld_matrix)
        self.high_gr_lvl_emphasis = self.calc_high_gr_lvl_emphasis(ngld_matrix)
        self.short_low_gr_lvl_emphasis = self.calc_short_low_gr_lvl_emphasis(ngld_matrix)
        self.short_high_gr_lvl_emphasis = self.calc_short_high_gr_lvl_emphasis(ngld_matrix)
        self.long_low_gr_lvl_emphasis = self.calc_long_low_gr_lvl_emphasis(ngld_matrix)
        self.long_high_gr_lvl_emphasis = self.calc_long_high_gr_lvl_emphasis(ngld_matrix)
        self.non_uniformity = self.calc_non_uniformity(ngld_matrix)
        self.norm_non_uniformity = self.calc_norm_non_uniformity(ngld_matrix)
        self.length_non_uniformity = self.calc_length_non_uniformity(ngld_matrix)
        self.norm_length_non_uniformity = self.calc_norm_length_non_uniformity(ngld_matrix)
        self.percentage = self.calc_percentage(ngld_matrix, np.sum(self.no_of_roi_voxels))
        self.gr_lvl_var = self.calc_gr_lvl_var(ngld_matrix)
        self.length_var = self.calc_length_var(ngld_matrix)
        self.entropy = self.calc_entropy(ngld_matrix)
        self.energy = self.calc_energy(ngld_matrix)

    def calc_3d_ngldm_features(self):
        ngldm = self.ngldm_3D_matrix
        self.short_runs_emphasis = self.calc_short_emphasis(ngldm)
        self.long_runs_emphasis = self.calc_long_emphasis(ngldm)
        self.low_grey_level_run_emphasis = self.calc_low_gr_lvl_emphasis(ngldm)
        self.high_gr_lvl_emphasis = self.calc_high_gr_lvl_emphasis(ngldm)
        self.short_low_gr_lvl_emphasis = self.calc_short_low_gr_lvl_emphasis(ngldm)
        self.short_high_gr_lvl_emphasis = self.calc_short_high_gr_lvl_emphasis(ngldm)
        self.long_low_gr_lvl_emphasis = self.calc_long_low_gr_lvl_emphasis(ngldm)
        self.long_high_gr_lvl_emphasis = self.calc_long_high_gr_lvl_emphasis(ngldm)
        self.non_uniformity = self.calc_non_uniformity(ngldm)
        self.norm_non_uniformity = self.calc_norm_non_uniformity(ngldm)
        self.length_non_uniformity = self.calc_length_non_uniformity(ngldm)
        self.norm_length_non_uniformity = self.calc_norm_length_non_uniformity(ngldm)
        self.percentage = self.calc_percentage(ngldm, self.tot_no_of_roi_voxels)
        self.gr_lvl_var = self.calc_gr_lvl_var(ngldm)
        self.length_var = self.calc_length_var(ngldm)
        self.entropy = self.calc_entropy(ngldm)
        self.energy = self.calc_energy(ngldm)
