import numpy as np
from scipy.ndimage import convolve

from .texture_base import TextureFeatureBase


class NGLDM(TextureFeatureBase):
    FEATURE_NAMES = [
        "short_runs_emphasis",
        "long_runs_emphasis",
        "low_grey_level_run_emphasis",
        "high_gr_lvl_emphasis",
        "short_low_gr_lvl_emphasis",
        "short_high_gr_lvl_emphasis",
        "long_low_gr_lvl_emphasis",
        "long_high_gr_lvl_emphasis",
        "non_uniformity",
        "norm_non_uniformity",
        "length_non_uniformity",
        "norm_length_non_uniformity",
        "percentage",
        "gr_lvl_var",
        "length_var",
        "entropy",
        "energy",
    ]

    def __init__(self, image, slice_weight=False, slice_median=False):

        super().__init__(image, self.FEATURE_NAMES, slice_weight, slice_median)
        self.lvl = int(np.nanmax(self.image) + 1)
        self.tot_no_of_roi_voxels = np.sum(~np.isnan(image))

        x_indices, y_indices, z_indices = np.where(~np.isnan(self.image))

        self.range_x = np.unique(x_indices)
        self.range_y = np.unique(y_indices)
        self.range_z = np.unique(z_indices)


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

    def _calc_features_from_matrix(self, M, roi_voxels):
        return {
            "short_runs_emphasis": self.calc_short_emphasis(M),
            "long_runs_emphasis": self.calc_long_emphasis(M),
            "low_grey_level_run_emphasis": self.calc_low_gr_lvl_emphasis(M),
            "high_gr_lvl_emphasis": self.calc_high_gr_lvl_emphasis(M),
            "short_low_gr_lvl_emphasis": self.calc_short_low_gr_lvl_emphasis(M),
            "short_high_gr_lvl_emphasis": self.calc_short_high_gr_lvl_emphasis(M),
            "long_low_gr_lvl_emphasis": self.calc_long_low_gr_lvl_emphasis(M),
            "long_high_gr_lvl_emphasis": self.calc_long_high_gr_lvl_emphasis(M),
            "non_uniformity": self.calc_non_uniformity(M),
            "norm_non_uniformity": self.calc_norm_non_uniformity(M),
            "length_non_uniformity": self.calc_length_non_uniformity(M),
            "norm_length_non_uniformity": self.calc_norm_length_non_uniformity(M),
            "percentage": self.calc_percentage(M, roi_voxels),
            "gr_lvl_var": self.calc_gr_lvl_var(M),
            "length_var": self.calc_length_var(M),
            "entropy": self.calc_entropy(M),
            "energy": self.calc_energy(M),
        }

    def calc_ngld_3d_matrix(self):
        x, y, z = self.image.shape
        ngldm = np.zeros((self.lvl, 27), dtype=np.int64)
        valid_mask = ~np.isnan(self.image)

        # Use a 3x3x3 kernel with the central voxel excluded.
        kernel = np.ones((3, 3, 3), dtype=int)
        kernel[1, 1, 1] = 0

        # Instead of iterating over each voxel, convolve the binary mask for each intensity.
        for lvl in range(self.lvl):
            M = ((self.image == lvl) & valid_mask).astype(np.int64)
            if np.sum(M) == 0:
                continue
            # Convolve to count how many neighbors (of the 26 possible) have the same level.
            neighbor_counts = convolve(M, kernel, mode='constant', cval=0)
            # Select the counts only for voxels that are part of the current level.
            counts = neighbor_counts[M.astype(bool)]
            if counts.size:
                bincounts = np.bincount(counts, minlength=27)
                ngldm[lvl, :len(bincounts)] += bincounts

        self.ngldm_3D_matrix = ngldm

    def calc_ngld_2d_matrices(self):
        """
        Computes the 2D Neighboring Gray Level Dependence Matrix (NGLDM) for each slice
        in a 3D image.
        """
        self.ngldm_2d_matrices = []
        self.no_of_roi_voxels = []

        # Offsets for 8-connectivity (neighbors excluding the center)
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1), (0, 1),
                   (1, -1), (1, 0), (1, 1)]

        def calc_ngldm_slice(array):
            """
            Computes the NGLDM for a single 2D slice.

            For each valid (non-NaN) pixel in the slice, count how many of its 8 neighbors
            have the same intensity. The results are accumulated in a matrix of shape
            (self.lvl, 9), where the row index corresponds to the intensity value (assumed to be
            in 0..self.lvl-1) and the column index is the neighbor count (0..8).
            """
            # Pad the array with NaN so that comparisons with out‐of‐bounds areas always fail.
            padded = np.pad(array, pad_width=1, mode='constant', constant_values=np.nan)
            # "Center" corresponds to the original image (without padding)
            center = padded[1:-1, 1:-1]
            # Preallocate the neighbor count array (same shape as center)
            neighbor_count = np.zeros_like(center, dtype=int)

            # For each of the 8 neighbors, extract the appropriately shifted sub–array
            # and add 1 where the neighbor equals the center pixel.
            for dx, dy in offsets:
                neighbor = padded[1 + dx: 1 + dx + center.shape[0],
                           1 + dy: 1 + dy + center.shape[1]]
                # (neighbor == center) returns False when either is NaN.
                neighbor_count += (neighbor == center)

            # Allocate the output matrix.
            # Each row corresponds to an intensity (0..self.lvl-1) and each column
            # (0..8) holds the number of pixels having that many matching neighbors.
            ngldm = np.zeros((self.lvl, 9), dtype=int)

            # Only consider valid (non-NaN) center pixels.
            valid = ~np.isnan(center)
            intensities = center[valid].astype(int)  # Intensity for each valid pixel.
            counts = neighbor_count[valid]  # Corresponding neighbor counts.

            # For each valid pixel, add 1 to the appropriate bin in the ngldm matrix.
            # This is done in a fully vectorized way.
            np.add.at(ngldm, (intensities, counts), 1)
            return ngldm

        # Process each 2D slice in the 3D image (assumed stored in self.image)
        for z in range(self.image.shape[2]):
            slice_ = self.image[:, :, z]
            if np.any(~np.isnan(slice_)):  # Process only if the slice contains valid data.
                self.no_of_roi_voxels.append(np.count_nonzero(~np.isnan(slice_)))
                self.ngldm_2d_matrices.append(calc_ngldm_slice(slice_))

        # Convert the list of matrices to a single NumPy array.
        self.ngldm_2d_matrices = np.array(self.ngldm_2d_matrices, dtype=np.int64)

    def calc_2d_ngldm_features(self):
        self._reset_feature_lists()

        number_of_slices = self.ngldm_2d_matrices.shape[0]
        weights = []

        for i in range(number_of_slices):
            ngldm_matrix = self.ngldm_2d_matrices[i]
            weight = 1
            if self.slice_weight:
                weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
            weights.append(weight)

            feats = self._calc_features_from_matrix(ngldm_matrix, self.no_of_roi_voxels[i])
            self._append_features(feats)

        self._finalize_features(weights)

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


