import numpy as np
from scipy.ndimage import distance_transform_cdt, label, minimum

from .texture_base import TextureFeatureBase


class GLDZM(TextureFeatureBase):
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

    def calc_gldz_3d_matrices(self, mask):

        flattened_array = self.image.flatten()
        _, counts = np.unique(flattened_array[~np.isnan(flattened_array)], return_counts=True)
        max_region_size = np.max(counts)

        def calc_dist_map_3d(image_orig):
            image = image_orig.copy()
            # image[np.isnan(image)] = 0
            larger_array = np.zeros((image.shape[0] + 2, image.shape[1] + 2, image.shape[2] + 2))
            larger_array[1:-1, 1:-1, 1:-1] = image
            distance_map = distance_transform_cdt(larger_array, metric='taxicab')[1:-1, 1:-1, 1:-1].astype(float)

            return distance_map

        dist_map = calc_dist_map_3d(mask)
        gldzm = np.zeros((self.lvl, np.max(self.image.shape)), dtype=int)

        def find_connected_region_3d(start, intensity):
            stack = [start]
            size = 0
            min_dist = np.inf
            x_max, y_max, z_max = self.image.shape

            while stack:
                x, y, z = stack.pop()

                # Boundary check before accessing array elements
                if 0 <= x < x_max and 0 <= y < y_max and 0 <= z < z_max:
                    if visited[x, y, z] == 0 and self.image[x, y, z] == intensity:
                        visited[x, y, z] = 1
                        size += 1
                        min_dist = min(min_dist, dist_map[x, y, z])

                        # Add valid neighbors (26-connectivity)
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                for dz in [-1, 0, 1]:
                                    if dx == 0 and dy == 0 and dz == 0:
                                        continue  # Skip self
                                    nx, ny, nz = x + dx, y + dy, z + dz
                                    if 0 <= nx < x_max and 0 <= ny < y_max and 0 <= nz < z_max:
                                        if visited[nx, ny, nz] == 0 and self.image[nx, ny, nz] == intensity:
                                            stack.append((nx, ny, nz))

            return size, min_dist

        visited = np.zeros_like(self.image, dtype=int)
        for x in self.range_x:
            for y in self.range_y:
                for z in self.range_z:
                    if visited[x, y, z] == 0 and not np.isnan(self.image[x, y, z]):
                        intensity = int(self.image[x, y, z])
                        size, min_dist = find_connected_region_3d((x, y, z), intensity)
                        if size > 0:
                            gldzm[intensity, int(min_dist) - 1] += 1

        self.gldzm_3D_matrix = gldzm.astype(np.int64)

    def calc_gldz_2d_matrices(self, mask):
        # Precompute a maximum region size based on overall intensity counts per slice.
        max_region_size_list = []
        for z_idx in self.range_z:
            z_slice = self.image[:, :, z_idx]
            valid = ~np.isnan(z_slice)
            if np.any(valid):
                # Count occurrences for each intensity (assumed integer)
                counts = np.bincount(z_slice[valid].astype(int))
                if counts.size:
                    max_region_size_list.append(counts.max())
        max_region_size = max(max_region_size_list) if max_region_size_list else 0

        def calc_dist_map_2d(image_orig):
            # Create a border (padding) around the mask before applying the taxicab distance transform.
            larger_array = np.zeros((image_orig.shape[0] + 2, image_orig.shape[1] + 2))
            larger_array[1:-1, 1:-1] = image_orig
            # Compute the taxicab (city-block) distance transform and remove the border.
            distance_map = distance_transform_cdt(larger_array, metric='taxicab')[1:-1, 1:-1].astype(float)
            return distance_map

        gldzm_matrices = []
        roi_voxels = []
        # Define an 8-connected structure (3x3 array of ones)
        structure = np.ones((3, 3), dtype=int)

        for z_idx in self.range_z:
            z_slice = self.image[:, :, z_idx]
            z_mask = mask[:, :, z_idx]
            roi_voxels.append(np.sum(~np.isnan(z_slice)))
            dist_map = calc_dist_map_2d(z_mask)
            # Allocate matrices: note that the number of columns is based on the maximum possible region size
            glszm = np.zeros((self.lvl, max_region_size), dtype=int)
            gldzm = np.zeros((self.lvl, np.max(self.image.shape)), dtype=int)

            # Process each intensity separately using connected component labeling.
            for intensity in range(self.lvl):
                comp_mask = (z_slice == intensity)
                if not np.any(comp_mask):
                    continue
                # Label connected components (8-connectivity)
                labeled, num_features = label(comp_mask, structure=structure)
                if num_features == 0:
                    continue

                # Region sizes (skip the background label 0)
                sizes = np.bincount(labeled.ravel())[1:]
                # Compute the minimum distance within each connected region using the precomputed distance map.
                min_dists = minimum(dist_map, labeled, index=np.arange(1, num_features + 1))

                # Update the gray level size zone matrix (GLSZM)
                unique_sizes, counts_sizes = np.unique(sizes, return_counts=True)
                for s, count in zip(unique_sizes, counts_sizes):
                    if s - 1 < glszm.shape[1]:
                        glszm[intensity, s - 1] += count

                # Update the gray level distance zone matrix (GLDZM)
                # (Convert min distances to int – they are taxicab distances)
                min_dists_int = min_dists.astype(int)
                unique_dists, counts_dists = np.unique(min_dists_int, return_counts=True)
                for d, count in zip(unique_dists, counts_dists):
                    if d - 1 < gldzm.shape[1]:
                        gldzm[intensity, d - 1] += count

            gldzm_matrices.append(gldzm.astype(np.int64))

        self.gldzm_2D_matrices = np.array(gldzm_matrices)
        self.no_of_roi_voxels = roi_voxels

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

    def calc_2d_gldzm_features(self):
        self._reset_feature_lists()

        number_of_slices = self.gldzm_2D_matrices.shape[0]
        weights = []

        for i in range(number_of_slices):
            M = self.gldzm_2D_matrices[i]
            weight = 1
            if self.slice_weight:
                weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
            weights.append(weight)

            feats = self._calc_features_from_matrix(M, self.no_of_roi_voxels[i])
            self._append_features(feats)

        self._finalize_features(weights)

    def calc_2_5d_gldzm_features(self):

        M = np.sum(self.gldzm_2D_matrices, axis=0)

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
        self.percentage = self.calc_percentage(M, np.sum(self.no_of_roi_voxels))
        self.gr_lvl_var = self.calc_gr_lvl_var(M)
        self.length_var = self.calc_length_var(M)
        self.entropy = self.calc_entropy(M)

    def calc_3d_gldzm_features(self):

        ngdzm = self.gldzm_3D_matrix.astype(np.int64)

        self.short_runs_emphasis = self.calc_short_emphasis(ngdzm)
        self.long_runs_emphasis = self.calc_long_emphasis(ngdzm)
        self.low_grey_level_run_emphasis = self.calc_low_gr_lvl_emphasis(ngdzm)
        self.high_gr_lvl_emphasis = self.calc_high_gr_lvl_emphasis(ngdzm)
        self.short_low_gr_lvl_emphasis = self.calc_short_low_gr_lvl_emphasis(ngdzm)
        self.short_high_gr_lvl_emphasis = self.calc_short_high_gr_lvl_emphasis(ngdzm)
        self.long_low_gr_lvl_emphasis = self.calc_long_low_gr_lvl_emphasis(ngdzm)
        self.long_high_gr_lvl_emphasis = self.calc_long_high_gr_lvl_emphasis(ngdzm)
        self.non_uniformity = self.calc_non_uniformity(ngdzm)
        self.norm_non_uniformity = self.calc_norm_non_uniformity(ngdzm)
        self.length_non_uniformity = self.calc_length_non_uniformity(ngdzm)
        self.norm_length_non_uniformity = self.calc_norm_length_non_uniformity(ngdzm)
        self.percentage = self.calc_percentage(ngdzm, self.tot_no_of_roi_voxels)
        self.gr_lvl_var = self.calc_gr_lvl_var(ngdzm)
        self.length_var = self.calc_length_var(ngdzm)
        self.entropy = self.calc_entropy(ngdzm)
