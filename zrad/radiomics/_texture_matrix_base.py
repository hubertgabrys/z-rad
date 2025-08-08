import numpy as np
from scipy.ndimage import distance_transform_cdt, label, minimum


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


class ZoneMatrixBase(TextureMatrixBase):
    """Base class for GLSZM and GLDZM feature extractors.

    Provides utilities to compute gray level size zone and distance zone matrices
    in both 2D and 3D."""

    def calc_glsz_gldz_3d_matrices(self, mask):
        flattened_array = self.image.flatten()
        _, counts = np.unique(flattened_array[~np.isnan(flattened_array)], return_counts=True)
        max_region_size = np.max(counts)

        def calc_dist_map_3d(image_orig):
            image = image_orig.copy()
            larger_array = np.zeros((image.shape[0] + 2, image.shape[1] + 2, image.shape[2] + 2))
            larger_array[1:-1, 1:-1, 1:-1] = image
            distance_map = distance_transform_cdt(larger_array, metric='taxicab')[1:-1, 1:-1, 1:-1].astype(float)
            return distance_map

        dist_map = calc_dist_map_3d(mask)
        glszm = np.zeros((self.lvl, max_region_size), dtype=int)
        gldzm = np.zeros((self.lvl, np.max(self.image.shape)), dtype=int)

        def find_connected_region_3d(start, intensity):
            stack = [start]
            size = 0
            min_dist = np.inf
            x_max, y_max, z_max = self.image.shape

            while stack:
                x, y, z = stack.pop()
                if 0 <= x < x_max and 0 <= y < y_max and 0 <= z < z_max:
                    if visited[x, y, z] == 0 and self.image[x, y, z] == intensity:
                        visited[x, y, z] = 1
                        size += 1
                        min_dist = min(min_dist, dist_map[x, y, z])
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                for dz in [-1, 0, 1]:
                                    if dx == dy == dz == 0:
                                        continue
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
                            glszm[intensity, size - 1] += 1
                            gldzm[intensity, int(min_dist) - 1] += 1

        self.glszm_3D_matrix = glszm.astype(np.int64)
        self.gldzm_3D_matrix = gldzm.astype(np.int64)

    def calc_glsz_gldz_2d_matrices(self, mask):
        max_region_size_list = []
        for z_idx in self.range_z:
            z_slice = self.image[:, :, z_idx]
            valid = ~np.isnan(z_slice)
            if np.any(valid):
                counts = np.bincount(z_slice[valid].astype(int))
                if counts.size:
                    max_region_size_list.append(counts.max())
        max_region_size = max(max_region_size_list) if max_region_size_list else 0

        def calc_dist_map_2d(image_orig):
            larger_array = np.zeros((image_orig.shape[0] + 2, image_orig.shape[1] + 2))
            larger_array[1:-1, 1:-1] = image_orig
            distance_map = distance_transform_cdt(larger_array, metric='taxicab')[1:-1, 1:-1].astype(float)
            return distance_map

        glszm_matrices = []
        gldzm_matrices = []
        roi_voxels = []
        structure = np.ones((3, 3), dtype=int)

        for z_idx in self.range_z:
            z_slice = self.image[:, :, z_idx]
            z_mask = mask[:, :, z_idx]
            roi_voxels.append(np.sum(~np.isnan(z_slice)))
            dist_map = calc_dist_map_2d(z_mask)
            glszm = np.zeros((self.lvl, max_region_size), dtype=int)
            gldzm = np.zeros((self.lvl, np.max(self.image.shape)), dtype=int)

            for intensity in range(self.lvl):
                comp_mask = (z_slice == intensity)
                if not np.any(comp_mask):
                    continue
                labeled, num_features = label(comp_mask, structure=structure)
                if num_features == 0:
                    continue
                sizes = np.bincount(labeled.ravel())[1:]
                min_dists = minimum(dist_map, labeled, index=np.arange(1, num_features + 1))
                unique_sizes, counts_sizes = np.unique(sizes, return_counts=True)
                for s, count in zip(unique_sizes, counts_sizes):
                    if s - 1 < glszm.shape[1]:
                        glszm[intensity, s - 1] += count
                min_dists_int = min_dists.astype(int)
                unique_dists, counts_dists = np.unique(min_dists_int, return_counts=True)
                for d, count in zip(unique_dists, counts_dists):
                    if d - 1 < gldzm.shape[1]:
                        gldzm[intensity, d - 1] += count

            glszm_matrices.append(glszm.astype(np.int64))
            gldzm_matrices.append(gldzm.astype(np.int64))

        self.glszm_2D_matrices = np.array(glszm_matrices)
        self.gldzm_2D_matrices = np.array(gldzm_matrices)
        self.no_of_roi_voxels = roi_voxels
