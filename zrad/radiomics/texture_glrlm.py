import numpy as np


class GLRLM:
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

    def rle_1d(self, arr, lvl, rlm):
        """
        Run‐length encode a 1D array of gray levels (with NaNs as breaks)
        and update the provided run‐length matrix.

        Parameters:
          arr: 1D numpy array (can contain np.nan)
          lvl: int, number of gray levels (assumed that valid pixel values are in 0..lvl-1)
          rlm: 2D numpy array of shape (lvl, max_length) to be updated in-place.
        """
        # Find indices of valid (non-NaN) entries.
        valid_idx = np.where(~np.isnan(arr))[0]
        if valid_idx.size == 0:
            return
        # Group valid indices into contiguous segments (i.e. ignoring gaps where NaN occurred)
        # A break occurs when the difference between consecutive valid indices is not 1.
        splits = np.where(np.diff(valid_idx) != 1)[0] + 1
        segments = np.split(valid_idx, splits)

        for seg in segments:
            # Get the values for the contiguous segment
            seg_vals = arr[seg]
            n = seg_vals.size
            if n == 0:
                continue
            # Find boundaries of runs within this contiguous segment.
            # A new run starts at index 0 or where the value changes.
            # Using np.diff, we locate indices where consecutive values differ.
            diff = np.diff(seg_vals)
            # run_breaks marks the indices where a new run starts (except the first element).
            run_breaks = np.where(diff != 0)[0] + 1

            # The start indices of runs are at 0 and at each run_break.
            run_starts = np.concatenate(([0], run_breaks))
            # The end indices are just before the next run or at the end.
            run_ends = np.concatenate((run_breaks, [n]))
            run_lengths = run_ends - run_starts

            # For each run, update the run-length matrix.
            for start, run_len in zip(run_starts, run_lengths):
                # Only process if the run length fits in our preallocated matrix.
                if run_len - 1 < rlm.shape[1]:
                    gray = int(seg_vals[start])
                    # (Assuming gray is between 0 and lvl-1)
                    rlm[gray, run_len - 1] += 1

    def process_horizontal(self, z_slice, lvl):
        """
        Process horizontal (0,1) direction: each row of the slice.
        """
        rows, cols = z_slice.shape
        rlm = np.zeros((lvl, max(rows, cols)), dtype=int)
        for i in range(rows):
            row = z_slice[i, :]
            self.rle_1d(row, lvl, rlm)
        return rlm

    def process_vertical(self, z_slice, lvl):
        """
        Process vertical (1,0) direction: each column of the slice.
        """
        rows, cols = z_slice.shape
        rlm = np.zeros((lvl, max(rows, cols)), dtype=int)
        for j in range(cols):
            col = z_slice[:, j]
            self.rle_1d(col, lvl, rlm)
        return rlm

    def process_diagonal(self, z_slice, lvl):
        """
        Process diagonal (1,1) direction: extract all diagonals of the slice.
        """
        rows, cols = z_slice.shape
        rlm = np.zeros((lvl, max(rows, cols)), dtype=int)
        # Diagonals: offsets from -(rows-1) to (cols-1)
        for offset in range(-rows + 1, cols):
            diag = np.diagonal(z_slice, offset=offset)
            self.rle_1d(diag, lvl, rlm)
        return rlm

    def process_antidiagonal(self, z_slice, lvl):
        """
        Process anti-diagonal (1,-1) direction: extract all anti-diagonals.
        """
        rows, cols = z_slice.shape
        rlm = np.zeros((lvl, max(rows, cols)), dtype=int)
        # Flip the slice left-right so that anti-diagonals become diagonals.
        flipped = np.fliplr(z_slice)
        for offset in range(-rows + 1, cols):
            diag = np.diagonal(flipped, offset=offset)
            self.rle_1d(diag, lvl, rlm)
        return rlm

    def calc_glrl_2d_matrices(self):
        """
        Computes the 2D Gray Level Run Length Matrix (GLRLM) for each slice in a 3D image.
        This optimized version projects each slice into 1D lines (for each direction)
        and performs run-length encoding on each line.
        """
        x, y, z = self.image.shape
        # Define the processing functions for each direction:
        # (0,1): horizontal, (1,0): vertical, (1,1): diagonal, (1,-1): anti-diagonal.
        direction_funcs = [
            self.process_horizontal,
            self.process_vertical,
            self.process_diagonal,
            self.process_antidiagonal,
        ]

        glrlm_2D_matrices = []
        no_of_roi_voxels = []

        # Process each slice in the z-dimension (using self.range_z)
        for z_slice_index in self.range_z:
            z_slice = self.image[:, :, z_slice_index]
            # Count valid (non-NaN) voxels.
            no_of_roi_voxels.append(np.count_nonzero(~np.isnan(z_slice)))

            slice_rlms = []
            # Process each direction.
            for func in direction_funcs:
                rlm = func(z_slice, self.lvl)
                slice_rlms.append(rlm)
            glrlm_2D_matrices.append(slice_rlms)

        # Store the results as attributes.
        self.glrlm_2D_matrices = np.array(glrlm_2D_matrices, dtype=np.int64)
        self.no_of_roi_voxels = no_of_roi_voxels

    def calc_glrl_3d_matrix(self):

        x, y, z = self.image.shape
        directions = np.array([
            (0, 0, 1), (0, 1, -1), (0, 1, 0),
            (0, 1, 1), (1, -1, -1), (1, -1, 0),
            (1, -1, 1), (1, 0, -1), (1, 0, 0),
            (1, 0, 1), (1, 1, -1), (1, 1, 0),
            (1, 1, 1)
        ])

        max_dim = max(x, y, z)
        self.glrlm_3D_matrix = np.zeros((len(directions), self.lvl, max_dim), dtype=np.int64)

        # Mask for NaN values
        nan_mask = np.isnan(self.image)

        # Iterate over directions (necessary to track full run lengths)
        for d_idx, (dx, dy, dz) in enumerate(directions):
            rlm = np.zeros((self.lvl, max_dim), dtype=np.int64)
            visited = np.zeros((x, y, z), dtype=bool)

            # Process all voxels in a vectorized manner
            valid_voxels = ~nan_mask
            i_idx, j_idx, k_idx = np.where(valid_voxels)

            for i, j, k in zip(i_idx, j_idx, k_idx):
                if visited[i, j, k]:
                    continue  # Skip already processed voxels

                gr_lvl = int(self.image[i, j, k])
                run_len = 1
                visited[i, j, k] = True

                # Follow the run in the given direction
                new_i, new_j, new_k = i + dx, j + dy, k + dz
                while (0 <= new_i < x and 0 <= new_j < y and 0 <= new_k < z and
                       self.image[new_i, new_j, new_k] == gr_lvl and
                       not visited[new_i, new_j, new_k] and
                       not nan_mask[new_i, new_j, new_k]):
                    visited[new_i, new_j, new_k] = True
                    run_len += 1
                    new_i += dx
                    new_j += dy
                    new_k += dz

                rlm[gr_lvl, run_len - 1] += 1  # Store run-length in RLM

            self.glrlm_3D_matrix[d_idx] = rlm

        self.glrlm_3D_matrix = self.glrlm_3D_matrix.astype(np.int64)

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

    def calc_2d_averaged_glrlm_features(self):

        number_of_slices = self.glrlm_2D_matrices.shape[0]
        number_of_directions = self.glrlm_2D_matrices.shape[1]
        weights = []

        for i in range(number_of_slices):
            for j in range(number_of_directions):
                M_ij = self.glrlm_2D_matrices[i][j]
                weight = 1
                if self.slice_weight:
                    weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
                weights.append(weight)

                self.short_runs_emphasis_list.append(self.calc_short_emphasis(M_ij))
                self.long_runs_emphasis_list.append(self.calc_long_emphasis(M_ij))
                self.low_grey_level_run_emphasis_list.append(self.calc_low_gr_lvl_emphasis(M_ij))
                self.high_gr_lvl_emphasis_list.append(self.calc_high_gr_lvl_emphasis(M_ij))
                self.short_low_gr_lvl_emphasis_list.append(self.calc_short_low_gr_lvl_emphasis(M_ij))
                self.short_high_gr_lvl_emphasis_list.append(self.calc_short_high_gr_lvl_emphasis(M_ij))
                self.long_low_gr_lvl_emphasis_list.append(self.calc_long_low_gr_lvl_emphasis(M_ij))
                self.long_high_gr_lvl_emphasis_list.append(self.calc_long_high_gr_lvl_emphasis(M_ij))
                self.non_uniformity_list.append(self.calc_non_uniformity(M_ij))
                self.norm_non_uniformity_list.append(self.calc_norm_non_uniformity(M_ij))
                self.length_non_uniformity_list.append(self.calc_length_non_uniformity(M_ij))
                self.norm_length_non_uniformity_list.append(self.calc_norm_length_non_uniformity(M_ij))
                self.percentage_list.append(self.calc_percentage(M_ij, self.no_of_roi_voxels[i]))
                self.gr_lvl_var_list.append(self.calc_gr_lvl_var(M_ij))
                self.length_var_list.append(self.calc_length_var(M_ij))
                self.entropy_list.append(self.calc_entropy(M_ij))

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

    def calc_2d_slice_merged_glrlm_features(self):

        number_of_slices = self.glrlm_2D_matrices.shape[0]
        number_of_directions = self.glrlm_2D_matrices.shape[1]
        averaged_M = np.sum(self.glrlm_2D_matrices, axis=1)  # / number_of_directions
        weights = []

        for i in range(number_of_slices):
            M_i = averaged_M[i]
            weight = 1
            if self.slice_weight:
                weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
            weights.append(weight)

            self.short_runs_emphasis_list.append(self.calc_short_emphasis(M_i))
            self.long_runs_emphasis_list.append(self.calc_long_emphasis(M_i))
            self.low_grey_level_run_emphasis_list.append(self.calc_low_gr_lvl_emphasis(M_i))
            self.high_gr_lvl_emphasis_list.append(self.calc_high_gr_lvl_emphasis(M_i))
            self.short_low_gr_lvl_emphasis_list.append(self.calc_short_low_gr_lvl_emphasis(M_i))
            self.short_high_gr_lvl_emphasis_list.append(self.calc_short_high_gr_lvl_emphasis(M_i))
            self.long_low_gr_lvl_emphasis_list.append(self.calc_long_low_gr_lvl_emphasis(M_i))
            self.long_high_gr_lvl_emphasis_list.append(self.calc_long_high_gr_lvl_emphasis(M_i))
            self.non_uniformity_list.append(self.calc_non_uniformity(M_i))
            self.norm_non_uniformity_list.append(self.calc_norm_non_uniformity(M_i))
            self.length_non_uniformity_list.append(self.calc_length_non_uniformity(M_i))
            self.norm_length_non_uniformity_list.append(self.calc_norm_length_non_uniformity(M_i))
            self.percentage_list.append(
                self.calc_percentage(M_i, self.no_of_roi_voxels[i]) * (1 / number_of_directions))
            self.gr_lvl_var_list.append(self.calc_gr_lvl_var(M_i))
            self.length_var_list.append(self.calc_length_var(M_i))
            self.entropy_list.append(self.calc_entropy(M_i))

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

    def calc_2_5d_merged_glrlm_features(self):

        number_of_directions = self.glrlm_2D_matrices.shape[1]
        glrlm = np.sum(np.sum(self.glrlm_2D_matrices, axis=1), axis=0)

        self.short_runs_emphasis = self.calc_short_emphasis(glrlm)
        self.long_runs_emphasis = self.calc_long_emphasis(glrlm)
        self.low_grey_level_run_emphasis = self.calc_low_gr_lvl_emphasis(glrlm)
        self.high_gr_lvl_emphasis = self.calc_high_gr_lvl_emphasis(glrlm)
        self.short_low_gr_lvl_emphasis = self.calc_short_low_gr_lvl_emphasis(glrlm)
        self.short_high_gr_lvl_emphasis = self.calc_short_high_gr_lvl_emphasis(glrlm)
        self.long_low_gr_lvl_emphasis = self.calc_long_low_gr_lvl_emphasis(glrlm)
        self.long_high_gr_lvl_emphasis = self.calc_long_high_gr_lvl_emphasis(glrlm)
        self.non_uniformity = self.calc_non_uniformity(glrlm)
        self.norm_non_uniformity = self.calc_norm_non_uniformity(glrlm)
        self.length_non_uniformity = self.calc_length_non_uniformity(glrlm)
        self.norm_length_non_uniformity = self.calc_norm_length_non_uniformity(glrlm)
        self.percentage = self.calc_percentage(glrlm, np.sum(self.no_of_roi_voxels)) / number_of_directions
        self.gr_lvl_var = self.calc_gr_lvl_var(glrlm)
        self.length_var = self.calc_length_var(glrlm)
        self.entropy = self.calc_entropy(glrlm)

    def calc_2_5d_direction_merged_glrlm_features(self):

        number_of_directions = self.glrlm_2D_matrices.shape[1]
        averaged_glrlm = np.sum(self.glrlm_2D_matrices, axis=0)

        for i in range(number_of_directions):
            glrlm_i = averaged_glrlm[i]
            self.short_runs_emphasis += self.calc_short_emphasis(glrlm_i)
            self.long_runs_emphasis += self.calc_long_emphasis(glrlm_i)
            self.low_grey_level_run_emphasis += self.calc_low_gr_lvl_emphasis(glrlm_i)
            self.high_gr_lvl_emphasis += self.calc_high_gr_lvl_emphasis(glrlm_i)
            self.short_low_gr_lvl_emphasis += self.calc_short_low_gr_lvl_emphasis(glrlm_i)
            self.short_high_gr_lvl_emphasis += self.calc_short_high_gr_lvl_emphasis(glrlm_i)
            self.long_low_gr_lvl_emphasis += self.calc_long_low_gr_lvl_emphasis(glrlm_i)
            self.long_high_gr_lvl_emphasis += self.calc_long_high_gr_lvl_emphasis(glrlm_i)
            self.non_uniformity += self.calc_non_uniformity(glrlm_i)
            self.norm_non_uniformity += self.calc_norm_non_uniformity(glrlm_i)
            self.length_non_uniformity += self.calc_length_non_uniformity(glrlm_i)
            self.norm_length_non_uniformity += self.calc_norm_length_non_uniformity(glrlm_i)
            self.percentage += self.calc_percentage(glrlm_i, np.sum(self.no_of_roi_voxels))
            self.gr_lvl_var += self.calc_gr_lvl_var(glrlm_i)
            self.length_var += self.calc_length_var(glrlm_i)
            self.entropy += self.calc_entropy(glrlm_i)

        self.short_runs_emphasis /= number_of_directions
        self.long_runs_emphasis /= number_of_directions
        self.low_grey_level_run_emphasis /= number_of_directions
        self.high_gr_lvl_emphasis /= number_of_directions
        self.short_low_gr_lvl_emphasis /= number_of_directions
        self.short_high_gr_lvl_emphasis /= number_of_directions
        self.long_low_gr_lvl_emphasis /= number_of_directions
        self.long_high_gr_lvl_emphasis /= number_of_directions
        self.non_uniformity /= number_of_directions
        self.norm_non_uniformity /= number_of_directions
        self.length_non_uniformity /= number_of_directions
        self.norm_length_non_uniformity /= number_of_directions
        self.percentage /= number_of_directions
        self.gr_lvl_var /= number_of_directions
        self.length_var /= number_of_directions
        self.entropy /= number_of_directions

    def calc_3d_averaged_glrlm_features(self):

        number_of_directions = self.glrlm_3D_matrix.shape[0]
        for i in range(number_of_directions):
            M_i = self.glrlm_3D_matrix[i]

            self.short_runs_emphasis += self.calc_short_emphasis(M_i)
            self.long_runs_emphasis += self.calc_long_emphasis(M_i)
            self.low_grey_level_run_emphasis += self.calc_low_gr_lvl_emphasis(M_i)
            self.high_gr_lvl_emphasis += self.calc_high_gr_lvl_emphasis(M_i)
            self.short_low_gr_lvl_emphasis += self.calc_short_low_gr_lvl_emphasis(M_i)
            self.short_high_gr_lvl_emphasis += self.calc_short_high_gr_lvl_emphasis(M_i)
            self.long_low_gr_lvl_emphasis += self.calc_long_low_gr_lvl_emphasis(M_i)
            self.long_high_gr_lvl_emphasis += self.calc_long_high_gr_lvl_emphasis(M_i)
            self.non_uniformity += self.calc_non_uniformity(M_i)
            self.norm_non_uniformity += self.calc_norm_non_uniformity(M_i)
            self.length_non_uniformity += self.calc_length_non_uniformity(M_i)
            self.norm_length_non_uniformity += self.calc_norm_length_non_uniformity(M_i)
            self.percentage += self.calc_percentage(M_i, self.tot_no_of_roi_voxels)
            self.gr_lvl_var += self.calc_gr_lvl_var(M_i)
            self.length_var += self.calc_length_var(M_i)
            self.entropy += self.calc_entropy(M_i)

        self.short_runs_emphasis /= number_of_directions
        self.long_runs_emphasis /= number_of_directions
        self.low_grey_level_run_emphasis /= number_of_directions
        self.high_gr_lvl_emphasis /= number_of_directions
        self.short_low_gr_lvl_emphasis /= number_of_directions
        self.short_high_gr_lvl_emphasis /= number_of_directions
        self.long_low_gr_lvl_emphasis /= number_of_directions
        self.long_high_gr_lvl_emphasis /= number_of_directions
        self.non_uniformity /= number_of_directions
        self.norm_non_uniformity /= number_of_directions
        self.length_non_uniformity /= number_of_directions
        self.norm_length_non_uniformity /= number_of_directions
        self.percentage /= number_of_directions
        self.gr_lvl_var /= number_of_directions
        self.length_var /= number_of_directions
        self.entropy /= number_of_directions

    def calc_3d_merged_glrlm_features(self):

        number_of_directions = self.glrlm_3D_matrix.shape[0]
        M = np.sum(self.glrlm_3D_matrix, axis=0)

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
        self.percentage = self.calc_percentage(M, self.tot_no_of_roi_voxels) / number_of_directions
        self.gr_lvl_var = self.calc_gr_lvl_var(M)
        self.length_var = self.calc_length_var(M)
        self.entropy = self.calc_entropy(M)
