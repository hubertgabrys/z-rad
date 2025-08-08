import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_cdt, label, generate_binary_structure, minimum
from scipy.ndimage.morphology import generate_binary_structure
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from scipy.special import legendre
from scipy.stats import iqr, skew, kurtosis
from skimage import measure
from sklearn.decomposition import PCA


class GLRLM_GLSZM_GLDZM_NGLDM:
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

    def calc_glsz_gldz_3d_matrices(self, mask):

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
        glszm = np.zeros((self.lvl, max_region_size), dtype=int)
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
                            glszm[intensity, size - 1] += 1
                            gldzm[intensity, int(min_dist) - 1] += 1

        self.glszm_3D_matrix = glszm.astype(np.int64)
        self.gldzm_3D_matrix = gldzm.astype(np.int64)

    def calc_glsz_gldz_2d_matrices(self, mask):
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

        glszm_matrices = []
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

            glszm_matrices.append(glszm.astype(np.int64))
            gldzm_matrices.append(gldzm.astype(np.int64))

        self.glszm_2D_matrices = np.array(glszm_matrices)
        self.gldzm_2D_matrices = np.array(gldzm_matrices)
        self.no_of_roi_voxels = roi_voxels

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

    def calc_ngld_3d_matrix(self):
        x, y, z = self.image.shape
        ngldm = np.zeros((self.lvl, 27), dtype=np.int64)

        # Generate valid 3D offsets (excluding (0,0,0))
        offsets = np.array([
            (dx, dy, dz) for dx in range(-1, 2)
            for dy in range(-1, 2)
            for dz in range(-1, 2)
            if not (dx == dy == dz == 0)
        ])

        # Get valid voxel positions (ignoring NaNs)
        valid_mask = ~np.isnan(self.image)

        for lvl in range(self.lvl):
            mask = (self.image == lvl) & valid_mask
            lvl_i, lvl_j, lvl_k = np.where(mask)

            if lvl_i.size == 0:
                continue  # Skip if no occurrences of the level

            # Compute neighbor positions using broadcasting
            neighbor_i = lvl_i[:, None] + offsets[:, 0]
            neighbor_j = lvl_j[:, None] + offsets[:, 1]
            neighbor_k = lvl_k[:, None] + offsets[:, 2]

            # Ensure neighbors are in bounds
            valid_neighbors = (
                    (0 <= neighbor_i) & (neighbor_i < x) &
                    (0 <= neighbor_j) & (neighbor_j < y) &
                    (0 <= neighbor_k) & (neighbor_k < z)
            )

            # Flatten and filter valid indices
            neighbor_i = neighbor_i[valid_neighbors]
            neighbor_j = neighbor_j[valid_neighbors]
            neighbor_k = neighbor_k[valid_neighbors]

            # Find valid neighbor mask
            valid_neighbor_mask = mask[neighbor_i, neighbor_j, neighbor_k]

            # Compute number of neighbors per voxel
            neighbor_counts = np.sum(valid_neighbor_mask.reshape(len(lvl_i), -1), axis=1)

            # Ensure counts fall within valid range before using bincount
            if neighbor_counts.size > 0:
                j_k = np.bincount(neighbor_counts, minlength=27)
                ngldm[lvl, :len(j_k)] += j_k

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

    def calc_2d_gldzm_features(self):

        number_of_slices = self.gldzm_2D_matrices.shape[0]
        weights = []

        for i in range(number_of_slices):
            M = self.gldzm_2D_matrices[i]
            weight = 1
            if self.slice_weight:
                weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
            weights.append(weight)

            self.short_runs_emphasis_list.append(self.calc_short_emphasis(M))
            self.long_runs_emphasis_list.append(self.calc_long_emphasis(M))
            self.low_grey_level_run_emphasis_list.append(self.calc_low_gr_lvl_emphasis(M))
            self.high_gr_lvl_emphasis_list.append(self.calc_high_gr_lvl_emphasis(M))
            self.short_low_gr_lvl_emphasis_list.append(self.calc_short_low_gr_lvl_emphasis(M))
            self.short_high_gr_lvl_emphasis_list.append(self.calc_short_high_gr_lvl_emphasis(M))
            self.long_low_gr_lvl_emphasis_list.append(self.calc_long_low_gr_lvl_emphasis(M))
            self.long_high_gr_lvl_emphasis_list.append(self.calc_long_high_gr_lvl_emphasis(M))
            self.non_uniformity_list.append(self.calc_non_uniformity(M))
            self.norm_non_uniformity_list.append(self.calc_norm_non_uniformity(M))
            self.length_non_uniformity_list.append(self.calc_length_non_uniformity(M))
            self.norm_length_non_uniformity_list.append(self.calc_norm_length_non_uniformity(M))
            self.percentage_list.append(self.calc_percentage(M, self.no_of_roi_voxels[i]))
            self.gr_lvl_var_list.append(self.calc_gr_lvl_var(M))
            self.length_var_list.append(self.calc_length_var(M))
            self.entropy_list.append(self.calc_entropy(M))

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


