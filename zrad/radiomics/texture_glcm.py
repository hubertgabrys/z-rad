import numpy as np

from .texture_base import TextureFeatureBase


class GLCM(TextureFeatureBase):

    FEATURE_NAMES = [
        "joint_max",
        "joint_average",
        "joint_var",
        "joint_entropy",
        "dif_average",
        "dif_var",
        "dif_entropy",
        "sum_average",
        "sum_var",
        "sum_entropy",
        "ang_second_moment",
        "contrast",
        "dissimilarity",
        "inv_diff",
        "norm_inv_diff",
        "inv_diff_moment",
        "norm_inv_diff_moment",
        "inv_variance",
        "cor",
        "autocor",
        "cluster_tendency",
        "cluster_shade",
        "cluster_prominence",
        "inf_cor_1",
        "inf_cor_2",
    ]

    def __init__(self, image, slice_weight=False, slice_median=False):
        super().__init__(image, self.FEATURE_NAMES, slice_weight, slice_median)
        self.lvl = int(np.nanmax(self.image) + 1)
        self.glcm_2d_matrices = []
        self.glcm_3d_matrix = None
        self.slice_no_of_roi_voxels = []

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _calc_features_from_glcm(self, glcm):
        """Calculate all features for a single GLCM matrix.

        The input matrix is normalised internally. A dictionary mapping the
        feature name to its value is returned. This method centralises the
        feature calculations and is reused across the different aggregation
        strategies to avoid code duplication.
        """

        glcm = glcm / np.sum(glcm)

        features = {}
        features["joint_max"] = np.max(glcm)
        joint_average = self.calc_joint_average(glcm)
        features["joint_average"] = joint_average
        features["joint_var"] = self.calc_joint_var(glcm, joint_average)
        features["joint_entropy"] = self.calc_joint_entropy(glcm)

        p_minus = self.calc_p_minus(glcm)
        dif_average = self.calc_diff_average(p_minus)
        features["dif_average"] = dif_average
        features["dif_var"] = self.calc_dif_var(p_minus, dif_average)
        features["dif_entropy"] = self.calc_diff_entropy(p_minus)

        p_plus = self.calc_p_plus(glcm)
        sum_average = self.calc_sum_average(p_plus)
        features["sum_average"] = sum_average
        features["sum_var"] = self.calc_sum_var(p_plus, sum_average)
        features["sum_entropy"] = self.calc_sum_entropy(p_plus)

        features["ang_second_moment"] = self.calc_second_moment(glcm)
        features["contrast"] = self.calc_contrast(glcm)
        features["dissimilarity"] = self.calc_dissimilarity(glcm)
        features["inv_diff"] = self.calc_inverse_diff(glcm)
        features["norm_inv_diff"] = self.calc_norm_inv_diff(glcm)
        features["inv_diff_moment"] = self.calc_inv_diff_moment(p_minus)
        features["norm_inv_diff_moment"] = self.calc_norm_inv_diff_moment(p_minus)
        features["inv_variance"] = self.calc_inv_variance(p_minus)

        features["cor"] = self.calc_correlation(glcm)
        features["autocor"] = self.calc_autocor(glcm)
        features["cluster_tendency"] = self.calc_cluster_tendency_shade_prominence(glcm, 2)
        features["cluster_shade"] = self.calc_cluster_tendency_shade_prominence(glcm, 3)
        features["cluster_prominence"] = self.calc_cluster_tendency_shade_prominence(glcm, 4)

        features["inf_cor_1"] = self.calc_information_correlation_1(glcm)
        features["inf_cor_2"] = self.calc_information_correlation_2(glcm)

        return features

    def _append_features(self, glcm):
        """Compute features for ``glcm`` and append them using the base util."""
        features = self._calc_features_from_glcm(glcm)
        super()._append_features(features)

    # ------------------------------------------------------------------
    # Original public API
    # ------------------------------------------------------------------

    def calc_glc_2d_matrices(self):

        def calc_2_d_glcm_slice(image, direction):
            # Ensure only the first two values are used for 2D GLCM
            dx, dy, *_ = direction  # Unpacks only dx and dy, ignoring dz

            rows, cols = image.shape
            glcm_slice = np.zeros((self.lvl, self.lvl), dtype=int)

            # Create mask for NaN values
            nan_mask = np.isnan(image)

            # Compute valid indices
            if dx >= 0:
                valid_i = np.arange(rows - dx)
            else:
                valid_i = np.arange(-dx, rows)

            if dy >= 0:
                valid_j = np.arange(cols - dy)
            else:
                valid_j = np.arange(-dy, cols)

            # Create meshgrid for valid pixel locations
            i_grid, j_grid = np.meshgrid(valid_i, valid_j, indexing='ij')

            # Get pixel values
            row_pixels = image[i_grid, j_grid]
            col_pixels = image[i_grid + dx, j_grid + dy]

            # Mask invalid (NaN) pairs
            valid_pairs = ~nan_mask[i_grid, j_grid] & ~nan_mask[i_grid + dx, j_grid + dy]

            # Update GLCM using NumPy indexing
            np.add.at(glcm_slice, (row_pixels[valid_pairs].astype(int), col_pixels[valid_pairs].astype(int)), 1)

            return glcm_slice

        self.tot_no_of_roi_voxels = np.sum(~np.isnan(self.image))
        for z in range(self.image.shape[2]):
            if not np.all(np.isnan(self.image[:, :, z])):
                self.slice_no_of_roi_voxels.append(np.sum(~np.isnan(self.image[:, :, z])))
                z_slice_list = []
                for direction_2D in [[1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0]]:
                    glcm = calc_2_d_glcm_slice(self.image[:, :, z], direction_2D)
                    z_slice_list.append((glcm + glcm.T))

                self.glcm_2d_matrices.append(z_slice_list)
        self.glcm_2d_matrices = np.array(self.glcm_2d_matrices)

    def calc_glc_3d_matrix(self):  # arr, dir_vector, n_bits):

        self.glcm_3d_matrix = []

        for direction_3D in [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, -1],
                             [1, 0, 1], [1, 0, -1], [1, 1, 0], [1, -1, 0], [1, 1, 1],
                             [1, 1, -1], [1, -1, 1], [1, -1, -1]]:
            co_matrix = np.zeros((self.lvl, self.lvl), dtype=np.float64)

            len_arr, len_arr_0, len_arr_0_0 = len(self.image), len(self.image[0]), len(self.image[0][0])
            min_i, min_y, min_x = max(0, -direction_3D[2]), max(0, -direction_3D[1]), max(0, -direction_3D[0])
            max_i, max_y, max_x = min(len_arr, len_arr - direction_3D[2]), min(len_arr_0,
                                                                               len_arr_0 - direction_3D[1]), min(
                len_arr_0_0, len_arr_0_0 - direction_3D[0])

            arr1 = self.image[min_i:max_i, min_y:max_y, min_x:max_x]
            arr2 = self.image[min_i + direction_3D[2]:max_i + direction_3D[2],
                   min_y + direction_3D[1]:max_y + direction_3D[1], min_x + direction_3D[0]:max_x + direction_3D[0]]

            not_nan_mask = np.logical_and(~np.isnan(arr1), ~np.isnan(arr2))

            y_cm_values = arr1[not_nan_mask].astype(int)
            x_cm_values = arr2[not_nan_mask].astype(int)

            np.add.at(co_matrix, (y_cm_values, x_cm_values), 1)
            np.add.at(co_matrix, (x_cm_values, y_cm_values), 1)

            self.glcm_3d_matrix.append(co_matrix)

        self.glcm_3d_matrix = np.array(self.glcm_3d_matrix)

    def calc_p_minus(self, matrix):
        matrix = np.asarray(matrix)  # Ensure input is a NumPy array
        n_g = matrix.shape[0]

        # Use NumPy advanced indexing to sum along diagonals
        p_minus = np.zeros(n_g)
        for k in range(n_g):  # k should start at 0 (not n_g - 1)
            mask = np.abs(np.subtract.outer(np.arange(n_g), np.arange(n_g))) == k
            p_minus[k] = matrix[mask].sum()

        return p_minus

    def calc_p_plus(self, matrix):
        matrix = np.asarray(matrix)  # Ensure input is a NumPy array
        n_g = matrix.shape[0]

        # Correct size of p_plus
        p_plus = np.zeros(2 * n_g - 1)

        for k in range(2 * n_g - 1):  # Adjust range to start from 0
            mask = np.add.outer(np.arange(n_g), np.arange(n_g)) == k
            p_plus[k] = matrix[mask].sum()
        return p_plus

    def calc_mu_i_and_sigma_i(self, matrix):
        p_i = np.sum(matrix, axis=0)
        mu_i = 0
        for i in range(len(p_i)):
            mu_i += p_i[i] * i

        sigma_i = 0
        for i in range(len(p_i)):
            sigma_i += (i - mu_i) ** 2 * p_i[i]
        sigma_i = np.sqrt(sigma_i)

        return mu_i, sigma_i

    def calc_correlation(self, matrix):
        i, j = np.indices(matrix.shape)
        mu_i, sigma_i = self.calc_mu_i_and_sigma_i(matrix)

        return (np.sum(matrix * i * j) - mu_i ** 2) / sigma_i ** 2

    def calc_cluster_tendency_shade_prominence(self, matrix, pover):

        mu_i, _ = self.calc_mu_i_and_sigma_i(matrix)
        i, j = np.indices(matrix.shape)

        return np.sum((i + j - 2 * mu_i) ** pover * matrix)

    def calc_information_correlation_1(self, matrix):
        p_i_j = matrix
        non_zero_mask_p_i_j = p_i_j != 0
        hxy = (-1) * np.sum(p_i_j[non_zero_mask_p_i_j] * np.log2(p_i_j[non_zero_mask_p_i_j]))

        p_i = np.sum(matrix, axis=0)
        non_zero_mask_p_i = p_i != 0
        hx = (-1) * np.sum(p_i[non_zero_mask_p_i] * np.log2(p_i[non_zero_mask_p_i]))

        hxy_1 = 0
        for i in range(len(p_i_j)):
            for j in range(len(p_i_j)):
                if p_i[i] != 0 and p_i[j] != 0:
                    hxy_1 += p_i_j[i][j] * np.log2(p_i[i] * p_i[j])
        hxy_1 *= (-1)

        return (hxy - hxy_1) / hx

    def calc_information_correlation_2(self, matrix):
        p_i_j = matrix
        non_zero_mask_p_i_j = p_i_j != 0
        hxy = (-1) * np.sum(p_i_j[non_zero_mask_p_i_j] * np.log2(p_i_j[non_zero_mask_p_i_j]))

        p_i = np.sum(matrix, axis=0)

        hxy_2 = 0
        for i in range(len(p_i_j)):
            for j in range(len(p_i_j)):
                if p_i[i] != 0 and p_i[j] != 0:
                    hxy_2 += p_i[i] * p_i[j] * np.log2(p_i[i] * p_i[j])
        hxy_2 *= (-1)

        return np.sqrt(1 - np.exp(-2 * (hxy_2 - hxy)))

    def calc_joint_average(self, matrix):
        i, j = np.indices(matrix.shape)
        return np.sum(matrix * i)

    def calc_joint_var(self, matrix, mu):
        i, j = np.indices(matrix.shape)
        return np.sum(matrix * (i - mu) ** 2)

    def calc_joint_entropy(self, matrix):
        non_zero_mask = matrix != 0
        return (-1) * np.sum(matrix[non_zero_mask] * np.log2(matrix[non_zero_mask]))

    def calc_diff_average(self, p_minus):
        k = np.indices(p_minus.shape)
        diff_average = np.sum(p_minus * k)
        return diff_average

    def calc_dif_var(self, p_minus, mu):
        k = np.indices(p_minus.shape)
        diff_var = np.sum(p_minus * (k - mu) ** 2)
        return diff_var

    def calc_diff_entropy(self, p_minus):
        non_zero_mask = p_minus != 0
        return (-1) * np.sum(p_minus[non_zero_mask] * np.log2(p_minus[non_zero_mask]))

    def calc_sum_average(self, p_plus):
        k = np.indices(p_plus.shape)
        sum_average = np.sum(p_plus * k)
        return sum_average

    def calc_sum_var(self, p_plus, mu):
        k = np.indices(p_plus.shape)
        sum_var = np.sum(p_plus * (k - mu) ** 2)
        return sum_var

    def calc_sum_entropy(self, p_plus):
        non_zero_mask = p_plus != 0
        return (-1) * np.sum(p_plus[non_zero_mask] * np.log2(p_plus[non_zero_mask]))

    def calc_second_moment(self, matrix):
        return np.sum(matrix * matrix)

    def calc_contrast(self, matrix):
        i, j = np.indices(matrix.shape)
        return np.sum(matrix * (i - j) ** 2)

    def calc_dissimilarity(self, matrix):
        i, j = np.indices(matrix.shape)
        return np.sum(matrix * abs(i - j))

    def calc_inverse_diff(self, matrix):
        i, j = np.indices(matrix.shape)
        return np.sum(matrix / (1 + abs(i - j)))

    def calc_norm_inv_diff(self, matrix):
        n_g = len(matrix) - 1
        i, j = np.indices(matrix.shape)
        return np.sum(matrix / (1 + abs(i - j) / n_g))

    def calc_inv_diff_moment(self, p_minus):
        k = np.indices(p_minus.shape)
        return np.sum(p_minus / (1 + k ** 2))

    def calc_norm_inv_diff_moment(self, p_minus):
        k = np.indices(p_minus.shape)
        n_g = len(p_minus) - 1
        return np.sum(p_minus / (1 + (k / n_g) ** 2))

    def calc_inv_variance(self, p_minus):
        k = np.indices(p_minus.shape)
        non_zero_mask = k != 0
        return np.sum(p_minus[1::] / (k[non_zero_mask] ** 2))

    def calc_autocor(self, matrix):
        i, j = np.indices(matrix.shape)
        return np.sum(matrix * i * j)

    def calc_2d_averaged_glcm_features(self):
        self._reset_feature_lists()

        number_of_slices = self.glcm_2d_matrices.shape[0]
        number_of_directions = self.glcm_2d_matrices.shape[1]
        weights = []
        for i in range(number_of_slices):
            weight = (
                self.slice_no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
                if self.slice_weight
                else 1
            )
            for j in range(number_of_directions):
                self._append_features(self.glcm_2d_matrices[i][j])
                weights.append(weight)

        self._finalize_features(weights)

    def calc_2d_slice_merged_glcm_features(self):
        self._reset_feature_lists()

        number_of_slices = self.glcm_2d_matrices.shape[0]
        weights = []

        averaged_glcm = np.sum(self.glcm_2d_matrices, axis=1)
        for slice_id in range(number_of_slices):
            self._append_features(averaged_glcm[slice_id])
            weight = (
                self.slice_no_of_roi_voxels[slice_id] / self.tot_no_of_roi_voxels
                if self.slice_weight
                else 1
            )
            weights.append(weight)

        self._finalize_features(weights)

    def calc_2_5d_merged_glcm_features(self):
        glcm = np.sum(np.sum(self.glcm_2d_matrices, axis=1), axis=0)
        features = self._calc_features_from_glcm(glcm)
        for name, value in features.items():
            setattr(self, name, value)

    def calc_2_5d_direction_merged_glcm_features(self):
        averaged_glcm = np.sum(self.glcm_2d_matrices, axis=0)
        number_of_directions = averaged_glcm.shape[0]
        feature_sums = {name: 0 for name in self.feature_names}

        for glcm in averaged_glcm:
            features = self._calc_features_from_glcm(glcm)
            for name in self.feature_names:
                feature_sums[name] += features[name]

        for name in self.feature_names:
            setattr(self, name, feature_sums[name] / number_of_directions)

    def calc_3d_averaged_glcm_features(self):
        feature_sums = {name: 0 for name in self.feature_names}
        n_dirs = len(self.glcm_3d_matrix)

        for glcm in self.glcm_3d_matrix:
            features = self._calc_features_from_glcm(glcm)
            for name in self.feature_names:
                feature_sums[name] += features[name]

        for name in self.feature_names:
            setattr(self, name, feature_sums[name] / n_dirs)

    def calc_3d_merged_glcm_features(self):
        glcm = np.sum(self.glcm_3d_matrix, axis=0)
        features = self._calc_features_from_glcm(glcm)
        for name, value in features.items():
            setattr(self, name, value)


