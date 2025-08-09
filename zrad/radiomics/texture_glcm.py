import numpy as np


class GLCM:

    def __init__(self, image, slice_weight=False, slice_median=False):
        self.image = image
        self.slice_weight = slice_weight
        self.slice_median = slice_median
        self.lvl = int(np.nanmax(self.image) + 1)
        self.glcm_2d_matrices = None
        self.glcm_3d_matrix = None

        self.glcm_2d_matrices = []
        self.slice_no_of_roi_voxels = []

        # Feature names used across the class. The first 25 match the benchmark
        # feature numbers used in the original implementation.
        self.feature_names = [
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

        # Dynamically create attributes for feature values and per-slice lists
        # to avoid repetitive code.
        for name in self.feature_names:
            setattr(self, name, 0)
            setattr(self, f"{name}_list", [])

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _reset_feature_lists(self):
        """Clear all feature lists used for intermediate calculations."""
        for name in self.feature_names:
            getattr(self, f"{name}_list").clear()

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
        """Compute features for ``glcm`` and append them to the lists."""
        features = self._calc_features_from_glcm(glcm)
        for name, value in features.items():
            getattr(self, f"{name}_list").append(value)

    def _finalize_features(self, weights):
        """Finalize feature values from stored lists using ``weights``.

        Depending on ``slice_median`` and ``slice_weight`` the features are
        aggregated using a median or a weighted average.
        """

        if self.slice_median and not self.slice_weight:
            for name in self.feature_names:
                setattr(self, name, np.median(getattr(self, f"{name}_list")))
        elif not self.slice_median:
            for name in self.feature_names:
                setattr(
                    self,
                    name,
                    np.average(getattr(self, f"{name}_list"), weights=weights),
                )
        else:
            print("Weighted median not supported. Aborted!")

    # ------------------------------------------------------------------
    # Common GLCM computation
    # ------------------------------------------------------------------

    def _calc_glcm(self, img, offsets):
        """Return GLCM matrices for ``img`` for each ``offset``.

        Parameters
        ----------
        img : ndarray
            2D or 3D image containing integer grey levels and NaNs.
        offsets : list of tuple
            Offsets specified in ``(dx, dy)`` for 2D or ``(dx, dy, dz)`` for 3D.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_dirs, lvl, lvl)`` with dtype ``np.int64``.
        """

        lvl = self.lvl
        glcms = np.zeros((len(offsets), lvl, lvl), dtype=np.int64)
        for idx, off in enumerate(offsets):
            shift = tuple(reversed(off))  # match array axis order (z, y, x)
            slices1 = []
            slices2 = []
            for axis, s in enumerate(shift):
                if s >= 0:
                    slices1.append(slice(0, img.shape[axis] - s))
                    slices2.append(slice(s, img.shape[axis]))
                else:
                    slices1.append(slice(-s, img.shape[axis]))
                    slices2.append(slice(0, img.shape[axis] + s))

            arr1 = img[tuple(slices1)]
            arr2 = img[tuple(slices2)]

            mask = ~np.isnan(arr1) & ~np.isnan(arr2)
            if np.any(mask):
                y = arr1[mask].astype(np.int64)
                x = arr2[mask].astype(np.int64)
                pairs = y * lvl + x
                hist = np.bincount(pairs, minlength=lvl * lvl).reshape(lvl, lvl)
                glcms[idx] = hist + hist.T

        return glcms

    # ------------------------------------------------------------------
    # Original public API
    # ------------------------------------------------------------------

    def calc_glc_2d_matrices(self):

        self.tot_no_of_roi_voxels = np.sum(~np.isnan(self.image))
        offsets_2d = [(1, 0), (1, 1), (0, 1), (-1, 1)]

        for z in range(self.image.shape[2]):
            slice_img = self.image[:, :, z]
            if np.all(np.isnan(slice_img)):
                continue
            self.slice_no_of_roi_voxels.append(np.sum(~np.isnan(slice_img)))
            glcm_slice = self._calc_glcm(slice_img, offsets_2d)
            self.glcm_2d_matrices.append(glcm_slice)

        self.glcm_2d_matrices = np.array(self.glcm_2d_matrices)

    def calc_glc_3d_matrix(self):  # arr, dir_vector, n_bits):

        offsets_3d = [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (0, 1, -1),
            (1, 0, 1),
            (1, 0, -1),
            (1, 1, 0),
            (1, -1, 0),
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (1, -1, -1),
        ]

        self.glcm_3d_matrix = self._calc_glcm(self.image, offsets_3d)

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


