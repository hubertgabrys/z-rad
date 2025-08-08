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

        self.joint_max = 0  # 3.6.1
        self.joint_average = 0  # 3.6.2
        self.joint_var = 0  # 3.6.3
        self.joint_entropy = 0  # 3.6.4
        self.dif_average = 0  # 3.6.5
        self.dif_var = 0  # 3.6.6
        self.dif_entropy = 0  # 3.6.7
        self.sum_average = 0  # 3.6.8
        self.sum_var = 0  # 3.6.9
        self.sum_entropy = 0  # 3.6.10
        self.ang_second_moment = 0  # 3.6.11
        self.contrast = 0  # 3.6.12
        self.dissimilarity = 0  # 3.6.13
        self.inv_diff = 0  # 3.6.14
        self.norm_inv_diff = 0  # 3.6.15
        self.inv_diff_moment = 0  # 3.6.16
        self.norm_inv_diff_moment = 0  # 3.6.17
        self.inv_variance = 0  # 3.6.18
        self.cor = 0  # 3.6.19
        self.autocor = 0  # 3.6.20
        self.cluster_tendency = 0  # 3.6.21
        self.cluster_shade = 0  # 3.6.22
        self.cluster_prominence = 0  # 3.6.23
        self.inf_cor_1 = 0  # 3.6.24
        self.inf_cor_2 = 0  # 3.6.25

        self.joint_max_list = []  # 3.6.1
        self.joint_average_list = []  # 3.6.2
        self.joint_var_list = []  # 3.6.3
        self.joint_entropy_list = []  # 3.6.4
        self.dif_average_list = []  # 3.6.5
        self.dif_var_list = []  # 3.6.6
        self.dif_entropy_list = []  # 3.6.7
        self.sum_average_list = []  # 3.6.8
        self.sum_var_list = []  # 3.6.9
        self.sum_entropy_list = []  # 3.6.10
        self.ang_second_moment_list = []  # 3.6.11
        self.contrast_list = []  # 3.6.12
        self.dissimilarity_list = []  # 3.6.13
        self.inv_diff_list = []  # 3.6.14
        self.norm_inv_diff_list = []  # 3.6.15
        self.inv_diff_moment_list = []  # 3.6.16
        self.norm_inv_diff_moment_list = []  # 3.6.17
        self.inv_variance_list = []  # 3.6.18
        self.cor_list = []  # 3.6.19
        self.autocor_list = []  # 3.6.20
        self.cluster_tendency_list = []  # 3.6.21
        self.cluster_shade_list = []  # 3.6.22
        self.cluster_prominence_list = []  # 3.6.23
        self.inf_cor_1_list = []  # 3.6.24
        self.inf_cor_2_list = []  # 3.6.25

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

    def calc_glcm_3d_matrix_my(self):
        x, y, z = self.image.shape
        directions = [(0, 0, 1), (0, 1, -1), (0, 1, 0),
                      (0, 1, 1), (1, -1, -1), (1, -1, 0),
                      (1, -1, 1), (1, 0, -1), (1, 0, 0),
                      (1, 0, 1), (1, 1, -1), (1, 1, 0),
                      (1, 1, 1)
                      ]

        self.glcm_3d_matrix = []
        for direction in directions:
            glcm = np.zeros((self.lvl, self.lvl))
            for i in range(x):
                for j in range(y):
                    for k in range(z):
                        if np.isnan(self.image[i, j, k]):
                            continue  # Skip cells with np.nan

                        dx, dy, dz = direction
                        gr_lvl = int(self.image[i, j, k])

                        new_i, new_j, new_k = i + dx, j + dy, k + dz
                        if 0 <= new_i < x and 0 <= new_j < y and 0 <= new_k < z and not np.isnan(
                                self.image[new_i, new_j, new_k]):
                            glcm[gr_lvl, int(self.image[new_i, new_j, new_k])] += 1

            self.glcm_3d_matrix.append(glcm + glcm.T)
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

        number_of_slices = self.glcm_2d_matrices.shape[0]
        number_of_directions = self.glcm_2d_matrices.shape[1]
        weights = []
        for i in range(number_of_slices):
            for j in range(number_of_directions):
                glcm_slice = self.glcm_2d_matrices[i][j] / np.sum(self.glcm_2d_matrices[i][j])
                weight = 1
                if self.slice_weight:
                    weight = self.slice_no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
                weights.append(weight)

                self.joint_max_list.append(np.max(glcm_slice))
                glcm_ij_joint_average = self.calc_joint_average(glcm_slice)
                self.joint_average_list.append(glcm_ij_joint_average)
                self.joint_var_list.append(self.calc_joint_var(glcm_slice, glcm_ij_joint_average))
                self.joint_entropy_list.append(self.calc_joint_entropy(glcm_slice))

                p_minus = self.calc_p_minus(glcm_slice)
                glcm_ij_dif_average = self.calc_diff_average(p_minus)
                self.dif_average_list.append(glcm_ij_dif_average)
                self.dif_var_list.append(self.calc_dif_var(p_minus, glcm_ij_dif_average))
                self.dif_entropy_list.append(self.calc_diff_entropy(p_minus))

                p_plus = self.calc_p_plus(glcm_slice)
                glcm_ij_sum_average = self.calc_sum_average(p_plus)
                self.sum_average_list.append(glcm_ij_sum_average)
                self.sum_var_list.append(self.calc_sum_var(p_plus, glcm_ij_sum_average))
                self.sum_entropy_list.append(self.calc_sum_entropy(p_plus))

                self.ang_second_moment_list.append(self.calc_second_moment(glcm_slice))
                self.contrast_list.append(self.calc_contrast(glcm_slice))
                self.dissimilarity_list.append(self.calc_dissimilarity(glcm_slice))
                self.inv_diff_list.append(self.calc_inverse_diff(glcm_slice))
                self.norm_inv_diff_list.append(self.calc_norm_inv_diff(glcm_slice))
                self.inv_diff_moment_list.append(self.calc_inv_diff_moment(p_minus))
                self.norm_inv_diff_moment_list.append(self.calc_norm_inv_diff_moment(p_minus))
                self.inv_variance_list.append(self.calc_inv_variance(p_minus))

                self.cor_list.append(self.calc_correlation(glcm_slice))
                self.autocor_list.append(self.calc_autocor(glcm_slice))
                self.cluster_tendency_list.append(self.calc_cluster_tendency_shade_prominence(glcm_slice, 2))
                self.cluster_shade_list.append(self.calc_cluster_tendency_shade_prominence(glcm_slice, 3))
                self.cluster_prominence_list.append(self.calc_cluster_tendency_shade_prominence(glcm_slice, 4))

                self.inf_cor_1_list.append(self.calc_information_correlation_1(glcm_slice))
                self.inf_cor_2_list.append(self.calc_information_correlation_2(glcm_slice))

        if self.slice_median and not self.slice_weight:
            self.joint_max = np.median(self.joint_max_list)
            self.joint_average = np.median(self.joint_average_list)
            self.joint_var = np.median(self.joint_var_list)
            self.joint_entropy = np.median(self.joint_entropy_list)
            self.dif_average = np.median(self.dif_average_list)
            self.dif_var = np.median(self.dif_var_list)
            self.dif_entropy = np.median(self.dif_entropy_list)
            self.sum_average = np.median(self.sum_average_list)
            self.sum_var = np.median(self.sum_var_list)
            self.sum_entropy = np.median(self.sum_entropy_list)
            self.ang_second_moment = np.median(self.ang_second_moment_list)
            self.contrast = np.median(self.contrast_list)
            self.dissimilarity = np.median(self.dissimilarity_list)
            self.inv_diff = np.median(self.inv_diff_list)
            self.norm_inv_diff = np.median(self.norm_inv_diff_list)
            self.inv_diff_moment = np.median(self.inv_diff_moment_list)
            self.norm_inv_diff_moment = np.median(self.norm_inv_diff_moment_list)
            self.inv_variance = np.median(self.inv_variance_list)
            self.cor = np.median(self.cor_list)
            self.autocor = np.median(self.autocor_list)
            self.cluster_tendency = np.median(self.cluster_tendency_list)
            self.cluster_shade = np.median(self.cluster_shade_list)
            self.cluster_prominence = np.median(self.cluster_prominence_list)
            self.inf_cor_1 = np.median(self.inf_cor_1_list)
            self.inf_cor_2 = np.median(self.inf_cor_2_list)

        elif not self.slice_median:
            self.joint_max = np.average(self.joint_max_list, weights=weights)
            self.joint_average = np.average(self.joint_average_list, weights=weights)
            self.joint_var = np.average(self.joint_var_list, weights=weights)
            self.joint_entropy = np.average(self.joint_entropy_list, weights=weights)
            self.dif_average = np.average(self.dif_average_list, weights=weights)
            self.dif_var = np.average(self.dif_var_list, weights=weights)
            self.dif_entropy = np.average(self.dif_entropy_list, weights=weights)
            self.sum_average = np.average(self.sum_average_list, weights=weights)
            self.sum_var = np.average(self.sum_var_list, weights=weights)
            self.sum_entropy = np.average(self.sum_entropy_list, weights=weights)
            self.ang_second_moment = np.average(self.ang_second_moment_list, weights=weights)
            self.contrast = np.average(self.contrast_list, weights=weights)
            self.dissimilarity = np.average(self.dissimilarity_list, weights=weights)
            self.inv_diff = np.average(self.inv_diff_list, weights=weights)
            self.norm_inv_diff = np.average(self.norm_inv_diff_list, weights=weights)
            self.inv_diff_moment = np.average(self.inv_diff_moment_list, weights=weights)
            self.norm_inv_diff_moment = np.average(self.norm_inv_diff_moment_list, weights=weights)
            self.inv_variance = np.average(self.inv_variance_list, weights=weights)
            self.cor = np.average(self.cor_list, weights=weights)
            self.autocor = np.average(self.autocor_list, weights=weights)
            self.cluster_tendency = np.average(self.cluster_tendency_list, weights=weights)
            self.cluster_shade = np.average(self.cluster_shade_list, weights=weights)
            self.cluster_prominence = np.average(self.cluster_prominence_list, weights=weights)
            self.inf_cor_1 = np.average(self.inf_cor_1_list, weights=weights)
            self.inf_cor_2 = np.average(self.inf_cor_2_list, weights=weights)
        else:
            print('Weighted median not supported. Aborted!')
            return

    def calc_2d_slice_merged_glcm_features(self):

        number_of_slices = self.glcm_2d_matrices.shape[0]
        weights = []

        averaged_glcm = np.sum(self.glcm_2d_matrices, axis=1)
        for slice_id in range(number_of_slices):
            glcm_slice = averaged_glcm[slice_id] / np.sum(averaged_glcm[slice_id])
            weight = 1
            if self.slice_weight:
                weight = self.slice_no_of_roi_voxels[slice_id] / self.tot_no_of_roi_voxels
            weights.append(weight)

            self.joint_max_list.append(np.max(glcm_slice))
            glcm_i_joint_average = self.calc_joint_average(glcm_slice)
            self.joint_average_list.append(glcm_i_joint_average)
            self.joint_var_list.append(self.calc_joint_var(glcm_slice, glcm_i_joint_average))
            self.joint_entropy_list.append(self.calc_joint_entropy(glcm_slice))

            p_minus = self.calc_p_minus(glcm_slice)
            glcm_i_dif_average = self.calc_diff_average(p_minus)
            self.dif_average_list.append(glcm_i_dif_average)
            self.dif_var_list.append(self.calc_dif_var(p_minus, glcm_i_dif_average))
            self.dif_entropy_list.append(self.calc_diff_entropy(p_minus))

            p_plus = self.calc_p_plus(glcm_slice)
            glcm_i_sum_average = self.calc_sum_average(p_plus)
            self.sum_average_list.append(glcm_i_sum_average)
            self.sum_var_list.append(self.calc_sum_var(p_plus, glcm_i_sum_average))
            self.sum_entropy_list.append(self.calc_sum_entropy(p_plus))

            self.ang_second_moment_list.append(self.calc_second_moment(glcm_slice))
            self.contrast_list.append(self.calc_contrast(glcm_slice))
            self.dissimilarity_list.append(self.calc_dissimilarity(glcm_slice))
            self.inv_diff_list.append(self.calc_inverse_diff(glcm_slice))
            self.norm_inv_diff_list.append(self.calc_norm_inv_diff(glcm_slice))
            self.inv_diff_moment_list.append(self.calc_inv_diff_moment(p_minus))
            self.norm_inv_diff_moment_list.append(self.calc_norm_inv_diff_moment(p_minus))
            self.inv_variance_list.append(self.calc_inv_variance(p_minus))

            self.cor_list.append(self.calc_correlation(glcm_slice))
            self.autocor_list.append(self.calc_autocor(glcm_slice))
            self.cluster_tendency_list.append(self.calc_cluster_tendency_shade_prominence(glcm_slice, 2))
            self.cluster_shade_list.append(self.calc_cluster_tendency_shade_prominence(glcm_slice, 3))
            self.cluster_prominence_list.append(self.calc_cluster_tendency_shade_prominence(glcm_slice, 4))

            self.inf_cor_1_list.append(self.calc_information_correlation_1(glcm_slice))
            self.inf_cor_2_list.append(self.calc_information_correlation_2(glcm_slice))

        if self.slice_median and not self.slice_weight:
            self.joint_max = np.median(self.joint_max_list)
            self.joint_average = np.median(self.joint_average_list)
            self.joint_var = np.median(self.joint_var_list)
            self.joint_entropy = np.median(self.joint_entropy_list)
            self.dif_average = np.median(self.dif_average_list)
            self.dif_var = np.median(self.dif_var_list)
            self.dif_entropy = np.median(self.dif_entropy_list)
            self.sum_average = np.median(self.sum_average_list)
            self.sum_var = np.median(self.sum_var_list)
            self.sum_entropy = np.median(self.sum_entropy_list)
            self.ang_second_moment = np.median(self.ang_second_moment_list)
            self.contrast = np.median(self.contrast_list)
            self.dissimilarity = np.median(self.dissimilarity_list)
            self.inv_diff = np.median(self.inv_diff_list)
            self.norm_inv_diff = np.median(self.norm_inv_diff_list)
            self.inv_diff_moment = np.median(self.inv_diff_moment_list)
            self.norm_inv_diff_moment = np.median(self.norm_inv_diff_moment_list)
            self.inv_variance = np.median(self.inv_variance_list)
            self.cor = np.median(self.cor_list)
            self.autocor = np.median(self.autocor_list)
            self.cluster_tendency = np.median(self.cluster_tendency_list)
            self.cluster_shade = np.median(self.cluster_shade_list)
            self.cluster_prominence = np.median(self.cluster_prominence_list)
            self.inf_cor_1 = np.median(self.inf_cor_1_list)
            self.inf_cor_2 = np.median(self.inf_cor_2_list)

        elif not self.slice_median:
            self.joint_max = np.average(self.joint_max_list, weights=weights)
            self.joint_average = np.average(self.joint_average_list, weights=weights)
            self.joint_var = np.average(self.joint_var_list, weights=weights)
            self.joint_entropy = np.average(self.joint_entropy_list, weights=weights)
            self.dif_average = np.average(self.dif_average_list, weights=weights)
            self.dif_var = np.average(self.dif_var_list, weights=weights)
            self.dif_entropy = np.average(self.dif_entropy_list, weights=weights)
            self.sum_average = np.average(self.sum_average_list, weights=weights)
            self.sum_var = np.average(self.sum_var_list, weights=weights)
            self.sum_entropy = np.average(self.sum_entropy_list, weights=weights)
            self.ang_second_moment = np.average(self.ang_second_moment_list, weights=weights)
            self.contrast = np.average(self.contrast_list, weights=weights)
            self.dissimilarity = np.average(self.dissimilarity_list, weights=weights)
            self.inv_diff = np.average(self.inv_diff_list, weights=weights)
            self.norm_inv_diff = np.average(self.norm_inv_diff_list, weights=weights)
            self.inv_diff_moment = np.average(self.inv_diff_moment_list, weights=weights)
            self.norm_inv_diff_moment = np.average(self.norm_inv_diff_moment_list, weights=weights)
            self.inv_variance = np.average(self.inv_variance_list, weights=weights)
            self.cor = np.average(self.cor_list, weights=weights)
            self.autocor = np.average(self.autocor_list, weights=weights)
            self.cluster_tendency = np.average(self.cluster_tendency_list, weights=weights)
            self.cluster_shade = np.average(self.cluster_shade_list, weights=weights)
            self.cluster_prominence = np.average(self.cluster_prominence_list, weights=weights)
            self.inf_cor_1 = np.average(self.inf_cor_1_list, weights=weights)
            self.inf_cor_2 = np.average(self.inf_cor_2_list, weights=weights)
        else:
            print('Weighted median not supported. Aborted!')
            return

    def calc_2_5d_merged_glcm_features(self):
        glcm = np.sum(np.sum(self.glcm_2d_matrices, axis=1), axis=0)

        glcm = glcm / np.sum(glcm)

        self.joint_max = np.max(glcm)
        glcm_joint_average = self.calc_joint_average(glcm)
        self.joint_average = glcm_joint_average
        self.joint_var = self.calc_joint_var(glcm, glcm_joint_average)
        self.joint_entropy = self.calc_joint_entropy(glcm)

        p_minus = self.calc_p_minus(glcm)
        glcm_dif_average = self.calc_diff_average(p_minus)
        self.dif_average = glcm_dif_average
        self.dif_var = self.calc_dif_var(p_minus, glcm_dif_average)
        self.dif_entropy = self.calc_diff_entropy(p_minus)

        p_plus = self.calc_p_plus(glcm)
        glcm_sum_average = self.calc_sum_average(p_plus)
        self.sum_average = glcm_sum_average
        self.sum_var = self.calc_sum_var(p_plus, glcm_sum_average)
        self.sum_entropy = self.calc_sum_entropy(p_plus)

        self.ang_second_moment = self.calc_second_moment(glcm)
        self.contrast = self.calc_contrast(glcm)
        self.dissimilarity = self.calc_dissimilarity(glcm)
        self.inv_diff = self.calc_inverse_diff(glcm)
        self.norm_inv_diff = self.calc_norm_inv_diff(glcm)
        self.inv_diff_moment = self.calc_inv_diff_moment(p_minus)
        self.norm_inv_diff_moment = self.calc_norm_inv_diff_moment(p_minus)
        self.inv_variance = self.calc_inv_variance(p_minus)

        self.cor = self.calc_correlation(glcm)
        self.autocor = self.calc_autocor(glcm)
        self.cluster_tendency = self.calc_cluster_tendency_shade_prominence(glcm, 2)
        self.cluster_shade = self.calc_cluster_tendency_shade_prominence(glcm, 3)
        self.cluster_prominence = self.calc_cluster_tendency_shade_prominence(glcm, 4)

        self.inf_cor_1 = self.calc_information_correlation_1(glcm)
        self.inf_cor_2 = self.calc_information_correlation_2(glcm)

    def calc_2_5d_direction_merged_glcm_features(self):
        number_of_directions = self.glcm_2d_matrices.shape[1]

        averaged_glcm = np.sum(self.glcm_2d_matrices, axis=0)  # / number_of_slices

        for i in range(number_of_directions):
            M_i = averaged_glcm[i] / np.sum(averaged_glcm[i])
            self.joint_max += np.max(M_i)
            glcm_i_joint_average = self.calc_joint_average(M_i)
            self.joint_average += glcm_i_joint_average
            self.joint_var += self.calc_joint_var(M_i, glcm_i_joint_average)
            self.joint_entropy += self.calc_joint_entropy(M_i)

            p_minus = self.calc_p_minus(M_i)
            glcm_i_dif_average = self.calc_diff_average(p_minus)
            self.dif_average += glcm_i_dif_average
            self.dif_var += self.calc_dif_var(p_minus, glcm_i_dif_average)
            self.dif_entropy += self.calc_diff_entropy(p_minus)

            p_plus = self.calc_p_plus(M_i)
            glcm_i_sum_average = self.calc_sum_average(p_plus)
            self.sum_average += glcm_i_sum_average
            self.sum_var += self.calc_sum_var(p_plus, glcm_i_sum_average)
            self.sum_entropy += self.calc_sum_entropy(p_plus)

            self.ang_second_moment += self.calc_second_moment(M_i)
            self.contrast += self.calc_contrast(M_i)
            self.dissimilarity += self.calc_dissimilarity(M_i)
            self.inv_diff += self.calc_inverse_diff(M_i)
            self.norm_inv_diff += self.calc_norm_inv_diff(M_i)
            self.inv_diff_moment += self.calc_inv_diff_moment(p_minus)
            self.norm_inv_diff_moment += self.calc_norm_inv_diff_moment(p_minus)
            self.inv_variance += self.calc_inv_variance(p_minus)

            self.cor += self.calc_correlation(M_i)
            self.autocor += self.calc_autocor(M_i)
            self.cluster_tendency += self.calc_cluster_tendency_shade_prominence(M_i, 2)
            self.cluster_shade += self.calc_cluster_tendency_shade_prominence(M_i, 3)
            self.cluster_prominence += self.calc_cluster_tendency_shade_prominence(M_i, 4)

            self.inf_cor_1 += self.calc_information_correlation_1(M_i)
            self.inf_cor_2 += self.calc_information_correlation_2(M_i)

        self.joint_max /= number_of_directions
        self.joint_average /= number_of_directions
        self.joint_var /= number_of_directions
        self.joint_entropy /= number_of_directions

        self.dif_average /= number_of_directions
        self.dif_var /= number_of_directions
        self.dif_entropy /= number_of_directions
        self.sum_average /= number_of_directions
        self.sum_var /= number_of_directions
        self.sum_entropy /= number_of_directions

        self.ang_second_moment /= number_of_directions
        self.contrast /= number_of_directions
        self.dissimilarity /= number_of_directions
        self.inv_diff /= number_of_directions
        self.norm_inv_diff /= number_of_directions
        self.inv_diff_moment /= number_of_directions
        self.norm_inv_diff_moment /= number_of_directions
        self.inv_variance /= number_of_directions

        self.cor /= number_of_directions
        self.autocor /= number_of_directions
        self.cluster_tendency /= number_of_directions
        self.cluster_shade /= number_of_directions
        self.cluster_prominence /= number_of_directions

        self.inf_cor_1 /= number_of_directions
        self.inf_cor_2 /= number_of_directions

    def calc_3d_averaged_glcm_features(self):

        nuber_of_dir_3D = 13

        for glcm_i in self.glcm_3d_matrix:
            norm = np.sum(glcm_i)
            glcm_i = glcm_i / norm
            self.joint_max += np.max(glcm_i)
            glcm_i_joint_average = self.calc_joint_average(glcm_i)
            self.joint_average += glcm_i_joint_average
            self.joint_var += self.calc_joint_var(glcm_i, glcm_i_joint_average)
            self.joint_entropy += self.calc_joint_entropy(glcm_i)

            p_minus = self.calc_p_minus(glcm_i)
            glcm_i_dif_average = self.calc_diff_average(p_minus)
            self.dif_average += glcm_i_dif_average
            self.dif_var += self.calc_dif_var(p_minus, glcm_i_dif_average)
            self.dif_entropy += self.calc_diff_entropy(p_minus)

            p_plus = self.calc_p_plus(glcm_i)
            glcm_i_sum_average = self.calc_sum_average(p_plus)
            self.sum_average += glcm_i_sum_average
            self.sum_var += self.calc_sum_var(p_plus, glcm_i_sum_average)
            self.sum_entropy += self.calc_sum_entropy(p_plus)

            self.ang_second_moment += self.calc_second_moment(glcm_i)
            self.contrast += self.calc_contrast(glcm_i)
            self.dissimilarity += self.calc_dissimilarity(glcm_i)
            self.inv_diff += self.calc_inverse_diff(glcm_i)
            self.norm_inv_diff += self.calc_norm_inv_diff(glcm_i)
            self.inv_diff_moment += self.calc_inv_diff_moment(p_minus)
            self.norm_inv_diff_moment += self.calc_norm_inv_diff_moment(p_minus)
            self.inv_variance += self.calc_inv_variance(p_minus)

            self.cor += self.calc_correlation(glcm_i)
            self.autocor += self.calc_autocor(glcm_i)
            self.cluster_tendency += self.calc_cluster_tendency_shade_prominence(glcm_i, 2)
            self.cluster_shade += self.calc_cluster_tendency_shade_prominence(glcm_i, 3)
            self.cluster_prominence += self.calc_cluster_tendency_shade_prominence(glcm_i, 4)

            self.inf_cor_1 += self.calc_information_correlation_1(glcm_i)
            self.inf_cor_2 += self.calc_information_correlation_2(glcm_i)

        self.joint_max /= nuber_of_dir_3D
        self.joint_average /= nuber_of_dir_3D
        self.joint_var /= nuber_of_dir_3D
        self.joint_entropy /= nuber_of_dir_3D

        self.dif_average /= nuber_of_dir_3D
        self.dif_var /= nuber_of_dir_3D
        self.dif_entropy /= nuber_of_dir_3D
        self.sum_average /= nuber_of_dir_3D
        self.sum_var /= nuber_of_dir_3D
        self.sum_entropy /= nuber_of_dir_3D

        self.ang_second_moment /= nuber_of_dir_3D
        self.contrast /= nuber_of_dir_3D
        self.dissimilarity /= nuber_of_dir_3D
        self.inv_diff /= nuber_of_dir_3D
        self.norm_inv_diff /= nuber_of_dir_3D
        self.inv_diff_moment /= nuber_of_dir_3D
        self.norm_inv_diff_moment /= nuber_of_dir_3D
        self.inv_variance /= nuber_of_dir_3D

        self.cor /= nuber_of_dir_3D
        self.autocor /= nuber_of_dir_3D
        self.cluster_tendency /= nuber_of_dir_3D
        self.cluster_shade /= nuber_of_dir_3D
        self.cluster_prominence /= nuber_of_dir_3D

        self.inf_cor_1 /= nuber_of_dir_3D
        self.inf_cor_2 /= nuber_of_dir_3D

    def calc_3d_merged_glcm_features(self):

        M = np.sum(self.glcm_3d_matrix, axis=0)
        M = M / np.sum(M)

        self.joint_max = np.max(M)
        self.joint_average = self.calc_joint_average(M)
        self.joint_var = self.calc_joint_var(M, self.joint_average)
        self.joint_entropy = self.calc_joint_entropy(M)

        p_minus = self.calc_p_minus(M)
        M_dif_average = self.calc_diff_average(p_minus)
        self.dif_average = M_dif_average
        self.dif_var = self.calc_dif_var(p_minus, M_dif_average)
        self.dif_entropy = self.calc_diff_entropy(p_minus)

        p_plus = self.calc_p_plus(M)
        M_sum_average = self.calc_sum_average(p_plus)
        self.sum_average = M_sum_average
        self.sum_var = self.calc_sum_var(p_plus, M_sum_average)
        self.sum_entropy = self.calc_sum_entropy(p_plus)

        self.ang_second_moment = self.calc_second_moment(M)
        self.contrast = self.calc_contrast(M)
        self.dissimilarity = self.calc_dissimilarity(M)
        self.inv_diff = self.calc_inverse_diff(M)
        self.norm_inv_diff = self.calc_norm_inv_diff(M)
        self.inv_diff_moment = self.calc_inv_diff_moment(p_minus)
        self.norm_inv_diff_moment = self.calc_norm_inv_diff_moment(p_minus)
        self.inv_variance = self.calc_inv_variance(p_minus)

        self.cor = self.calc_correlation(M)
        self.autocor = self.calc_autocor(M)
        self.cluster_tendency = self.calc_cluster_tendency_shade_prominence(M, 2)
        self.cluster_shade = self.calc_cluster_tendency_shade_prominence(M, 3)
        self.cluster_prominence = self.calc_cluster_tendency_shade_prominence(M, 4)

        self.inf_cor_1 = self.calc_information_correlation_1(M)
        self.inf_cor_2 = self.calc_information_correlation_2(M)


