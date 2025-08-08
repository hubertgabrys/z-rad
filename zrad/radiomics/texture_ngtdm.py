import numpy as np
from scipy.ndimage import convolve


# list of feature names calculated from the NGTDM matrix
FEATURES = ("coarseness", "contrast", "busyness", "complexity", "strength")


class NGTDM:
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

        self.ngtd_2d_matrices = []
        self.ngtd_3d_matrix = None
        self.slice_no_of_roi_voxels = []

        # initialise feature attributes and containers in a generic way
        self.feature_lists = {name: [] for name in FEATURES}
        for name in FEATURES:
            setattr(self, name, 0)
            setattr(self, f"{name}_list", self.feature_lists[name])

    def _calc_ngtdm(self, img, kernel):
        """Calculate the NGTDM matrix for a provided image and kernel.

        Parameters
        ----------
        img : ndarray
            2D or 3D image slice with NaNs marking out-of-ROI voxels.
        kernel : ndarray
            Convolution kernel describing the neighbourhood (center must be 0).

        Returns
        -------
        tuple (matrix, count)
            The NGTDM matrix and number of valid voxels in ``img``.
        """
        valid = ~np.isnan(img)
        n_vox = int(valid.sum())
        if n_vox == 0:
            return np.zeros((self.lvl, 2), dtype=np.float64), 0

        filled = np.where(valid, img, 0.0)
        neighbor_sum = convolve(filled, kernel, mode="constant", cval=0.0)
        neighbor_count = convolve(valid.astype(np.int8), kernel, mode="constant", cval=0)

        ngtdm = np.zeros((self.lvl, 2), dtype=np.float64)
        for lvl in range(self.lvl):
            mask = (img == lvl) & (neighbor_count > 0)
            n_i = int(mask.sum())
            if n_i > 0:
                mean_nb = neighbor_sum[mask] / neighbor_count[mask]
                s_i = np.abs(lvl - mean_nb).sum()
            else:
                s_i = 0.0
            ngtdm[lvl, 0] = n_i
            ngtdm[lvl, 1] = s_i

        return ngtdm, n_vox

    def calc_ngtd_3d_matrix(self):
        """Compute the 3‑D NGTDM matrix."""
        kernel = np.ones((3, 3, 3), dtype=np.int8)
        kernel[1, 1, 1] = 0
        self.ngtd_3d_matrix, _ = self._calc_ngtdm(self.image, kernel)

    def calc_ngtd_2d_matrices(self):
        """Compute the NGTDM matrix for each axial slice."""
        kernel2d = np.ones((3, 3), dtype=np.int8)
        kernel2d[1, 1] = 0

        slice_matrices = []
        slice_voxel_counts = []

        for z in self.range_z:
            sl = self.image[:, :, z]
            ngtdm_slice, n_vox = self._calc_ngtdm(sl, kernel2d)
            if n_vox == 0:
                continue
            slice_matrices.append(ngtdm_slice)
            slice_voxel_counts.append(n_vox)

        # store results back into object
        self.slice_no_of_roi_voxels = slice_voxel_counts
        self.ngtd_2d_matrices = np.array(slice_matrices)
    def calc_coarseness(self, matrix):
        num = np.sum(matrix[:, 0])
        denum = 0
        for i in range(matrix.shape[0]):
            denum += matrix[i, 0] * matrix[i, 1]
        if denum == 0:
            return 1_000_000  # IBSI 1 QCDE
        else:
            return num / denum

    def calc_contrast(self, matrix):
        n = np.sum(matrix[:, 0])
        n_g = np.sum(matrix[:, 0] != 0)
        s_1 = 0
        s_2 = 0
        for i in range(matrix.shape[0]):
            s_2 += matrix[i, 1]
            for j in range(matrix.shape[0]):
                s_1 += (matrix[i, 0] * matrix[j, 0] * (i - j) ** 2) / (n ** 2)
        num = (s_1 * s_2)
        denum = (n_g * (n_g - 1) * np.sum(matrix[:, 0]))
        if denum == 0:
            return 0
        else:
            return num / denum

    def calc_busyness(self, matrix):
        n = np.sum(matrix[:, 0])
        if n == 0:
            pass
        num = 0
        denum = 0
        for i in range(matrix.shape[0]):
            num += (matrix[i, 0] * matrix[i, 1]) / n
            for j in range(matrix.shape[0]):
                if matrix[i, 0] != 0 and matrix[j, 0] != 0:
                    denum += abs(i * matrix[i, 0] - j * matrix[j, 0]) / n
        if denum == 0:
            return 0
        else:
            return num / denum

    def calc_complexity(self, matrix):
        n = np.sum(matrix[:, 0])
        if n == 0:
            return 0

        sum_compl = 0.0
        # build the double‐sum
        for i in range(matrix.shape[0]):
            p_i, s_i = matrix[i, 0], matrix[i, 1]
            if p_i == 0:
                continue
            for j in range(matrix.shape[0]):
                p_j, s_j = matrix[j, 0], matrix[j, 1]
                if p_j == 0:
                    continue

                # per-IBSI numerator and denominator
                num = (p_i * s_i + p_j * s_j) * abs(i - j) / n
                den = (p_i + p_j) / n
                sum_compl += num / den

        # normalize by N_{v,c} = sum_i p_i
        N_vc = n
        if N_vc == 0:
            return 0
        return sum_compl / N_vc

    def calc_strength(self, matrix):
        n = np.sum(matrix[:, 0])
        num = 0
        denum = 0
        for i in range(matrix.shape[0]):
            denum += matrix[i, 1]
            for j in range(matrix.shape[0]):
                if matrix[i, 0] != 0 and matrix[j, 0] != 0:
                    num += ((matrix[i, 0] + matrix[j, 0]) * (i - j) ** 2) / n
        if denum == 0:
            return 0
        else:
            return num / denum

    def _calc_features(self, matrix):
        """Calculate all texture features for a given NGTDM matrix."""
        return {name: getattr(self, f"calc_{name}")(matrix) for name in FEATURES}

    def calc_2d_ngtdm_features(self):

        number_of_slices = self.ngtd_2d_matrices.shape[0]
        weights = []

        for i in range(number_of_slices):
            ngtdm_slice = self.ngtd_2d_matrices[i]
            weight = 1
            if self.slice_weight:
                weight = self.slice_no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
            weights.append(weight)

            feats = self._calc_features(ngtdm_slice)
            for name, value in feats.items():
                self.feature_lists[name].append(value)

        if self.slice_median and not self.slice_weight:
            for name in FEATURES:
                setattr(self, name, np.median(self.feature_lists[name]))
        elif not self.slice_median:
            for name in FEATURES:
                setattr(self, name, np.average(self.feature_lists[name], weights=weights))
        else:
            print('Weighted median not supported. Aborted!')
            return

    def calc_2_5d_ngtdm_features(self):

        ngtdm_merged = np.sum(self.ngtd_2d_matrices, axis=0)
        feats = self._calc_features(ngtdm_merged)
        for name, value in feats.items():
            setattr(self, name, value)

    def calc_3d_ngtdm_features(self):

        feats = self._calc_features(self.ngtd_3d_matrix)
        for name, value in feats.items():
            setattr(self, name, value)
