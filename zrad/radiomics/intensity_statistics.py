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


class IntensityBasedStatFeatures:
    def __init__(self):  # , image):
        # self.spacing = spacing
        # self.array_image = image
        # self.array_mask = mask
        # ----------------------
        self.array_image_2 = None
        self.mean_intensity = None  # 3.3.1
        self.intensity_variance = None  # 3.3.2
        self.intensity_skewness = None  # 3.3.3
        self.intensity_kurtosis = None  # 3.3.4
        self.median_intensity = None  # 3.3.5
        self.min_intensity = None  # 3.3.6
        self.intensity_10th_percentile = None  # 3.3.7
        self.intensity_90th_percentile = None  # 3.3.8
        self.max_intensity = None  # 3.3.9
        self.intensity_iqr = None  # 3.3.10
        self.intensity_range = None  # 3.3.11
        self.intensity_based_mean_abs_deviation = None  # 3.3.12
        self.intensity_based_robust_mean_abs_deviation = None  # 3.3.13
        self.intensity_based_median_abs_deviation = None  # 3.3.14
        self.intensity_based_variation_coef = None  # 3.3.15
        self.intensity_based_quartile_coef_dispersion = None  # 3.3.16
        self.intensity_based_energy = None  # 3.3.17
        self.root_mean_square_intensity = None  # 3.3.18
        # ----------------------------------------------
        self.mean_discret_intensity = None  # 3.4.1
        self.discret_intensity_variance = None  # 3.4.2
        self.discret_intensity_skewness = None  # 3.4.3
        self.discret_intensity_kurtosis = None  # 3.4.4
        self.median_discret_intensity = None  # 3.4.5
        self.minimum_discret_intensity = None  # 3.4.6
        self.discret_intensity_10th_percentile = None  # 3.4.7
        self.discret_intensity_90th_percentile = None  # 3.4.8
        self.maximum_discret_intensity = None  # 3.4.9
        self.intensity_hist_mode = None  # 3.4.10
        self.discret_intensity_iqr = None  # 3.4.11
        self.discret_intensity_range = None  # 3.4.12
        self.intensity_hist_mean_abs_deviation = None  # 3.4.13
        self.intensity_hist_robust_mean_abs_deviation = None  # 3.4.14
        self.intensity_hist_median_abs_deviation = None  # 3.4.15
        self.intensity_hist_variation_coef = None  # 3.4.16
        self.intensity_hist_quartile_coef_dispersion = None  # 3.4.17
        self.discret_intensity_entropy = None  # 3.4.18
        self.discret_intensity_uniformity = None  # 3.4.19
        self.max_hist_gradient = None  # 3.4.20
        self.max_hist_gradient_intensity = None  # .3.4.21
        self.min_hist_gradient = None  # 3.4.22
        self.min_hist_gradient_intensity = None  # .3.4.23

    def calc_mean_intensity(self, array):  # 3.3.1, 3.4.1
        self.mean_intensity = np.nanmean(array)

    def calc_intensity_variance(self, array):  # 3.3.2, 3.4.2
        self.intensity_variance = np.nanstd(array) ** 2

    def calc_intensity_skewness(self, array):  # 3.3.3, 3.4.3
        self.intensity_skewness = skew(array, axis=None, nan_policy='omit')

    def calc_intensity_kurtosis(self, array):  # 3.3.4, 3.4.4
        self.intensity_kurtosis = kurtosis(array, axis=None, nan_policy='omit')

    def calc_median_intensity(self, array):  # 3.3.5, 3.4.5
        self.median_intensity = np.nanmedian(array)

    def calc_min_intensity(self, array):  # 3.3.6, 3.4.6
        self.min_intensity = np.nanmin(array)

    def calc_intensity_10th_percentile(self, array):  # 3.3.7, 3.4.7
        self.intensity_10th_percentile = np.nanpercentile(array, 10)

    def calc_intensity_90th_percentile(self, array):  # 3.3.8, 3.4.8
        self.intensity_90th_percentile = np.nanpercentile(array, 90)

    def calc_max_intensity(self, array):  # 3.3.9, 3.4.9
        self.max_intensity = np.nanmax(array)

    def calc_intensity_iqr(self, array):  # 3.3.10, 3.4.11
        self.intensity_iqr = iqr(array, nan_policy='omit')

    def calc_intensity_range(self, array):  # 3.3.11, 3.4.12
        self.intensity_range = np.nanmax(array) - np.nanmin(array)

    def calc_intensity_based_mean_abs_deviation(self, array):  # .3.3.12, 3.4.13
        self.intensity_based_mean_abs_deviation = np.nanmean(np.absolute(array - np.nanmean(array)))

    def calc_intensity_based_robust_mean_abs_deviation(self, array):  # 3.3.13, 3.4.14
        self.array_image_2 = array.copy()
        p10 = np.nanpercentile(self.array_image_2, 10)
        p90 = np.nanpercentile(self.array_image_2, 90)
        ind = np.where((self.array_image_2 < p10) | (self.array_image_2 > p90))
        self.array_image_2[ind] = np.nan
        self.intensity_based_robust_mean_abs_deviation = np.nanmean(
            np.absolute(self.array_image_2 - np.nanmean(self.array_image_2)))

    def calc_intensity_based_median_abs_deviation(self, array):  # 3.3.14, 3.4.15
        self.intensity_based_median_abs_deviation = np.nanmean(np.absolute(array - np.nanmedian(array)))

    def calc_intensity_based_variation_coef(self, array):  # 3.3.15, 3.4.16
        denum = np.nanmean(array)
        if denum == 0:
            self.intensity_based_variation_coef = 1_000_000
        else:
            self.intensity_based_variation_coef = np.nanstd(array) / np.nanmean(array)

    def calc_intensity_based_quartile_coef_dispersion(self, array):  # 3.3.16, 3.4.17
        p25 = np.nanpercentile(array, 25)
        p75 = np.nanpercentile(array, 75)
        denum = (p75 + p25)
        if denum == 0:
            self.intensity_based_quartile_coef_dispersion = 1_000_000
        else:
            self.intensity_based_quartile_coef_dispersion = (p75 - p25) / denum

    def calc_intensity_based_energy(self, array):  # 3.3.17
        self.intensity_based_energy = np.nansum(array ** 2)

    def calc_root_mean_square_intensity(self, array):  # .3.3.18
        self.root_mean_square_intensity = np.sqrt(np.nanmean(array ** 2))

    def calc_discretised_intensity_mode(self, array):  # 3.4.10
        values, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        max_count_index = np.argmax(counts)
        self.intensity_hist_mode = values[max_count_index]

    def calc_discretised_intensity_entropy(self, array):  # 3.4.18
        values, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        p = counts / np.sum(counts)
        self.discret_intensity_entropy = (-1) * np.sum(p * np.log2(p))

    def calc_discretised_intensity_uniformity(self, array):  # 3.4.19
        values, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        p = counts / np.sum(counts)
        self.discret_intensity_uniformity = np.sum(p * p)

    def calc_max_hist_gradient(self, array):  # 3.4.20
        values, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        self.max_hist_gradient = np.max(np.gradient(counts))

    def calc_max_hist_gradient_intensity(self, array):  # 3.4.21
        values, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        self.max_hist_gradient_intensity = values[np.argmax(np.gradient(counts))]

    def calc_min_hist_gradient(self, array):  # 3.4.22
        values, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        self.min_hist_gradient = np.min(np.gradient(counts))

    def calc_min_hist_gradient_intensity(self, array):  # 3.4.23
        values, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        self.min_hist_gradient_intensity = values[np.argmin(np.gradient(counts))]


