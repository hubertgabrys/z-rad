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


class IntensityVolumeHistogramFeatures:
    def __init__(self, array, min_intensity, max_intensity, discr=1):
        # Flatten array and remove NaN values
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.valid_values = array.ravel()[~np.isnan(array.ravel())]
        # Create a discretized list of intensities using the given step size
        self.intensities = np.arange(min_intensity, max_intensity + discr, discr)
        self.fractional_volumes = np.zeros(len(self.intensities))
        self.intensity_fractions = np.zeros(len(self.intensities))
        # Copy the discretized intensities (optional, kept for clarity)
        self.intensity = np.copy(self.intensities)

        self._fractions()

    def _fractions(self):
        # Calculate fractions for each discrete intensity value.
        for idx, intensity_value in enumerate(self.intensities):
            # Calculate fractional volume (νi): fraction of values with intensity >= intensity_value
            self.fractional_volumes[idx] = 1 - np.sum(self.valid_values < intensity_value) / len(self.valid_values)
            # Calculate intensity fraction (γi): relative position of intensity_value in the intensity range
            self.intensity_fractions[idx] = (intensity_value - self.min_intensity) / (self.max_intensity - self.min_intensity)

    def calc_volume_at_intensity_fraction(self, x):
        valid_indices = np.where(self.intensity_fractions > x / 100)
        return np.max(self.fractional_volumes[valid_indices])

    def calc_intensity_at_volume_fraction(self, x):
        return np.min(self.intensity[self.fractional_volumes <= x / 100])

    def calc_volume_fraction_diff_intensity_fractions(self):
        return self.calc_volume_at_intensity_fraction(10) - self.calc_volume_at_intensity_fraction(90)

    def calc_intensity_fraction_diff_volume_fractions(self):
        return self.calc_intensity_at_volume_fraction(10) - self.calc_intensity_at_volume_fraction(90)


