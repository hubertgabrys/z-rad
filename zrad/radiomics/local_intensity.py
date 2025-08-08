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


class LocalIntensityFeatures:

    def __init__(self, image, masked_image, spacing):

        self.array_image = image
        self.array_masked_image = masked_image
        self.spacing = spacing

        # ---------mech-----------
        self.local_intensity_peak = None  # 3.2.1
        self.global_intensity_peak = None  # 3.2.2

    def calc_local_intensity_peak(self):  # 3.2.1

        radius_mm = 6.2
        # Find the indices of the maximum intensity voxels
        max_intensity = np.nanmax(self.array_masked_image)
        max_voxels = np.argwhere(self.array_masked_image == max_intensity)
        highest_peak = []
        for voxel in max_voxels:
            distances = np.sqrt(
                ((np.indices(self.array_masked_image.shape).T * self.spacing - voxel * self.spacing) ** 2).sum(axis=3))
            # Create a mask for selected voxels within the sphere radius
            sphere_mask = (distances <= radius_mm)
            # Ensure the mask is applied in all three dimensions
            selected_voxels = self.array_image[sphere_mask.T]

            # Calculate the mean intensity of the selected voxels
            mean_intensity = np.mean(selected_voxels)

            # Update the highest peak if this one is higher
            highest_peak.append(mean_intensity)

        self.local_intensity_peak = max(highest_peak)

    def calc_global_intensity_peak(self):  # 3.2.2
        """
        Calculate the global intensity peak feature.

        The global intensity peak is defined as the highest mean intensity computed over
        a spherical neighborhood (1 cm³, radius ≈ 6.2 mm) centered at every voxel in the ROI.
        This is efficiently implemented by convolving the image with a normalized spherical mean filter.
        """
        radius_mm = 6.2
        spacing = np.array(self.spacing)  # Expected order: (z, y, x)

        # Determine the half-size of the filter in voxels along each dimension
        half_sizes = np.ceil(radius_mm / spacing).astype(int)

        # Build coordinate ranges for the spherical filter kernel.
        # The kernel will span from -half_size to +half_size in each dimension.
        grid_ranges = [np.arange(-hs, hs + 1) for hs in half_sizes]

        # Create the coordinate grid; note that the outputs correspond to (z, y, x)
        zz, yy, xx = np.meshgrid(grid_ranges[0], grid_ranges[1], grid_ranges[2], indexing='ij')

        # Compute Euclidean distances from the kernel center (in mm)
        # Make sure to multiply z-coordinates by spacing[0], y by spacing[1], and x by spacing[2]
        distances = np.sqrt((zz * spacing[0]) ** 2 +
                            (yy * spacing[1]) ** 2 +
                            (xx * spacing[2]) ** 2)

        # Create a spherical mask: 1 inside the sphere, 0 outside.
        spherical_mask = distances <= radius_mm

        # Normalize the kernel to compute the mean intensity (i.e., spherical mean filter).
        N_s = np.sum(spherical_mask)
        kernel = spherical_mask.astype(float) / N_s

        # Convolve the full image with the spherical mean filter.
        # The result is a map of local mean intensities.
        local_means = convolve(self.array_image, kernel, mode='constant', cval=0.0)

        # Restrict to the ROI: consider only voxels where the masked image is not NaN.
        roi_mask = ~np.isnan(self.array_masked_image)

        # The global intensity peak is the maximum local mean within the ROI.
        self.global_intensity_peak = np.max(local_means[roi_mask])


