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


class MorphologicalFeatures:
    def __init__(self,  # image,
                 mask, spacing):

        self.spacing = spacing
        # self.array_image = image
        self.array_mask = mask
        self.unit_vol = self.spacing[0] * self.spacing[1] * self.spacing[2]

        # ---------mech-----------
        self.mesh_verts = None
        self.mesh_faces = None
        self.mesh_normals = None
        self.mesh_values = None
        self.vol_mesh = None  # 3.1.1
        self.vol_count = None  # 3.1.2
        self.area_mesh = None  # 3.1.3
        self.surf_to_vol_ratio = None  # 3.1.4
        self.compactness_1 = None  # 3.1.5
        self.compactness_2 = None  # 3.1.6
        self.spherical_disproportion = None  # 3.1.7
        self.sphericity = None  # 3.1.8
        self.asphericity = None  # 3.1.9
        # ------------------------------
        self.centre_of_shift = None  # 3.1.10
        # -------------------------------
        self.conv_hull = None
        self.max_diameter = None  # 3.1.11

        # ------------PCA based---------
        self.pca_eigenvalues = None
        self.major_axis_len = None  # 3.1.12
        self.minor_axis_len = None  # 3.1.13
        self.least_axis_len = None  # 3.1.14
        self.elongation = None  # 3.1.15
        self.flatness = None  # 3.1.16

        # ----axis-aligned bounding box----
        self.vol_density_aabb = None  # 3.1.17
        self.area_density_aabb = None  # 3.1.18
        # -------------------------------------
        # 3.1.19 and 3.1.20 no cross-center validation
        # ------------AEE-----------------------
        self.vol_density_aee = None  # 3.1.21
        self.area_density_aee = None  # 3.1.22
        # -----------------------------------
        # 3.1.23 and 3.1.24 no cross-center validation
        # --------convex hull based-------------
        self.vol_density_ch = None  # 3.1.25
        self.area_density_ch = None  # 3.1.26
        # --------------------------------------
        self.integrated_intensity = None  # 3.1.27
        # --------------------------------------
        self.moran_i = None
        self.geary_c = None

    def calc_mesh(self):
        self.mesh_verts, self.mesh_faces, self.mesh_normals, self.mesh_values = measure.marching_cubes(self.array_mask,
                                                                                                       level=0.5)
        self.mesh_verts = self.mesh_verts * self.spacing

    def calc_vol_and_area_mesh(self):
        faces = np.asarray(self.mesh_faces)  # Ensure it's an array
        verts = np.asarray(self.mesh_verts)  # Ensure it's an array

        # Get vertices corresponding to each face (Nx3x3 array)
        a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]

        # Compute cross products
        cross_bc = np.cross(b, c)  # Cross product for volume
        cross_ba_ca = np.cross(b - a, c - a)  # Cross product for area

        # Compute volume using vectorized operations
        self.vol_mesh = abs(np.einsum('ij,ij->i', a, cross_bc).sum() / 6)

        # Compute area using vectorized operations
        self.area_mesh = np.linalg.norm(cross_ba_ca, axis=1).sum() / 2

    def calc_vol_count(self):
        self.vol_count = np.sum(self.array_mask) * self.unit_vol

    def calc_surf_to_vol_ratio(self):
        self.surf_to_vol_ratio = self.area_mesh / self.vol_mesh

    def calc_compactness_1(self):
        self.compactness_1 = self.vol_mesh / (np.pi ** (1 / 2) * self.area_mesh ** (3 / 2))

    def calc_compactness_2(self):
        self.compactness_2 = 36 * np.pi * (self.vol_mesh ** 2 / self.area_mesh ** 3)

    def calc_spherical_disproportion(self):
        self.spherical_disproportion = self.area_mesh / (36 * np.pi * self.vol_mesh ** 2) ** (1 / 3)

    def calc_sphericity(self):
        self.sphericity = (36 * np.pi * self.vol_mesh ** 2) ** (1 / 3) / self.area_mesh

    def calc_asphericity(self):
        self.asphericity = (self.area_mesh ** 3 / (36 * np.pi * self.vol_mesh ** 2)) ** (1 / 3) - 1

    def calc_centre_of_shift(self, image_array):
        dx, dy, dz = self.spacing
        morph_voxels = np.argwhere(self.array_mask)
        morph_voxels_scaled = morph_voxels * [dx, dy, dz]
        com_geom = np.mean(morph_voxels_scaled, axis=0)

        # Indices of voxels in the intensity mask and their corresponding intensities
        intensity_voxels = np.argwhere(~np.isnan(image_array))
        intensities = image_array[intensity_voxels[:, 0], intensity_voxels[:, 1], intensity_voxels[:, 2]]
        # Scale voxel positions by their dimensions
        intensity_voxels_scaled = intensity_voxels * [dx, dy, dz]
        # Calculate intensity-weighted center of mass
        com_gl = np.average(intensity_voxels_scaled, axis=0, weights=intensities)

        self.centre_of_shift = np.linalg.norm(com_geom - com_gl)

    def calc_convex_hull(self):
        self.conv_hull = ConvexHull(self.mesh_verts)

    def calc_max_diameter(self):
        # Extract the vertices from the convex hull.
        hull_verts = self.mesh_verts[self.conv_hull.vertices]

        # If there are fewer than 2 vertices, the diameter is zero.
        if hull_verts.shape[0] < 2:
            self.max_diameter = 0
        else:
            # Compute all pairwise distances between the convex hull vertices.
            # pdist returns a 1D array of distances.
            self.max_diameter = np.max(pdist(hull_verts))

    def calc_pca(self):

        voxel_indices = np.argwhere(self.array_mask == 1)

        # Convert voxel indices to float to allow scaling
        scaled_voxel_indices = voxel_indices.astype(np.float64)

        # Scale the voxel indices according to voxel dimensions
        scaled_voxel_indices *= self.spacing

        # Perform PCA on the scaled indices
        pca = PCA(n_components=3)
        pca.fit(scaled_voxel_indices)

        # Extract the eigenvalues
        self.pca_eigenvalues = pca.explained_variance_

    def calc_major_minor_least_axes_len(self):
        self.major_axis_len = 4 * np.sqrt(self.pca_eigenvalues[0])
        self.minor_axis_len = 4 * np.sqrt(self.pca_eigenvalues[1])
        self.least_axis_len = 4 * np.sqrt(self.pca_eigenvalues[2])

    def calc_elongation(self):
        self.elongation = np.sqrt(self.pca_eigenvalues[1] / self.pca_eigenvalues[0])

    def calc_flatness(self):
        self.flatness = np.sqrt(self.pca_eigenvalues[2] / self.pca_eigenvalues[0])

    def calc_vol_and_area_densities_aabb(self):
        x_dim, y_dim, z_dim = self.spacing
        # Determine the AABB of the ROI
        x_coords, y_coords, z_coords = np.where(self.array_mask == 1)
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        z_min, z_max = z_coords.min(), z_coords.max()

        # Calculate the dimensions of the AABB
        aabb_x_dim = (x_max - x_min + 1) * x_dim
        aabb_y_dim = (y_max - y_min + 1) * y_dim
        aabb_z_dim = (z_max - z_min + 1) * z_dim

        # Calculate the volume of the AABB
        aabb_volume = aabb_x_dim * aabb_y_dim * aabb_z_dim
        self.vol_density_aabb = self.vol_mesh / aabb_volume

        # Calculate the area of the AABB
        aabb_surface_area = 2 * (aabb_x_dim * aabb_y_dim + aabb_x_dim * aabb_z_dim + aabb_y_dim * aabb_z_dim)
        self.area_density_aabb = self.area_mesh / aabb_surface_area

    def calc_vol_density_aee(self):
        self.vol_density_aee = (8 * 3 * self.vol_mesh) / (
                    4 * np.pi * self.major_axis_len * self.minor_axis_len * self.least_axis_len)

    def calc_area_density_aee(self):
        a = self.major_axis_len / 2
        b = self.minor_axis_len / 2
        c = self.least_axis_len / 2

        alpha = np.sqrt(1 - (b ** 2 / a ** 2))
        beta = np.sqrt(1 - (c ** 2 / a ** 2))
        sum_series = 0
        max_nu = 20  # Def by IBSI
        for nu in range(max_nu + 1):
            p_nu = legendre(nu)
            sum_series += ((alpha * beta) ** nu / (1 - (4 * nu ** 2))) * p_nu(
                (alpha ** 2 + beta ** 2) / (2 * alpha * beta))

        area_aee = 4 * np.pi * a * b * sum_series
        self.area_density_aee = self.area_mesh / area_aee

    def calc_vol_density_ch(self):
        self.vol_density_ch = self.vol_mesh / self.conv_hull.volume

    def calc_area_density_ch(self):
        self.area_density_ch = self.area_mesh / self.conv_hull.area

    def calc_integrated_intensity(self, image_array):
        self.integrated_intensity = np.nanmean(image_array) * self.vol_mesh

    def calc_moran_i(self, image_array):

        # Get indices of voxels in the ROI intensity mask
        indices = np.argwhere(self.array_mask)
        # Scale indices by voxel spacing to obtain physical coordinates
        scaled_indices = indices * self.spacing

        # Extract intensity values at these voxel indices
        intensities = image_array[indices[:, 0], indices[:, 1], indices[:, 2]]

        # Filter out any NaN intensity values
        valid = ~np.isnan(intensities)
        if np.sum(valid) < 2:
            self.moran_i = np.nan
            return

        scaled_indices = scaled_indices[valid]
        intensities = intensities[valid]

        # Total number of valid voxels
        N = len(intensities)
        # Mean intensity
        mu = np.mean(intensities)

        # Compute pairwise distances between voxel coordinates
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(scaled_indices))

        # Create a weight matrix: weight = 1/distance for nonzero distances, 0 otherwise
        weights = np.zeros_like(distances)
        nonzero_mask = distances > 0
        weights[nonzero_mask] = 1.0 / distances[nonzero_mask]
        # weights = np.where(distances > 0, 1.0 / distances, 0)

        # Sum of all weights (excluding self-pairs, since diagonal is 0)
        S0 = np.sum(weights)

        # Compute the numerator: sum_{i≠j} w_{ij} (X_i - μ)(X_j - μ)
        diff = intensities - mu
        diff_outer = np.outer(diff, diff)
        numerator = np.sum(weights * diff_outer)

        # Compute the denominator: sum_{k} (X_k - μ)^2
        denominator = np.sum(diff ** 2)

        # Calculate Moran's I
        self.moran_i = (N / S0) * (numerator / denominator)

    def calc_geary_c(self, image_array):

        # Extract voxel indices from the ROI mask and scale them by voxel spacing
        indices = np.argwhere(self.array_mask)
        scaled_indices = indices * self.spacing

        # Retrieve intensity values for these voxels from the image_array
        intensities = image_array[indices[:, 0], indices[:, 1], indices[:, 2]]

        # Exclude NaN values from the intensity data
        valid = ~np.isnan(intensities)
        if np.sum(valid) < 2:
            self.geary_c = np.nan
            return

        scaled_indices = scaled_indices[valid]
        intensities = intensities[valid]

        # Total number of valid voxels and mean intensity
        N = len(intensities)
        mu = np.mean(intensities)

        # Compute pairwise Euclidean distances between voxel coordinates
        distances = squareform(pdist(scaled_indices))

        # Define weights as the inverse of distance (with 0 weight for zero distances)
        weights = np.zeros_like(distances)
        nonzero_mask = distances > 0
        weights[nonzero_mask] = 1.0 / distances[nonzero_mask]
        #weights = np.where(distances > 0, 1.0 / distances, 0)

        # Sum of all weights
        S0 = np.sum(weights)

        # Calculate the numerator: sum_{i≠j} w_{ij} (X_i - X_j)^2
        diff_matrix = np.subtract.outer(intensities, intensities)
        squared_diff = diff_matrix ** 2
        numerator = np.sum(weights * squared_diff)

        # Calculate the denominator: sum_{i} (X_i - μ)^2
        denominator = np.sum((intensities - mu) ** 2)

        # Compute Geary's C measure
        self.geary_c = ((N - 1) / (2 * S0)) * (numerator / denominator)


