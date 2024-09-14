import sys

import numpy as np

from .image import Image
from .radiomics_definitions import MorphologicalFeatures, LocalIntensityFeatures, IntensityBasedStatFeatures, \
    GLCM, GLRLM_GLSZM_GLDZM_NGLDM, NGTDM
from .toolbox_logic import handle_uncaught_exception
from ..logic.exceptions import DataStructureError

sys.excepthook = handle_uncaught_exception


def _get_bounding_box(arr):
    """
    Crops the input 3D array so that no face of the array is entirely NaN.

    Parameters:
    arr (np.ndarray): Input 3D array to crop.

    Returns:
    np.ndarray: Cropped 3D array.
    """
    # Create a mask of where the non-NaN values are located
    mask = ~np.isnan(arr)

    # Find the min and max indices along each dimension where non-NaN values exist
    z_non_nan = np.where(mask.any(axis=(1, 2)))[0]
    y_non_nan = np.where(mask.any(axis=(0, 2)))[0]
    x_non_nan = np.where(mask.any(axis=(0, 1)))[0]

    # Determine the bounds for cropping
    zmin, zmax = z_non_nan[0], z_non_nan[-1] + 1
    ymin, ymax = y_non_nan[0], y_non_nan[-1] + 1
    xmin, xmax = x_non_nan[0], x_non_nan[-1] + 1

    # Crop the array using the determined bounds
    return arr[zmin:zmax, ymin:ymax, xmin:xmax]


class Radiomics:

    def __init__(self,
                 aggr_dim='3D', aggr_method='AVER',
                 intensity_range=None, outlier_range=None,
                 number_of_bins=None, bin_size=None,
                 slice_weighting=False, slice_median=False):
        self.patient_morphological_mask = None
        self.patient_intensity_mask = None

        if slice_weighting and slice_median:
            raise ValueError('Only one slice median averaging is not supported with weighting strategy.')

        else:
            self.slice_weighting = slice_weighting
            self.slice_median = slice_median

        self.calc_intensity_mask = False
        if intensity_range is not None:
            self.calc_intensity_mask = True
            self.intensity_range = intensity_range
            self.discret_min_val = intensity_range[0]

        self.calc_outlier_mask = False
        if str(outlier_range).strip().replace('.', '').isdigit():
            self.calc_outlier_mask = True
            self.outlier_range = outlier_range

        self.calc_discr_bin_number = False
        self.calc_discr_bin_size = False

        if number_of_bins is not None:
            self.calc_discr_bin_number = True
            self.bin_number = number_of_bins

        if bin_size is not None:
            self.calc_discr_bin_size = True
            self.bin_size = bin_size

        if aggr_dim in ['2D', '2.5D', '3D']:
            self.aggr_dim = aggr_dim
        else:
            raise ValueError(f"Wrong aggregation dim {aggr_dim}. Available '2D', '2.5D', and '3D'.")

        if aggr_method in ['MERG', 'AVER', 'SLICE_MERG', 'DIR_MERG']:
            self.aggr_method = aggr_method
        else:
            raise ValueError(f"Wrong aggregation dim {aggr_method}. "
                             "Available 'MERG', 'AVER', 'SLICE_MERG', and 'DIR_MERG'.")

        self.patient_folder = None
        self.patient_number = None

        self.columns = [
            'vol_mesh', 'vol_count', 'area_mesh', 'surf_to_vol_ratio', 'compactness_1', 'compactness_2',
            'spherical_disproportion', 'sphericity', 'asphericity', 'centre_of_shift', 'max_diameter', 'major_axis_len',
            'minor_axis_len', 'least_axis_len', 'elongation', 'flatness', 'vol_density_aabb', 'area_density_aabb',
            'vol_density_aee', 'area_density_aee', 'vol_density_ch', 'area_density_ch', 'integrated_intensity',
            'local_intensity_peak', 'mean_intensity', 'intensity_variance', 'intensity_skewness', 'intensity_kurtosis',
            'median_intensity', 'min_intensity', 'intensity_10th_percentile', 'intensity_90th_percentile',
            'max_intensity', 'intensity_iqr', 'intensity_range', 'intensity_based_mean_abs_deviation',
            'intensity_based_robust_mean_abs_deviation', 'intensity_based_median_abs_deviation',
            'intensity_based_variation_coef', 'intensity_based_quartile_coef_dispersion', 'intensity_based_energy',
            'root_mean_square_intensity',
            'discr_mean_intensity', 'discr_intensity_varaince', 'discr_intensity_skewness', 'discr_intensity_kurtosis',
            'discr_median_intensity', 'discr_min_intensity', 'discr_intensity_10th_percentile',
            'discr_intensity_90th_percentile', 'discr_max_intensity', 'discr_intensity_hist_mode',
            'discr_intensity_iqr', 'discr_intensity_range', 'discr_intensity_based_mean_abs_deviation',
            'discr_intensity_based_robust_mean_abs_deviation', 'discr_intensity_based_median_abs_deviation',
            'discr_intensity_based_variation_coef', 'discr_intensity_based_quartile_coef_dispersion',
            'discr_intensity_entropy', 'discr_intensity_uniformity', 'discr_max_hist_gradient',
            'discr_max_hist_gradient_intensity', 'discr_min_hist_gradient', 'discr_min_hist_gradient_intensity',
            'glcm_joint_max', 'glcm_joint_average', 'glcm_joint_var', 'glcm_joint_entropy', 'glcm_dif_average',
            'glcm_dif_var', 'glcm_dif_entropy', 'glcm_sum_average', 'glcm_sum_var', 'glcm_sum_entropy',
            'glcm_ang_second_moment', 'glcm_contrast', 'glcm_dissimilarity', 'glcm_inv_diff', 'glcm_norm_inv_diff',
            'glcm_inv_diff_moment', 'glcm_norm_inv_diff_moment', 'glcm_inv_variance', 'glcm_cor', 'glcm_autocor',
            'glcm_cluster_tendency', 'glcm_cluster_shade', 'glcm_cluster_prominence', 'glcm_inf_cor_1',
            'glcm_inf_cor_2',
            'glrlm_short_runs_emphasis', 'glrlm_long_runs_emphasis', 'glrlm_low_grey_level_run_emphasis',
            'glrlm_high_gr_lvl_emphasis', 'glrlm_short_run_low_gr_lvl_emphasis', 'glrlm_short_run_high_gr_lvl_emphasis',
            'glrlm_long_run_low_gr_lvl_emphasis', 'glrlm_long_run_high_gr_lvl_emphasis', 'glrlm_non_uniformity',
            'glrlm_norm_non_uniformity', 'glrlm_run_length_non_uniformity', 'glrlm_norm_run_length_non_uniformity',
            'glrlm_percentage', 'glrlm_gr_lvl_var', 'glrlm_run_length_var', 'glrlm_run_entropy',
            'glszm_small_zone_emphasis', 'glszm_large_zone_emphasis', 'glszm_low_grey_level_zone_emphasis',
            'glszm_high_gr_lvl_zone_emphasis', 'glszm_small_zone_low_gr_lvl_emphasis',
            'glszm_small_zone_high_gr_lvl_emphasis', 'glszm_large_zone_low_gr_lvl_emphasis',
            'glszm_large_zone_high_gr_lvl_emphasis', 'glszm_non_uniformity', 'glszm_norm_non_uniformity',
            'glszm_zone_non_uniformity', 'glszm_norm_zone_non_uniformity', 'glszm_percentage', 'glszm_gr_lvl_var',
            'glszm_zone_size_var', 'glszm_entropy',
            'gldzm_small_dist_emphasis', 'gldzm_large_dist_emphasis', 'gldzm_low_grey_level_zone_emphasis',
            'gldzm_high_gr_lvl_zone_emphasis', 'gldzm_small_dist_low_gr_lvl_emphasis',
            'gldzm_small_dist_high_gr_lvl_emphasis', 'gldzm_large_dist_low_gr_lvl_emphasis',
            'gldzm_large_dist_high_gr_lvl_emphasis', 'gldzm_non_uniformity', 'gldzm_norm_non_uniformity',
            'gldzm_zone_dist_non_uniformity', 'gldzm_norm_zone_dist_non_uniformity', 'gldzm_percentage',
            'gldzm_gr_lvl_var', 'gldzm_zone_dist_var', 'gldzm_entropy',
            'ngtdm_coarseness', 'ngtdm_contrast', 'ngtdm_busyness', 'ngtdm_complexity', 'ngtdm_strength',
            'ngldm_low_depend_emphasis', 'ngldm_high_depend_emphasis', 'ngldm_low_gr_lvl_emphasis',
            'ngldm_high_gr_lvl_emphasis', 'ngldm_low_depend_low_gr_lvl_emphasis',
            'ngldm_low_depend_high_gr_lvl_emphasis', 'ngldm_high_depend_low_gr_lvl_emphasis',
            'ngldm_high_depend_high_gr_lvl_emphasis', 'ngldm_non_uniformity', 'ngldm_norm_non_uniformity',
            'ngldm_depend_count_non_uniformity', 'ngldm_norm_depend_count_non_uniformity', 'ngldm_percentage',
            'ngldm_gr_lvl_var', 'ngldm_depend_count_var', 'ngldm_entropy', 'ngldm_energy'
        ]

    def extract_features(self, image, mask, filtered_image=None):
        self.pat_binned_masked_image = {}
        self.patient_morf_features_list = []
        self.patient_local_intensity_features_list = []
        self.intensity_features_list = []
        self.discr_intensity_features_list = []
        self.glcm_features_list = []
        self.glrlm_features_list = []
        self.glszm_features_list = []
        self.gldzm_features_list = []
        self.ngtdm_features_list = []
        self.ngldm_features_list = []

        self.orig_patient_image = image
        if filtered_image:
            self.patient_image = filtered_image
        else:
            self.patient_image = image

        self.patient_morphological_mask = mask.copy()
        self.patient_morphological_mask.array = self.patient_morphological_mask.array.astype(np.int8)

        self.patient_intensity_mask = mask.copy()
        self.patient_intensity_mask.array = np.where(self.patient_intensity_mask.array > 0, self.patient_image.array, np.nan)

        # extract features
        self._calc_mask_intensity_features()
        self._calc_mask_morphological_features()
        self._calc_discretized_intensity_features()
        self._calc_texture_features()
        self._validate_mask()

        # compile features
        all_features_list = [self.patient_morf_features_list, self.patient_local_intensity_features_list,
                             self.intensity_features_list, self.discr_intensity_features_list,
                             self.glcm_features_list,
                             self.glrlm_features_list, self.glszm_features_list,
                             self.gldzm_features_list, self.ngtdm_features_list, self.ngldm_features_list]
        all_features_list_flat = [item for sublist in all_features_list for item in sublist[0]]
        self.features_ = dict(zip(self.columns, all_features_list_flat))

    def _validate_mask(self):
        """
        Validates the intensity mask for a patient by checking the bounding box dimensions
        and the number of valid voxels within the mask, with criteria differing based on
        the aggregation dimension (2D/2.5D or 3D).

        Returns:
            self.patient_intensity_mask: The validated intensity mask if it meets the
                                         criteria for minimum bounding box size and
                                         valid voxel count.
            None: If the mask does not meet the criteria, `None` is returned and
                  processing for this mask is skipped.

        Criteria for validation:
            - For 3D aggregation:
                - The smallest dimension of the 3D bounding box must be greater than `min_box_size`.
                - The number of valid (non-NaN) voxels in the bounding box must be greater
                  than `min_voxel_number_3d`.
            - For 2D or 2.5D aggregation:
                - The smallest dimension of the 2D slice (ignoring the third dimension) must be greater than `min_box_size`.
                - The number of valid (non-NaN) voxels in the bounding box must be greater
                  than `min_voxel_number_2d`.

        Logs:
            - A message is logged if the mask is skipped due to not meeting the
              minimum bounding box size requirement.
            - A message is logged if the mask is skipped due to insufficient valid voxels.
            - The log messages include the patient number and mask name for easier identification.
        """
        # Calculate the bounding box around the intensity mask and determine its shape.
        bounding_box = _get_bounding_box(self.patient_intensity_mask.array)
        bounding_box_shape = bounding_box.shape
        no_valid_voxels = np.sum(~np.isnan(bounding_box))

        # Define the minimum size and voxel count requirements for validation.
        min_box_size = 3
        min_voxel_number_3d = 27  # For 3D: minimum 3x3x3 volume
        min_voxel_number_2d = 9  # For 2D/2.5D: minimum 3x3 area

        # Check the bounding box size and the number of voxels based on the aggregation dimension.
        if self.aggr_dim == '3D':
            # Check if the bounding box size meets the minimum requirement for 3D.
            if min(bounding_box_shape) < min_box_size:
                raise DataStructureError(f'The minimum size of the bounding box must be > {min_box_size}. Consider finer resampling.')
            # Check if the number of valid voxels meets the minimum requirement for 3D.
            if no_valid_voxels < min_voxel_number_3d:
                raise DataStructureError(f'The number of valid voxels must be > {min_voxel_number_3d}. Consider finer resampling.')
        else:  # For 2D or 2.5D aggregation, only consider the first two dimensions.
            if min(bounding_box_shape[:2]) < min_box_size:
                raise DataStructureError(f'The minimum size of the bounding box in the first two dimensions must be > {min_box_size}. Consider finer resampling.')
            # Check if the number of valid voxels meets the minimum requirement for 2D.
            if no_valid_voxels < min_voxel_number_2d:
                raise DataStructureError(f'The number of valid voxels must be > {min_voxel_number_2d}. Consider finer resampling.')

        # If the mask meets all criteria, return the validated intensity mask.
        return self.patient_intensity_mask

    def _calc_mask_intensity_features(self):

        if self.calc_intensity_mask:
            intensity_range_mask = np.where((self.orig_patient_image.array <= self.intensity_range[1])
                                            & (self.orig_patient_image.array >= self.intensity_range[0]),
                                            1, 0)
            self.patient_intensity_mask = Image(array=np.where((intensity_range_mask > 0)
                                                               & (~np.isnan(self.patient_intensity_mask.array)),
                                                               self.patient_intensity_mask.array, np.nan),
                                                origin=self.patient_intensity_mask.origin,
                                                spacing=self.patient_intensity_mask.spacing,
                                                direction=self.patient_intensity_mask.direction,
                                                shape=self.patient_intensity_mask.shape)
        if self.calc_outlier_mask:
            flattened_image = np.where(self.patient_morphological_mask.array > 0,
                                       self.orig_patient_image.array, np.nan).ravel()
            valid_values = flattened_image[~np.isnan(flattened_image)]
            mean = np.mean(valid_values)
            std = np.std(valid_values)
            outlier_mask = np.where((self.orig_patient_image.array <= mean + self.outlier_range * std)
                                    & (self.orig_patient_image.array >= mean - self.outlier_range * std)
                                    & (~np.isnan(self.patient_intensity_mask.array)),
                                    1, 0)

            self.patient_intensity_mask = Image(array=np.where((outlier_mask > 0)
                                                               & (~np.isnan(self.patient_intensity_mask.array)),
                                                               self.patient_intensity_mask.array, np.nan),
                                                origin=self.patient_intensity_mask.origin,
                                                spacing=self.patient_intensity_mask.spacing,
                                                direction=self.patient_intensity_mask.direction,
                                                shape=self.patient_intensity_mask.shape)

        local_intensity_features = LocalIntensityFeatures(self.patient_image.array,
                                                          self.patient_intensity_mask.array,
                                                          (self.patient_image.spacing[::-1]))

        local_intensity_features.calc_local_intensity_peak()
        self.local_intensity_features = [local_intensity_features.local_intensity_peak]
        self.patient_local_intensity_features_list.append(self.local_intensity_features)

        intensity_features = IntensityBasedStatFeatures()
        intensity_features.calc_mean_intensity(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_variance(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_skewness(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_kurtosis(self.patient_intensity_mask.array)
        intensity_features.calc_median_intensity(self.patient_intensity_mask.array)
        intensity_features.calc_min_intensity(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_10th_percentile(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_90th_percentile(self.patient_intensity_mask.array)
        intensity_features.calc_max_intensity(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_iqr(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_range(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_mean_abs_deviation(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_robust_mean_abs_deviation(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_median_abs_deviation(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_variation_coef(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_quartile_coef_dispersion(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_energy(self.patient_intensity_mask.array)
        intensity_features.calc_root_mean_square_intensity(self.patient_intensity_mask.array)

        self.intensity_based_features = [intensity_features.mean_intensity,
                                         intensity_features.intensity_variance,
                                         intensity_features.intensity_skewness,
                                         intensity_features.intensity_kurtosis,
                                         intensity_features.median_intensity,
                                         intensity_features.min_intensity,
                                         intensity_features.intensity_10th_percentile,
                                         intensity_features.intensity_90th_percentile,
                                         intensity_features.max_intensity,
                                         intensity_features.intensity_iqr,
                                         intensity_features.intensity_range,
                                         intensity_features.intensity_based_mean_abs_deviation,
                                         intensity_features.intensity_based_robust_mean_abs_deviation,
                                         intensity_features.intensity_based_median_abs_deviation,
                                         intensity_features.intensity_based_variation_coef,
                                         intensity_features.intensity_based_quartile_coef_dispersion,
                                         intensity_features.intensity_based_energy,
                                         intensity_features.root_mean_square_intensity]

        self.intensity_features_list.append(self.intensity_based_features)

    def _calc_discretized_intensity_features(self):
        if self.calc_discr_bin_size:
            if self.calc_intensity_mask:
                self.patient_intensity_mask = Image(array=np.floor(
                    (self.patient_intensity_mask.array - self.discret_min_val) / self.bin_size) + 1,
                                                         origin=self.patient_image.origin,
                                                         spacing=self.patient_image.spacing,
                                                         direction=self.patient_image.direction,
                                                         shape=self.patient_image.shape
                                                         )
            else:
                self.patient_intensity_mask = Image(array=np.floor((self.patient_intensity_mask.array - np.nanmin(
                    self.patient_intensity_mask.array)) / self.bin_size) + 1,
                                                         origin=self.patient_image.origin,
                                                         spacing=self.patient_image.spacing,
                                                         direction=self.patient_image.direction,
                                                         shape=self.patient_image.shape
                                                         )
        if self.calc_discr_bin_number:
            self.patient_intensity_mask = Image(
                array=np.where(self.patient_intensity_mask.array != np.nanmax(self.patient_intensity_mask.array),
                               np.floor(self.bin_number * (self.patient_intensity_mask.array - np.nanmin(
                                   self.patient_intensity_mask.array))
                                        / (np.nanmax(self.patient_intensity_mask.array)
                                           - np.nanmin(self.patient_intensity_mask.array))) + 1, self.bin_number),
                origin=self.patient_image.origin,
                spacing=self.patient_image.spacing,
                direction=self.patient_image.direction,
                shape=self.patient_image.shape
                )

        discr_intensity_features = IntensityBasedStatFeatures()
        discr_intensity_features.calc_mean_intensity(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_variance(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_skewness(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_kurtosis(self.patient_intensity_mask.array)
        discr_intensity_features.calc_median_intensity(self.patient_intensity_mask.array)
        discr_intensity_features.calc_min_intensity(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_10th_percentile(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_90th_percentile(self.patient_intensity_mask.array)
        discr_intensity_features.calc_max_intensity(self.patient_intensity_mask.array)
        discr_intensity_features.calc_discretised_intensity_mode(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_iqr(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_range(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_based_mean_abs_deviation(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_based_robust_mean_abs_deviation(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_based_median_abs_deviation(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_based_variation_coef(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_based_quartile_coef_dispersion(self.patient_intensity_mask.array)
        discr_intensity_features.calc_discretised_intensity_entropy(self.patient_intensity_mask.array)
        discr_intensity_features.calc_discretised_intensity_uniformity(self.patient_intensity_mask.array)
        discr_intensity_features.calc_max_hist_gradient(self.patient_intensity_mask.array)  # 3.4.20
        discr_intensity_features.calc_max_hist_gradient_intensity(self.patient_intensity_mask.array)  # 3.4.21
        discr_intensity_features.calc_min_hist_gradient(self.patient_intensity_mask.array)  # 3.4.22
        discr_intensity_features.calc_min_hist_gradient_intensity(self.patient_intensity_mask.array)  # 3.4.23

        self.discr_intensity_based_features = [discr_intensity_features.mean_intensity,
                                               discr_intensity_features.intensity_variance,
                                               discr_intensity_features.intensity_skewness,
                                               discr_intensity_features.intensity_kurtosis,
                                               discr_intensity_features.median_intensity,
                                               discr_intensity_features.min_intensity,
                                               discr_intensity_features.intensity_10th_percentile,
                                               discr_intensity_features.intensity_90th_percentile,
                                               discr_intensity_features.max_intensity,
                                               discr_intensity_features.intensity_hist_mode,
                                               discr_intensity_features.intensity_iqr,
                                               discr_intensity_features.intensity_range,
                                               discr_intensity_features.intensity_based_mean_abs_deviation,
                                               discr_intensity_features.intensity_based_robust_mean_abs_deviation,
                                               discr_intensity_features.intensity_based_median_abs_deviation,
                                               discr_intensity_features.intensity_based_variation_coef,
                                               discr_intensity_features.intensity_based_quartile_coef_dispersion,
                                               discr_intensity_features.discret_intensity_entropy,
                                               discr_intensity_features.discret_intensity_uniformity,
                                               discr_intensity_features.max_hist_gradient,
                                               discr_intensity_features.max_hist_gradient_intensity,
                                               discr_intensity_features.min_hist_gradient,
                                               discr_intensity_features.min_hist_gradient_intensity]

        self.discr_intensity_features_list.append(self.discr_intensity_based_features)

    def _calc_texture_features(self):
        glcm = GLCM(image=self.patient_intensity_mask.array.T, slice_weight=self.slice_weighting,
                    slice_median=self.slice_median)
        if self.aggr_dim == '3D':
            glcm.calc_glc_3d_matrix()
            if self.aggr_method == 'AVER':
                glcm.calc_3d_averaged_glcm_features()
            elif self.aggr_method == 'MERG':
                glcm.calc_3d_merged_glcm_features()

        elif self.aggr_dim == '2.5D' or self.aggr_dim == '2D':
            glcm.calc_glc_2d_matrices()
            if self.aggr_method == 'DIR_MERG':
                glcm.calc_2_5d_direction_merged_glcm_features()
            elif self.aggr_method == 'MERG':
                glcm.calc_2_5d_merged_glcm_features()
            elif self.aggr_method == 'AVER':
                glcm.calc_2d_averaged_glcm_features()
            elif self.aggr_method == 'SLICE_MERG':
                glcm.calc_2d_slice_merged_glcm_features()

        self.glcm_features = [glcm.joint_max,
                              glcm.joint_average,
                              glcm.joint_var,
                              glcm.joint_entropy,
                              glcm.dif_average,
                              glcm.dif_var,
                              glcm.dif_entropy,
                              glcm.sum_average,
                              glcm.sum_var,
                              glcm.sum_entropy,
                              glcm.ang_second_moment,
                              glcm.contrast,
                              glcm.dissimilarity,
                              glcm.inv_diff,
                              glcm.norm_inv_diff,
                              glcm.inv_diff_moment,
                              glcm.norm_inv_diff_moment,
                              glcm.inv_variance,
                              glcm.cor,
                              glcm.autocor,
                              glcm.cluster_tendency,
                              glcm.cluster_shade,
                              glcm.cluster_prominence,
                              glcm.inf_cor_1,
                              glcm.inf_cor_2]
        self.glcm_features_list.append(self.glcm_features)
        glrlm = GLRLM_GLSZM_GLDZM_NGLDM(image=self.patient_intensity_mask.array.T, slice_weight=self.slice_weighting,
                                        slice_median=self.slice_median)
        if self.aggr_dim == '3D':

            glrlm.calc_glrl_3d_matrix()
            if self.aggr_method == 'AVER':
                glrlm.calc_3d_averaged_glrlm_features()
            elif self.aggr_method == 'MERG':
                glrlm.calc_3d_merged_glrlm_features()

        elif self.aggr_dim == '2.5D' or self.aggr_dim == '2D':
            glrlm.calc_glrl_2d_matrices()

            if self.aggr_method == 'DIR_MERG':
                glrlm.calc_2_5d_direction_merged_glrlm_features()
            elif self.aggr_method == 'MERG':
                glrlm.calc_2_5d_merged_glrlm_features()
            elif self.aggr_method == 'AVER':
                glrlm.calc_2d_averaged_glrlm_features()
            elif self.aggr_method == 'SLICE_MERG':
                glrlm.calc_2d_slice_merged_glrlm_features()

        self.glrlm_features = [glrlm.short_runs_emphasis,
                               glrlm.long_runs_emphasis,
                               glrlm.low_grey_level_run_emphasis,
                               glrlm.high_gr_lvl_emphasis,
                               glrlm.short_low_gr_lvl_emphasis,
                               glrlm.short_high_gr_lvl_emphasis,
                               glrlm.long_low_gr_lvl_emphasis,
                               glrlm.long_high_gr_lvl_emphasis,
                               glrlm.non_uniformity,
                               glrlm.norm_non_uniformity,
                               glrlm.length_non_uniformity,
                               glrlm.norm_length_non_uniformity,
                               glrlm.percentage,
                               glrlm.gr_lvl_var,
                               glrlm.length_var,
                               glrlm.entropy]
        self.glrlm_features_list.append(self.glrlm_features)

        glszm_gldzm = GLRLM_GLSZM_GLDZM_NGLDM(image=self.patient_intensity_mask.array.T,
                                              slice_weight=self.slice_weighting, slice_median=self.slice_median)
        if self.aggr_dim == '3D':
            glszm_gldzm.calc_glsz_gldz_3d_matrices(self.patient_morphological_mask.array.T)
            glszm_gldzm.calc_3d_glszm_features()

            self.glszm_features = [glszm_gldzm.short_runs_emphasis,
                                   glszm_gldzm.long_runs_emphasis,
                                   glszm_gldzm.low_grey_level_run_emphasis,
                                   glszm_gldzm.high_gr_lvl_emphasis,
                                   glszm_gldzm.short_low_gr_lvl_emphasis,
                                   glszm_gldzm.short_high_gr_lvl_emphasis,
                                   glszm_gldzm.long_low_gr_lvl_emphasis,
                                   glszm_gldzm.long_high_gr_lvl_emphasis,
                                   glszm_gldzm.non_uniformity,
                                   glszm_gldzm.norm_non_uniformity,
                                   glszm_gldzm.length_non_uniformity,
                                   glszm_gldzm.norm_length_non_uniformity,
                                   glszm_gldzm.percentage,
                                   glszm_gldzm.gr_lvl_var,
                                   glszm_gldzm.length_var,
                                   glszm_gldzm.entropy]
            self.glszm_features_list.append(self.glszm_features)

            glszm_gldzm.reset_fields()
            glszm_gldzm.calc_3d_gldzm_features()

            self.gldzm_features = [glszm_gldzm.short_runs_emphasis,
                                   glszm_gldzm.long_runs_emphasis,
                                   glszm_gldzm.low_grey_level_run_emphasis,
                                   glszm_gldzm.high_gr_lvl_emphasis,
                                   glszm_gldzm.short_low_gr_lvl_emphasis,
                                   glszm_gldzm.short_high_gr_lvl_emphasis,
                                   glszm_gldzm.long_low_gr_lvl_emphasis,
                                   glszm_gldzm.long_high_gr_lvl_emphasis,
                                   glszm_gldzm.non_uniformity,
                                   glszm_gldzm.norm_non_uniformity,
                                   glszm_gldzm.length_non_uniformity,
                                   glszm_gldzm.norm_length_non_uniformity,
                                   glszm_gldzm.percentage,
                                   glszm_gldzm.gr_lvl_var,
                                   glszm_gldzm.length_var,
                                   glszm_gldzm.entropy]
            self.gldzm_features_list.append(self.gldzm_features)

        else:
            glszm_gldzm.calc_glsz_gldz_2d_matrices(self.patient_morphological_mask.array.T)
            if self.aggr_dim == '2.5D':
                glszm_gldzm.calc_2_5d_glszm_features()

                self.glszm_features = [glszm_gldzm.short_runs_emphasis,
                                       glszm_gldzm.long_runs_emphasis,
                                       glszm_gldzm.low_grey_level_run_emphasis,
                                       glszm_gldzm.high_gr_lvl_emphasis,
                                       glszm_gldzm.short_low_gr_lvl_emphasis,
                                       glszm_gldzm.short_high_gr_lvl_emphasis,
                                       glszm_gldzm.long_low_gr_lvl_emphasis,
                                       glszm_gldzm.long_high_gr_lvl_emphasis,
                                       glszm_gldzm.non_uniformity,
                                       glszm_gldzm.norm_non_uniformity,
                                       glszm_gldzm.length_non_uniformity,
                                       glszm_gldzm.norm_length_non_uniformity,
                                       glszm_gldzm.percentage,
                                       glszm_gldzm.gr_lvl_var,
                                       glszm_gldzm.length_var,
                                       glszm_gldzm.entropy]
                self.glszm_features_list.append(self.glszm_features)

                glszm_gldzm.reset_fields()
                glszm_gldzm.calc_2_5d_gldzm_features()

                self.gldzm_features = [glszm_gldzm.short_runs_emphasis,
                                       glszm_gldzm.long_runs_emphasis,
                                       glszm_gldzm.low_grey_level_run_emphasis,
                                       glszm_gldzm.high_gr_lvl_emphasis,
                                       glszm_gldzm.short_low_gr_lvl_emphasis,
                                       glszm_gldzm.short_high_gr_lvl_emphasis,
                                       glszm_gldzm.long_low_gr_lvl_emphasis,
                                       glszm_gldzm.long_high_gr_lvl_emphasis,
                                       glszm_gldzm.non_uniformity,
                                       glszm_gldzm.norm_non_uniformity,
                                       glszm_gldzm.length_non_uniformity,
                                       glszm_gldzm.norm_length_non_uniformity,
                                       glszm_gldzm.percentage,
                                       glszm_gldzm.gr_lvl_var,
                                       glszm_gldzm.length_var,
                                       glszm_gldzm.entropy]
                self.gldzm_features_list.append(self.gldzm_features)
            else:
                glszm_gldzm.calc_2d_glszm_features()

                self.glszm_features = [glszm_gldzm.short_runs_emphasis,
                                       glszm_gldzm.long_runs_emphasis,
                                       glszm_gldzm.low_grey_level_run_emphasis,
                                       glszm_gldzm.high_gr_lvl_emphasis,
                                       glszm_gldzm.short_low_gr_lvl_emphasis,
                                       glszm_gldzm.short_high_gr_lvl_emphasis,
                                       glszm_gldzm.long_low_gr_lvl_emphasis,
                                       glszm_gldzm.long_high_gr_lvl_emphasis,
                                       glszm_gldzm.non_uniformity,
                                       glszm_gldzm.norm_non_uniformity,
                                       glszm_gldzm.length_non_uniformity,
                                       glszm_gldzm.norm_length_non_uniformity,
                                       glszm_gldzm.percentage,
                                       glszm_gldzm.gr_lvl_var,
                                       glszm_gldzm.length_var,
                                       glszm_gldzm.entropy]
                self.glszm_features_list.append(self.glszm_features)

                glszm_gldzm.reset_fields()
                glszm_gldzm.calc_2d_gldzm_features()

                self.gldzm_features = [glszm_gldzm.short_runs_emphasis,
                                       glszm_gldzm.long_runs_emphasis,
                                       glszm_gldzm.low_grey_level_run_emphasis,
                                       glszm_gldzm.high_gr_lvl_emphasis,
                                       glszm_gldzm.short_low_gr_lvl_emphasis,
                                       glszm_gldzm.short_high_gr_lvl_emphasis,
                                       glszm_gldzm.long_low_gr_lvl_emphasis,
                                       glszm_gldzm.long_high_gr_lvl_emphasis,
                                       glszm_gldzm.non_uniformity,
                                       glszm_gldzm.norm_non_uniformity,
                                       glszm_gldzm.length_non_uniformity,
                                       glszm_gldzm.norm_length_non_uniformity,
                                       glszm_gldzm.percentage,
                                       glszm_gldzm.gr_lvl_var,
                                       glszm_gldzm.length_var,
                                       glszm_gldzm.entropy]
                self.gldzm_features_list.append(self.gldzm_features)

        ngtdm = NGTDM(image=self.patient_intensity_mask.array.T, slice_weight=self.slice_weighting,
                      slice_median=self.slice_median)
        if self.aggr_dim == '3D':
            ngtdm.calc_ngtd_3d_matrix()
            ngtdm.calc_3d_ngtdm_features()
        elif self.aggr_dim == '2.5D':
            ngtdm.calc_ngtd_2d_matrices()
            ngtdm.calc_2_5d_ngtdm_features()
        elif self.aggr_dim == '2D':
            ngtdm.calc_ngtd_2d_matrices()
            ngtdm.calc_2d_ngtdm_features()

        self.ngtdm_features = [ngtdm.coarseness,
                               ngtdm.contrast,
                               ngtdm.busyness,
                               ngtdm.complexity,
                               ngtdm.strength]
        self.ngtdm_features_list.append(self.ngtdm_features)

        ngldm = GLRLM_GLSZM_GLDZM_NGLDM(image=self.patient_intensity_mask.array.T, slice_weight=self.slice_weighting,
                                        slice_median=self.slice_median)
        if self.aggr_dim == '3D':
            ngldm.calc_ngld_3d_matrix()
            ngldm.calc_3d_ngldm_features()
        elif self.aggr_dim == '2.5D':
            ngldm.calc_ngld_2d_matrices()
            ngldm.calc_2_5d_ngldm_features()
        elif self.aggr_dim == '2D':
            ngldm.calc_ngld_2d_matrices()
            ngldm.calc_2d_ngldm_features()

        self.ngldm_features = [ngldm.short_runs_emphasis,
                               ngldm.long_runs_emphasis,
                               ngldm.low_grey_level_run_emphasis,
                               ngldm.high_gr_lvl_emphasis,
                               ngldm.short_low_gr_lvl_emphasis,
                               ngldm.short_high_gr_lvl_emphasis,
                               ngldm.long_low_gr_lvl_emphasis,
                               ngldm.long_high_gr_lvl_emphasis,
                               ngldm.non_uniformity,
                               ngldm.norm_non_uniformity,
                               ngldm.length_non_uniformity,
                               ngldm.norm_length_non_uniformity,
                               ngldm.percentage,
                               ngldm.gr_lvl_var,
                               ngldm.length_var,
                               ngldm.entropy,
                               ngldm.energy]
        self.ngldm_features_list.append(self.ngldm_features)

    def _calc_mask_morphological_features(self):
        morf_features = MorphologicalFeatures(self.patient_morphological_mask.array,
                                              (self.patient_morphological_mask.spacing[::-1]))
        morf_features.calc_mesh()
        morf_features.calc_vol_and_area_mesh()
        morf_features.calc_vol_count()
        morf_features.calc_surf_to_vol_ratio()
        morf_features.calc_compactness_1()
        morf_features.calc_compactness_2()
        morf_features.calc_spherical_disproportion()
        morf_features.calc_sphericity()
        morf_features.calc_asphericity()
        morf_features.calc_centre_of_shift(self.patient_intensity_mask.array)
        morf_features.calc_convex_hull()
        morf_features.calc_max_diameter()
        morf_features.calc_pca()
        morf_features.calc_major_minor_least_axes_len()
        morf_features.calc_elongation()
        morf_features.calc_flatness()
        morf_features.calc_vol_and_area_densities_aabb()
        morf_features.calc_vol_density_aee()
        morf_features.calc_area_density_aee()
        morf_features.calc_vol_density_ch()
        morf_features.calc_area_density_ch()
        morf_features.calc_integrated_intensity(self.patient_intensity_mask.array)

        self.mort_features = [morf_features.vol_mesh,
                              morf_features.vol_count,
                              morf_features.area_mesh,
                              morf_features.surf_to_vol_ratio,
                              morf_features.compactness_1,
                              morf_features.compactness_2,
                              morf_features.spherical_disproportion,
                              morf_features.sphericity,
                              morf_features.asphericity,
                              morf_features.centre_of_shift,
                              morf_features.max_diameter,
                              morf_features.major_axis_len,
                              morf_features.minor_axis_len,
                              morf_features.least_axis_len,
                              morf_features.elongation,
                              morf_features.flatness,
                              morf_features.vol_density_aabb,
                              morf_features.area_density_aabb,
                              morf_features.vol_density_aee,
                              morf_features.area_density_aee,
                              morf_features.vol_density_ch,
                              morf_features.area_density_ch,
                              morf_features.integrated_intensity
                              ]

        self.patient_morf_features_list.append(self.mort_features)