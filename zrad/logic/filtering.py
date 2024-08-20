import os
import sys
from datetime import datetime
from itertools import permutations
from multiprocessing import Pool, cpu_count

import SimpleITK as sitk
import numpy as np
import pywt
from scipy import ndimage as ndi

from .toolbox_logic import extract_nifti_image, Image, list_folders_in_defined_range, extract_dicom, \
    check_dicom_tags, get_logger, handle_uncaught_exception, close_all_loggers

sys.excepthook = handle_uncaught_exception


class Mean:
    def __init__(self, padding_type, support, dimensionality):
        close_all_loggers()
        self.filter_logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.filter_logger = get_logger(self.filter_logger_date_time+"_Mean_filter")

        self.type = 'Mean'

        if dimensionality in ['2D', '3D']:
            self.dimensionality = dimensionality
        else:
            self.filter_logger.error(f"Wrong dimensionality '{dimensionality}'. Available dimensions '2D' and '3D'.")
            raise ValueError(f"Wrong dimensionality '{dimensionality}'. Available dimensions '2D' and '3D'.")

        if isinstance(support, int):
            self.support = support
        else:
            self.filter_logger.error(f"Support should be int but '{type(support)}' detected.")
            raise ValueError(f"Support should be int but '{type(support)}' detected.")

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            self.filter_logger.error(f"Wrong padding type '{padding_type}'. "
                                     "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")
            raise ValueError(f"Wrong padding type '{padding_type}'. "
                             "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")

        self.filter_logger.debug(f"Defined {dimensionality} mean filter with support of {support}, "
                                 f"and {padding_type} padding type.")

    def implement(self, img):
        if self.dimensionality == "2D":
            filt_mat = np.ones([self.support, self.support])
            filt_mat = filt_mat / np.prod(filt_mat.shape)
            filtered_img = np.ones(img.shape)
            for i in range(img.shape[2]):
                filtered_img[:, :, i] = ndi.convolve(input=img[:, :, i], weights=filt_mat, mode=self.padding_type)
        elif self.dimensionality == "3D":
            filt_mat = np.ones([self.support, self.support, self.support])
            filt_mat = filt_mat / np.prod(filt_mat.shape)
            filtered_img = ndi.convolve(input=img, weights=filt_mat, mode=self.padding_type)
        else:
            filtered_img = None
        return filtered_img


class LoG:
    """LoG"""

    def __init__(self, padding_type, sigma_mm, cutoff, dimensionality):

        close_all_loggers()
        self.filter_logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.filter_logger = get_logger(self.filter_logger_date_time+'_LoG_filter')

        self.type = 'Laplacian of Gaussian'

        if dimensionality in ['2D', '3D']:
            self.dimensionality = dimensionality
        else:
            self.filter_logger.error(f"Wrong dimensionality '{dimensionality}'. Available dimensions '2D' and '3D'.")
            raise ValueError(f"Wrong dimensionality '{dimensionality}'. Available dimensions '2D' and '3D'.")

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            self.filter_logger.error(f"Wrong padding type '{padding_type}'. "
                                     f"Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")
            raise ValueError(f"Wrong padding type '{padding_type}'. "
                             f"Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")

        if isinstance(sigma_mm, (int, float)):
            self.sigma_mm = sigma_mm
        else:
            self.filter_logger.error(f'Sigma (in mm) should be int or float but {type(sigma_mm)} detected.')
            raise ValueError(f'Sigma (in mm) should be int or float but {type(sigma_mm)} detected.')

        if isinstance(cutoff, (int, float)):
            self.cutoff = cutoff
        else:
            self.filter_logger.error(f'Cutoff should be int or float but {type(cutoff)} detected.')
            raise ValueError(f'Cutoff should be int or float but {type(cutoff)} detected.')

        self.padding_constant = 0.0
        self.res_mm = None

        self.filter_logger.debug(f"Defined {dimensionality} LoG filter with sigma {sigma_mm}, cutoff {cutoff}, "
                                 f"and {padding_type} padding type.")

    def implement(self, img):
        sigma = self.sigma_mm / self.res_mm
        if self.dimensionality == "3D":
            filtered_img = ndi.gaussian_laplace(img, sigma=sigma, mode=self.padding_type, cval=self.padding_constant,
                                                truncate=self.cutoff)
        elif self.dimensionality == "2D":
            filtered_img = np.nan * np.ones(img.shape)
            for i in range(img.shape[2]):
                filtered_img[:, :, i] = ndi.gaussian_laplace(img[:, :, i], sigma=sigma, mode=self.padding_type,
                                                             cval=self.padding_constant, truncate=self.cutoff)
        else:
            filtered_img = None
        return filtered_img


class Wavelets2D:
    """Wavelet filtering in 2D."""

    def __init__(self, wavelet_type, padding_type, response_map, decomposition_level, rotation_invariance=False):

        close_all_loggers()
        self.filter_logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.filter_logger = get_logger(self.filter_logger_date_time+'_Wavelets2D')

        self.type = 'Wavelets'

        self.dimensionality = '2D'

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            self.filter_logger.error(f"Wrong padding type '{padding_type}'. "
                                     "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")
            raise ValueError(f"Wrong padding type '{padding_type}'. "
                             "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")

        if wavelet_type in ['db3', 'db2', 'coif1', 'haar']:
            self.wavelet_type = wavelet_type
        else:
            self.filter_logger.error(f"Wrong wavelet type '{wavelet_type}'. "
                                     "Available wavelet types are: 'db3', 'db2', 'coif1', 'haar'.")
            raise ValueError(f"Wrong wavelet type '{wavelet_type}'. "
                             "Available wavelet types are: 'db3', 'db2', 'coif1', 'haar'.")

        if decomposition_level in [1, 2]:
            self.decomposition_level = decomposition_level
        else:
            self.filter_logger.error(f"Wrong decomposition_level' {decomposition_level}'. "
                                     "Decomposition level should be integer. "
                                     "Available decomposition levels are: 1 and 2.")
            raise ValueError(f"Wrong decomposition_level' {decomposition_level}'. "
                             "Decomposition level should be integer. Available decomposition levels are: 1 and 2.")

        if response_map in ['LL', 'HL', 'LH', 'HH']:
            self.response_map = response_map
        else:
            self.filter_logger.error(f"Wrong response_map' {response_map}'. "
                                     "Available response_maps are: 'LL', 'HL', 'LH', 'HH'.")
            raise ValueError(f"Wrong response_map' {response_map}'. "
                             "Available response_maps are: 'LL', 'HL', 'LH', 'HH'.")

        if isinstance(rotation_invariance, bool):
            self.rotation_invariance = rotation_invariance
        else:
            self.filter_logger.error("Rotation Invariance should be "
                                     f"True or False but '{type(rotation_invariance)}' detected.")
            raise ValueError(f"Rotation Invariance should be True or False but '{type(rotation_invariance)}' detected.")

        self.filter_logger.debug(f"Defined 2D {wavelet_type} Wavelet filter with response map {response_map}, "
                                 f"decomposition level {decomposition_level}, "
                                 f"pseudo rotation invariance is {rotation_invariance}"
                                 f"and {padding_type} padding type.")

    def _get_kernel(self, response, decomposition_level=1):
        if response == "L":
            kernel = pywt.Wavelet(name=self.wavelet_type).filter_bank[0]
        elif response == "H":
            kernel = pywt.Wavelet(name=self.wavelet_type).filter_bank[1]
        else:
            kernel = None
        if decomposition_level == 2:
            kernel = [[e, 0] for e in kernel]
            kernel = [item for sublist in kernel for item in sublist]
        return kernel

    def _filter(self, img, x_filter, y_filter):
        filtered_img = ndi.convolve1d(img, x_filter, axis=1, mode=self.padding_type)
        filtered_img = ndi.convolve1d(filtered_img, y_filter, axis=0, mode=self.padding_type)
        return filtered_img

    def implement(self, img):
        if self.decomposition_level == 1:
            x_filter = self._get_kernel(self.response_map[0])
            y_filter = self._get_kernel(self.response_map[1])
            if self.rotation_invariance:
                final_image = np.zeros(img.shape)
                for i in range(img.shape[2]):
                    for k in range(4):
                        final_image[:, :, i] += np.rot90(self._filter(np.rot90(img[:, :, i], k=k, axes=(0, 1)),
                                                         x_filter, y_filter), k=k, axes=(1, 0))
                filtered_img = final_image / 4
            else:
                filtered_img = np.zeros(img.shape)
                for i in range(img.shape[2]):
                    filtered_img[:, :, i] = self._filter(img[:, :, i], x_filter, y_filter)
        elif self.decomposition_level == 2:
            l_filter = self._get_kernel("L")
            x_filter = self._get_kernel(self.response_map[0], decomposition_level=2)
            y_filter = self._get_kernel(self.response_map[1], decomposition_level=2)

            final_image = np.zeros(img.shape)
            for i in range(img.shape[2]):
                for k in range(4):
                    img_level0 = np.rot90(img[:, :, i], k=k, axes=(0, 1))  # original
                    img_level1 = self._filter(img_level0, l_filter, l_filter)
                    img_level2 = self._filter(img_level1, x_filter, y_filter)
                    final_image[:, :, i] += np.rot90(img_level2, k=k, axes=(1, 0))
            filtered_img = final_image / 4

        return filtered_img


class Wavelets3D:
    """Wavelet filtering."""

    def __init__(self, wavelet_type, padding_type, response_map, decomposition_level, rotation_invariance=False):

        close_all_loggers()
        self.filter_logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.filter_logger = get_logger(self.filter_logger_date_time+'_Wavelets3D')

        self.type = 'Wavelets'

        self.dimensionality = '3D'

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            self.filter_logger.error(f"Wrong padding type '{padding_type}'. "
                                     "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")
            raise ValueError(f"Wrong padding type '{padding_type}'. "
                             "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")

        if wavelet_type in ['db3', 'db2', 'coif1', 'haar']:
            self.wavelet_type = wavelet_type
        else:
            self.filter_logger.error(f"Wrong wavelet type '{wavelet_type}'. "
                                     "Available wavelet types are: 'db3', 'db2', 'coif1', 'haar'.")
            raise ValueError(f"Wrong wavelet type '{wavelet_type}'. "
                             "Available wavelet types are: 'db3', 'db2', 'coif1', 'haar'.")

        if decomposition_level in [1, 2]:
            self.decomposition_level = decomposition_level
        else:
            self.filter_logger.error(f"Wrong decomposition_level' {decomposition_level}'. "
                                     "Decomposition level should be integer. "
                                     "Available decomposition levels are: 1 and 2.")
            raise ValueError(f"Wrong decomposition_level' {decomposition_level}'. "
                             "Decomposition level should be integer. Available decomposition levels are: 1 and 2.")

        if response_map in ['LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', 'HLH', 'HHH']:
            self.response_map = response_map
        else:
            self.filter_logger.error(f"Wrong response_map' {response_map}'. "
                                     "Available response_maps are: "
                                     "'LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', 'HLH', 'HHH'.")
            raise ValueError(f"Wrong response_map' {response_map}'. "
                             "Available response_maps are: 'LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', 'HLH', 'HHH'.")

        if isinstance(rotation_invariance, bool):
            self.rotation_invariance = rotation_invariance
        else:
            self.filter_logger.error("Rotation Invariance should be "
                                     f"True or False but '{type(rotation_invariance)}' detected.")
            raise ValueError(f"Rotation Invariance should be True or False but '{type(rotation_invariance)}' detected.")

        self.pooling = None

        self.filter_logger.debug(f"Defined 3D {wavelet_type} Wavelet filter with response map {response_map}, "
                                 f"decomposition level {decomposition_level}, "
                                 f"pseudo rotation invariance is {rotation_invariance}"
                                 f"and {padding_type} padding type.")

    def _get_kernel(self, response, decomposition_level=1):
        if response == "L":
            kernel = pywt.Wavelet(name=self.wavelet_type).filter_bank[0]
        elif response == "H":
            kernel = pywt.Wavelet(name=self.wavelet_type).filter_bank[1]
        else:
            kernel = None
        if decomposition_level == 2:
            kernel = [[e, 0] for e in kernel]
            kernel = [item for sublist in kernel for item in sublist]
        return kernel

    def _filter(self, img, x_filter, y_filter, z_filter):
        filtered_img = ndi.convolve1d(img, x_filter, axis=1, mode=self.padding_type)
        filtered_img = ndi.convolve1d(filtered_img, y_filter, axis=0, mode=self.padding_type)
        filtered_img = ndi.convolve1d(filtered_img, z_filter, axis=2, mode=self.padding_type)
        return filtered_img

    def implement(self, img):
        if self.decomposition_level == 1:
            x_filter = self._get_kernel(self.response_map[0])
            y_filter = self._get_kernel(self.response_map[1])
            z_filter = self._get_kernel(self.response_map[2])
            if self.rotation_invariance:
                final_image = np.zeros(img.shape)
                kernels_permutation = [(x_filter, y_filter, z_filter),
                                       (z_filter, x_filter, y_filter),
                                       (y_filter, z_filter, x_filter)]
                for kernels in kernels_permutation:
                    final_image += self._filter(img, kernels[0], kernels[1], kernels[2])
                    final_image += self._filter(img[::-1, :, :], kernels[0], kernels[1], kernels[2])[::-1, :, :]
                    final_image += self._filter(img[:, ::-1, :], kernels[0], kernels[1], kernels[2])[:, ::-1, :]
                    final_image += self._filter(img[:, :, ::-1], kernels[0], kernels[1], kernels[2])[:, :, ::-1]
                    final_image += self._filter(img[::-1, ::-1, :], kernels[0], kernels[1], kernels[2])[::-1, ::-1, :]
                    final_image += self._filter(img[::-1, :, ::-1], kernels[0], kernels[1], kernels[2])[::-1, :, ::-1]
                    final_image += self._filter(img[:, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[:, ::-1, ::-1]
                    final_image += self._filter(img[::-1, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[::-1, ::-1, ::-1]
                filtered_img = final_image / (8 * len(kernels_permutation))
            else:
                filtered_img = self._filter(img, x_filter, y_filter, z_filter)
        else:
            # First, do low pass LLL
            x_filter = self._get_kernel("L")
            y_filter = self._get_kernel("L")
            z_filter = self._get_kernel("L")
            kernels_permutation = [(x_filter, y_filter, z_filter),
                                   (z_filter, x_filter, y_filter),
                                   (y_filter, z_filter, x_filter)]
            level1_responses = list()
            for kernels in kernels_permutation:
                level1_responses.append(self._filter(img, kernels[0], kernels[1], kernels[2]))
                level1_responses.append(self._filter(img[::-1, :, :], kernels[0], kernels[1], kernels[2])[::-1, :, :])
                level1_responses.append(self._filter(img[:, ::-1, :], kernels[0], kernels[1], kernels[2])[:, ::-1, :])
                level1_responses.append(self._filter(img[:, :, ::-1], kernels[0], kernels[1], kernels[2])[:, :, ::-1])
                level1_responses.append(self._filter(img[::-1, ::-1, :], kernels[0], kernels[1], kernels[2])[::-1, ::-1, :])
                level1_responses.append(self._filter(img[::-1, :, ::-1], kernels[0], kernels[1], kernels[2])[::-1, :, ::-1])
                level1_responses.append(self._filter(img[:, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[:, ::-1, ::-1])
                level1_responses.append(self._filter(img[::-1, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[::-1, ::-1, ::-1])

            # Now do 2nd level filtering
            x_filter = self._get_kernel(self.response_map[0], decomposition_level=2)
            y_filter = self._get_kernel(self.response_map[1], decomposition_level=2)
            z_filter = self._get_kernel(self.response_map[2], decomposition_level=2)
            final_image = np.zeros(img.shape)
            kernels_permutation = [(x_filter, y_filter, z_filter),
                                   (z_filter, x_filter, y_filter),
                                   (y_filter, z_filter, x_filter)]
            for kernels in kernels_permutation:
                final_image += self._filter(level1_responses[0], kernels[0], kernels[1], kernels[2])
                final_image += self._filter(level1_responses[1][::-1, :, :], kernels[0], kernels[1], kernels[2])[::-1, :, :]
                final_image += self._filter(level1_responses[2][:, ::-1, :], kernels[0], kernels[1], kernels[2])[:, ::-1, :]
                final_image += self._filter(level1_responses[3][:, :, ::-1], kernels[0], kernels[1], kernels[2])[:, :, ::-1]
                final_image += self._filter(level1_responses[4][::-1, ::-1, :], kernels[0], kernels[1], kernels[2])[::-1, ::-1, :]
                final_image += self._filter(level1_responses[5][::-1, :, ::-1], kernels[0], kernels[1], kernels[2])[::-1, :, ::-1]
                final_image += self._filter(level1_responses[6][:, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[:, ::-1, ::-1]
                final_image += self._filter(level1_responses[7][::-1, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[::-1, ::-1, ::-1]
            filtered_img = final_image / (8 * len(kernels_permutation))
        return filtered_img


class Laws:
    """Laws2"""

    def __init__(self, response_map, padding_type, distance, energy_map, dimensionality,
                 rotation_invariance=False, pooling=None):

        close_all_loggers()
        self.filter_logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.filter_logger = get_logger(self.filter_logger_date_time+'_Laws_kernels')

        self.type = 'Laws Kernels'

        if dimensionality in ['2D', '3D']:
            self.dimensionality = dimensionality
        else:
            self.filter_logger.error(f"Wrong dimensionality '{dimensionality}'. Available dimensions '2D' and '3D'.")
            raise ValueError(f"Wrong dimensionality '{dimensionality}'. Available dimensions '2D' and '3D'.")

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            self.filter_logger.error(f"Wrong padding type '{padding_type}'. "
                                     "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")
            raise ValueError(f"Wrong padding type '{padding_type}'. "
                             "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")

        if isinstance(distance, int):
            self.distance = distance
        else:
            self.filter_logger.error(f"Distance should be 'int' but '{type(distance)}' detected.")
            raise ValueError(f"Distance should be 'int' but '{type(distance)}' detected.")

        if isinstance(energy_map, bool):
            self.energy_map = energy_map
        else:
            self.filter_logger.error('Energy map can be only True or False.')
            raise ValueError('Energy map can be only True or False.')

        if isinstance(rotation_invariance, bool):
            self.rotation_invariance = rotation_invariance
        else:
            self.filter_logger.error("Rotation Invariance should be "
                                     f"True or False but '{type(rotation_invariance)}' detected.")
            raise ValueError(f"Rotation Invariance should be True or False but '{type(rotation_invariance)}' detected.")

        self.response_map = response_map
        self.pooling = pooling

        self.filter_logger.debug(f"Defined {dimensionality} Laws Kernels filter with energy map is {energy_map}, "
                                 f"response map {response_map}, distance {distance} "
                                 f"pseudo rotation invariance is {rotation_invariance}, pooling {pooling},"
                                 f"and {padding_type} padding type.")

    def _get_kernel(self, l_type, support):
        if l_type == "L":
            if support == 3:
                return 1 / np.sqrt(6) * np.array([1, 2, 1])
            elif support == 5:
                return 1 / np.sqrt(70) * np.array([1, 4, 6, 4, 1])
        elif l_type == "E":
            if support == 3:
                return 1 / np.sqrt(2) * np.array([-1, 0, 1])
            elif support == 5:
                return 1 / np.sqrt(10) * np.array([-1, -2, 0, 2, 1])
        elif l_type == "S":
            if support == 3:
                return 1 / np.sqrt(6) * np.array([-1, 2, -1])
            elif support == 5:
                return 1 / np.sqrt(6) * np.array([-1, 0, 2, 0, -1])
        elif l_type == "W":
            if support == 5:
                return 1 / np.sqrt(10) * np.array([-1, 2, 0, -2, 1])
        elif l_type == "R":
            if support == 5:
                return 1 / np.sqrt(70) * np.array([1, -4, 6, -4, 1])

    def _get_response_maps(self):
        parts = [self.response_map[i:i + 2] for i in range(0, len(self.response_map), 2)]
        return [''.join(e) for e in permutations(parts)]

    def _filter(self, img, response_map):
        if self.dimensionality == "3D":
            x_filt = self._get_kernel(response_map[0], int(response_map[1]))
            y_filt = self._get_kernel(response_map[2], int(response_map[3]))
            z_filt = self._get_kernel(response_map[4], int(response_map[5]))

            filtered_img = ndi.convolve1d(img, x_filt, axis=1, mode=self.padding_type)
            filtered_img = ndi.convolve1d(filtered_img, y_filt, axis=0, mode=self.padding_type)
            filtered_img = ndi.convolve1d(filtered_img, z_filt, axis=2, mode=self.padding_type)
        elif self.dimensionality == "2D":
            x_filt = self._get_kernel(response_map[0], int(response_map[1]))
            y_filt = self._get_kernel(response_map[2], int(response_map[3]))

            filtered_img = ndi.convolve1d(img, x_filt, axis=1, mode=self.padding_type)
            filtered_img = ndi.convolve1d(filtered_img, y_filt, axis=0, mode=self.padding_type)
        else:
            filtered_img = None
        return filtered_img

    def implement(self, img):
        final_image = None
        if self.rotation_invariance:
            response_maps = self._get_response_maps()

            if self.pooling == "avg":
                # average pooling of Laws filtering was not tested in IBSI 2.
                final_image = np.nan * np.ones(img.shape)
                for response_map in response_maps:
                    final_image += self._filter(img, response_map)
                    final_image += self._filter(img[::-1, :, :], response_map)[::-1, :, :]
                    final_image += self._filter(img[:, ::-1, :], response_map)[:, ::-1, :]
                    final_image += self._filter(img[:, :, ::-1], response_map)[:, :, ::-1]
                    final_image += self._filter(img[::-1, ::-1, :], response_map)[::-1, ::-1, :]
                    final_image += self._filter(img[::-1, :, ::-1], response_map)[::-1, :, ::-1]
                    final_image += self._filter(img[:, ::-1, ::-1], response_map)[:, ::-1, ::-1]
                    final_image += self._filter(img[::-1, ::-1, ::-1], response_map)[::-1, ::-1, ::-1]
                final_image = final_image / 24  # shouldn't it rather be len(response_maps) * 8 = 48?

            elif self.pooling == "max":
                final_image = np.NINF * np.ones(img.shape)
                for response_map in response_maps:
                    final_image = np.maximum(final_image, self._filter(img, response_map))
                    final_image = np.maximum(final_image, self._filter(img[::-1, :, :], response_map)[::-1, :, :])
                    final_image = np.maximum(final_image, self._filter(img[:, ::-1, :], response_map)[:, ::-1, :])
                    final_image = np.maximum(final_image, self._filter(img[:, :, ::-1], response_map)[:, :, ::-1])
                    final_image = np.maximum(final_image, self._filter(img[::-1, ::-1, :], response_map)[::-1, ::-1, :])
                    final_image = np.maximum(final_image, self._filter(img[::-1, :, ::-1], response_map)[::-1, :, ::-1])
                    final_image = np.maximum(final_image, self._filter(img[:, ::-1, ::-1], response_map)[:, ::-1, ::-1])
                    final_image = np.maximum(final_image,
                                             self._filter(img[::-1, ::-1, ::-1], response_map)[::-1, ::-1, ::-1])
        else:
            final_image = self._filter(img, self.response_map)

        if self.energy_map:
            final_image = self._get_energy_map(final_image)

        return final_image

    def _get_energy_map(self, img):
        if self.dimensionality == "2D":
            filt_mat = np.ones([2 * self.distance + 1, 2 * self.distance + 1])
            filt_mat = filt_mat / np.prod(filt_mat.shape)
            energy_map = np.nan * np.ones(img.shape)
            for i in range(img.shape[2]):
                energy_map[:, :, i] = ndi.convolve(input=np.abs(img[:, :, i]), weights=filt_mat, mode='reflect')
        elif self.dimensionality == "3D":
            filt_mat = np.ones([2 * self.distance + 1, 2 * self.distance + 1, 2 * self.distance + 1])
            filt_mat = filt_mat / np.prod(filt_mat.shape)
            energy_map = ndi.convolve(input=np.abs(img), weights=filt_mat, mode='reflect')
        else:
            energy_map = None
        return energy_map


class Filtering:

    def __init__(self, input_dir, output_dir,
                 input_data_type, input_imaging_modality,
                 filter_type,
                 start_folder=None, stop_folder=None, list_of_patient_folders=None,
                 nifti_image=None,
                 number_of_threads=1):

        close_all_loggers()
        self.logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.logger = get_logger(self.logger_date_time+'_Filtering')
        self.logger.info("Preliminary Data Check Started")

        if os.path.exists(input_dir):
            self.input_dir = input_dir
        else:
            self.logger.error(f"Load directory '{input_dir}' does not exist.")
            raise ValueError(f"Load directory '{input_dir}' does not exist.")

        if os.path.exists(output_dir):
            self.output_dir = output_dir
        else:
            os.makedirs(output_dir)
            self.output_dir = output_dir

        if start_folder is not None and stop_folder is not None:
            self.list_of_patient_folders = list_folders_in_defined_range(start_folder, stop_folder, self.input_dir)
        elif list_of_patient_folders is not None and list_of_patient_folders not in [[], ['']]:
            self.list_of_patient_folders = list_of_patient_folders
        elif list_of_patient_folders is None and start_folder is None and stop_folder is None:
            self.list_of_patient_folders = [
                e for e in os.listdir(input_dir)
                if not e.startswith('.') and os.path.isdir(os.path.join(input_dir, e))
            ]
        else:
            raise ValueError('Incorrectly selected patient folders.')

        if input_data_type in ['DICOM', 'NIfTI']:
            self.input_data_type = input_data_type
        else:
            raise ValueError("Wrong input data types, available types: 'DICOM', 'NIfTI'.")

        if self.input_data_type == 'DICOM':
            list_pat_id_to_del = []
            for pat_index, pat_path in enumerate(self.list_of_patient_folders):
                if check_dicom_tags(os.path.join(input_dir, pat_path), pat_path, self.logger, filter_type.dimensionality):
                    list_pat_id_to_del.append(pat_path)
            for pat_to_del in np.unique(list_pat_id_to_del):
                self.list_of_patient_folders.remove(pat_to_del)

        if isinstance(number_of_threads, int) and 0 < number_of_threads <= cpu_count():
            self.number_of_threads = number_of_threads
        else:
            raise ValueError('Number of threads is not an integer or selected number is greater '
                             f'than maximum number of available СPU. (Max available {cpu_count()} units).')

        if self.input_data_type == 'NIfTI':
            if nifti_image is not None:
                image_exists = True
                for folder in self.list_of_patient_folders:
                    if (not os.path.isfile(os.path.join(input_dir, str(folder), nifti_image + '.nii.gz'))
                            and not os.path.isfile(os.path.join(input_dir, str(folder), nifti_image + '.nii'))):
                        image_exists = False
                        if not image_exists:
                            print('The NIFTI image file does not exists: '
                                  + os.path.join(input_dir, str(folder), nifti_image))
                if image_exists:
                    self.nifti_image = nifti_image
            else:
                raise ValueError('Select the NIfTI image file')

        if input_imaging_modality in ['CT', 'PT', 'MR']:
            self.input_imaging_mod = input_imaging_modality
        else:
            raise ValueError("Wrong input imaging type, available types: 'CT', 'PT', 'MR'.")

        if filter_type.type in ['Mean', 'Laplacian of Gaussian', 'Laws Kernels', 'Wavelets']:
            self.filtering_type = filter_type.type
        else:
            raise ValueError(f"Wrong filter_type: {filter_type.type}, available types: "
                             "'Mean', 'Laplacian of Gaussian', 'Laws Kernels', and 'Wavelets'.")

        self.filter = filter_type

        self.patient_folder = None
        self.patient_number = None
        self.logger.info("Preliminary Data Check Completed")

    def filtering(self):
        if len(self.list_of_patient_folders) > 0:
            try:
                self.logger.info('Filtering Started')
                with Pool(self.number_of_threads) as pool:
                    pool.map(self._load_patient, sorted(self.list_of_patient_folders))
                self.logger.info('Filtering Completed!')
            except Exception:
                self.logger.error("Caught an exception", exc_info=True)
        else:
            self.logger.info('No patients to filter.')

    def _load_patient(self, patient_number):

        self.patient_logger = get_logger(self.logger_date_time+'_Filtering')
        self.patient_logger.info(f'Working on patient: {patient_number}')
        self.patient_number = str(patient_number)
        self.patient_folder = os.path.join(self.input_dir, str(self.patient_number))
        self.pat_image = None
        self.filtered_image = None
        if self.input_data_type == 'NIfTI':
            self._process_nifti_files()
        elif self.input_data_type == 'DICOM':
            self._process_dicom_files()
        self._apply_filter()
        self._save_as_nifti()
        self.patient_logger.info(f'Completed patient: {patient_number}')

    def _process_nifti_files(self):
        self.pat_image = self._extract_nifti('IMAGE')

    def _extract_nifti(self, key):
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        if key == 'IMAGE':
            return extract_nifti_image(self, reader)

    def _process_dicom_files(self):
        self.pat_image = extract_dicom(dicom_dir=self.patient_folder, rtstract=False, modality=self.input_imaging_mod)

    def _apply_filter(self):
        if self.filtering_type == 'Laplacian of Gaussian':
            self.filter.res_mm = float(self.pat_image.spacing[0])
        self.filtered_image = self.filter.implement(self.pat_image.array.astype(np.float64).transpose(1, 2, 0))
        self.filtered_image_to_save = Image(array=self.filtered_image.transpose(2, 0, 1),
                                            origin=self.pat_image.origin,
                                            spacing=self.pat_image.spacing,
                                            direction=self.pat_image.direction,
                                            shape=self.pat_image.shape)

    def _save_as_nifti(self):
        if self.filtering_type == 'Mean':
            filter_key=(f'{self.filtering_type}_{self.filter.dimensionality}_{self.filter.support}support'
                 f'_{self.filter.padding_type}')
        elif self.filtering_type == 'Laplacian of Gaussian':
            filter_key = (f'{self.filtering_type}_{self.filter.dimensionality}_{self.filter.sigma_mm}sigma'
                          f'_{self.filter.cutoff}cutoff_'
                          f'{self.filter.padding_type}')
        elif self.filtering_type == 'Laws Kernels':
            filter_key = (f'{self.filtering_type}_{self.filter.dimensionality}_{self.filter.response_map}_'
                   f'{self.filter.rotation_invariance}_{self.filter.pooling}_{self.filter.energy_map}'
                   f'{self.filter.distance}_{self.filter.padding_type}')
        else:
            filter_key = (f'{self.filter.wavelet_type}_{self.filter.dimensionality}_{self.filter.response_map}_'
                   f'{self.filter.decomposition_level}_{self.filter.rotation_invariance}_'
                   f'{self.filter.padding_type}')

        self.filtered_image_to_save.save_as_nifti(instance=self, key=filter_key)
