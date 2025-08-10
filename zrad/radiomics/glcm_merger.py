"""Utilities for aggregating Grey Level Co-occurrence Matrices (GLCM).

This module contains a small helper class used to merge or aggregate GLCM
matrices before feature extraction.  Aggregation can be performed by slice,
by direction or by fully merging all matrices into a single one.  The class is
intentionally lightweight so that it can be used independently from the feature
extraction code and allows calculating and merging matrices without having to
compute any features.
"""

from __future__ import annotations

import numpy as np


class GLCMMerger:
    """Aggregate GLCM matrices according to a given strategy.

    Parameters
    ----------
    merge_type : {"slice", "direction", "full", None}, optional
        Defines how matrices should be merged:

        ``"slice"``
            Sum matrices over directions for every slice.  Input is expected to
            have the shape ``(n_slices, n_dirs, lvl, lvl)`` and the result will
            have the shape ``(n_slices, lvl, lvl)``.

        ``"direction"``
            Sum matrices over slices for every direction.  Input is expected to
            have the shape ``(n_slices, n_dirs, lvl, lvl)`` and the result will
            have the shape ``(n_dirs, lvl, lvl)``.

        ``"full"``
            Sum matrices over both slices and directions resulting in a single
            matrix with the shape ``(lvl, lvl)``.

        ``None``
            No merging is performed and the input is returned unchanged.
    """

    def __init__(self, merge_type: str | None = None):
        self.merge_type = merge_type

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def merge(self, matrices: np.ndarray) -> np.ndarray:
        """Merge ``matrices`` according to ``merge_type``.

        Parameters
        ----------
        matrices : np.ndarray
            Array containing GLCM matrices.  Supported shapes are ``(n_slices,
            n_dirs, lvl, lvl)`` or ``(n_dirs, lvl, lvl)``.  For the latter only
            ``"direction"`` and ``"full"`` merging are meaningful.

        Returns
        -------
        np.ndarray
            Aggregated matrices following the specified merge strategy.
        """

        if self.merge_type == "slice":
            # Sum over directions for each slice
            return np.sum(matrices, axis=1)

        if self.merge_type == "direction":
            # Sum over slices for each direction
            return np.sum(matrices, axis=0)

        if self.merge_type == "full":
            # Sum over slices and directions
            return np.sum(matrices, axis=tuple(range(matrices.ndim - 2)))

        # ``None`` or unrecognised value -> return matrices unchanged
        return matrices


__all__ = ["GLCMMerger"]

