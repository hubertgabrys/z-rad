import numpy as np


class TextureFeatureBase:
    """Base class for texture feature extractors.

    Parameters
    ----------
    image : ndarray
        Input image array.
    feature_names : sequence of str
        Names of the texture features provided by the subclass.
    slice_weight, slice_median : bool, optional
        Aggregation options used when finalising per-slice features.
    """

    def __init__(self, image, feature_names, slice_weight=False, slice_median=False):
        self.image = image
        self.slice_weight = slice_weight
        self.slice_median = slice_median
        self.feature_names = list(feature_names)

        # Create attributes for the final values and per-slice lists
        for name in self.feature_names:
            setattr(self, name, 0)
            setattr(self, f"{name}_list", [])

    # ------------------------------------------------------------------
    # Helper utilities reused by texture feature classes
    # ------------------------------------------------------------------
    def _reset_feature_lists(self):
        """Clear all per-slice feature lists."""
        for name in self.feature_names:
            getattr(self, f"{name}_list").clear()

    def _append_features(self, feature_dict):
        """Append calculated features to the per-slice lists.

        Parameters
        ----------
        feature_dict : dict
            Mapping of feature names to their calculated values.
        """
        for name, value in feature_dict.items():
            if name in self.feature_names:
                getattr(self, f"{name}_list").append(value)

    def _finalize_features(self, weights):
        """Aggregate per-slice feature values into final attributes."""
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
