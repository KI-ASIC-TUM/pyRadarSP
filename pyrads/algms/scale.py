#!/usr/bin/env python3
"""
Algorithm for scaling data
"""
# Standard libraries
import numpy as np
# Local libraries
import pyrads.algorithm


class Scale(pyrads.algorithm.Algorithm):
    """
    Class for scaling data with a constant value
    """
    NAME = "Scale"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load os-cfar parameters
        self.scaling_factor = kwargs.get("scaling_factor", 1)
        self.mode = kwargs.get("mode", "fixed")


    def calculate_out_shape(self):
        """
        Do not alter the data dimensionality
        """
        self.out_data_shape = self.in_data_shape


    def _run(self, in_data):
        """
        Return data without offset

        Apply the algorithm only on last dimension.
        """
        # Remove DC offset across samples per ramp
        if self.mode == "fixed":
            result = in_data / self.scaling_factor
        elif self.mode == "max":
            data_max = in_data.max(axis=-1)
            scale_factor = np.where(data_max==0, 1, data_max)
            scale_shape = scale_factor.shape + (1,)
            result = in_data /scale_factor.reshape(scale_shape)
        return result
