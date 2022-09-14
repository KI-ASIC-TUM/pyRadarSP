#!/usr/bin/env python3
"""
Algorithm for offset removal
"""
# Standard libraries
import numpy as np
# Local libraries
import pyrads.algorithm


class RemoveOffset(pyrads.algorithm.Algorithm):
    """
    Parent class for radar algorithms
    """
    NAME = "RemoveOffset"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
        offset = np.mean(in_data, -1)
        result = in_data - np.expand_dims(offset, -1) 
        return result
