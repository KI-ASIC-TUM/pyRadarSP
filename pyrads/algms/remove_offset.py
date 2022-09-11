#!/usr/bin/env python3
"""
RemoveOffset algorithm.

Remove offset from last dimension.
"""
# Standard libraries
import numpy as np
# Local libraries
import pyrads.algorithm


class RemoveOffset(pyrads.algorithm.Algorithm):
    """
    Parent class for radar algorithms
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load smoothenting parameters


    def calculate_out_shape(self):
        """
        Do not alter the data dimensionality
        """
        self.out_data_shape = self.in_data_shape

    def _run(self, in_data):
        """
        Return smoothened data.
        Apply only on last dimension.
        """

        result = in_data - np.expand_dims(np.mean(in_data, -1), -1)  # Remove DC offset across samples per ramp

        return result