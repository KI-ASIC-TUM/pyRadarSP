#!/usr/bin/env python3
"""
Identity algorithm. Used mostly for debugging
"""
# Standard libraries
import numpy as np
# Local libraries
import pyrads.algorithm


class Identity(pyrads.algorithm.Algorithm):
    """
    Parent class for radar algorithms
    """
    NAME = "Identity"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load os-cfar parameters


    def calculate_out_shape(self):
        """
        Do not alter the data dimensionality
        """
        self.out_data_shape = self.in_data_shape


    def _run(self, in_data):
        """
        Return input data
        """
        result = in_data
        return result
