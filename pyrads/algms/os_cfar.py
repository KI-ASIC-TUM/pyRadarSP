#!/usr/bin/env python3
"""
OS-CFAR algorithm
"""
# Standard libraries
import numpy as np
# Local libraries
import pyrads.algorithm


class OSCFAR(pyrads.algorithm.Algorithm):
    """
    Parent class for radar algorithms
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load os-cfar parameters
        self.n_dims = kwargs.get("n_dims")
        self.window_width = kwargs.get("window_width")
        self.ordered_k = kwargs.get("ordered_k")
        self.alpha = kwargs.get("alpha")
        self.n_guard_cells = kwargs.get("n_guard_cells")


    def calculate_out_shape(self):
        """
        CFAR algorithms do not alter the data dimensionality
        """
        self.out_data_shape = self.in_data_shape


    def padding(self, data):
        """
        Pad the input data with zeros

        The size of the padding is dependent on the CFAR window size
        and the number of guard cells.
        """
        pad = np.zeros(self.n_guard_cells+self.window_width)//2
        padded_data = np.hstack((pad, data, pad))
        return padded_data


    def get_window(self, data, index):
        # Construct the window around the current value
        init = index
        end = index + self.n_guard_cells + self.window_width + 1
        pre_window = data[init:init+self.window_width//2]
        post_window = data[end-self.window_width//2:end]
        window = np.hstack((pre_window, post_window))
        return window


    def run_1d(self, data):
        """
        Run the OS-CFAR in 1D
        """
        padded_data = self.padding(data)
        threshold = np.zeros_like(data)
        for index in range(data.size):
            window = self.get_window(padded_data, index)
            # Find the kth highest value in the window
            ordered_window = np.sort(window)[::-1]
            threshold[index] = ordered_window[self.ordered_k]
        # Compute object detection
        result = data*self.alpha > threshold
        return result


    def run_2d(self):
        """
        Run the OS-CFAR in 2D
        """
        pass


    def _run(self, in_data):
        if self.n_dims== 1:
            result = self.run_1d(in_data)
        elif self.n_dims==2:
            result = self.run_2d(in_data)
        return result
