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
    NAME = "OS-CFAR"

    def __init__(self,*args,  **kwargs):
        super().__init__(*args, **kwargs)
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
        Pad the input data with zeros along the sample dimension

        It is assumed that the sample dimension is the last dimension of
        the input data.

        The size of the padding is dependent on the CFAR window size
        and the number of guard cells.
        """
        pad_size = (self.n_guard_cells+self.window_width) // 2
        if self.n_dims==1:
            pad_shape = (data.shape[-1]+2*pad_size, )
            padded_data_shape =  data.shape[:-1] + pad_shape
            padded_data = np.zeros(padded_data_shape)
            padded_data[..., pad_size:-pad_size] = data
        if self.n_dims==2:
            pad_shape = (data.shape[-2]+2*pad_size, data.shape[-1]+2*pad_size)
            # Final shape is same as input shape
            # with the last two dimensions modified with the padding
            padded_data_shape =  data.shape[:-2] + pad_shape
            padded_data = np.zeros(padded_data_shape)
            padded_data[..., pad_size:-pad_size, pad_size:-pad_size] = data
        return padded_data


    def get_window(self, data, index):
        """
        Fetch the CFAR neighbouring window for the specified data index
        """
        # Construct the window around the current value
        init = index
        end = index + self.n_guard_cells + self.window_width + 1
        # The central value and guard cells are ignored and not considered
        # neighbours fol calculating the CFAR threshold
        pre_window = data[..., init:init+self.window_width//2]
        post_window = data[... ,end-self.window_width//2:end]
        window = np.concatenate((pre_window, post_window), axis=-1)
        return window


    def run_1d(self, data):
        """
        Run the OS-CFAR in 1D
        """
        padded_data = self.padding(data)
        threshold = np.zeros_like(data)
        for index in range(data.shape[-1]):
            window = self.get_window(padded_data, index)
            # Find the kth highest value in the window
            ordered_window = np.sort(window, axis=-1)[..., ::-1]
            threshold[..., index] = ordered_window[..., self.ordered_k]
        # Compute object detection
        result = data*self.alpha > threshold
        return result


    def run_2d(self, data):
        """
        Run the OS-CFAR in 2D
        """
        padded_data = self.padding(data)
        raise NotImplementedError
        pass


    def _run(self, in_data):
        if self.n_dims== 1:
            result = self.run_1d(in_data)
        elif self.n_dims==2:
            result = self.run_2d(in_data)
        return result
