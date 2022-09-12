#!/usr/bin/env python3
"""
EinsumWindow algorithm.

No clue how to name it.
"""
# Standard libraries
import numpy as np
import logging
# Local libraries
import pyrads.algorithm


class Window(pyrads.algorithm.Algorithm):
    """
    Parent class for radar algorithms
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load smoothenting parameters
        # window type
        self.window_type = kwargs.get("window_type")
        self.axis = kwargs.get("axis")
        self.n_dim = len(self.in_data_shape)
        self.n_samples = self.in_data_shape[self.axis]
        self.window = self.get_window()

        if self.axis >= self.n_dim:
            logging.error("Axis {0} is greater then dimension of input shape {1}".format(self.axis, self.in_data_shape))


        self.indices_str = ''
        for i in range(97, 97+self.n_dim):
            self.indices_str += chr(i)
        self.axis_str = self.indices_str[self.axis]

        self.einsum_str = self.indices_str + ',' + self.axis_str + '->' + self.indices_str


    def calculate_out_shape(self):
        """
        Do not alter the data dimensionality
        """
        self.out_data_shape = self.in_data_shape

    def get_window(self):
        """
        Return smoothen window.
        Supported types: 
            'hann': hanning window

        """

        if self.window_type=='hann':
           window = np.hanning(self.n_samples) 
        
        # normalize window
        window = window * self.n_samples/np.sum(window)

        return window


    def _run(self, in_data):
        """
        Return smoothened data.
        Apply only on last dimension.
        """

        result = np.einsum(self.einsum_str, in_data, self.window)

        return result