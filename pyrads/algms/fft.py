"""
FFT algorithm.

Specific for our radar data.
TODO: Generalization
"""
# Standard libraries
import numpy as np
# Local libraries
import pyrads.algorithm


class FFT(pyrads.algorithm.Algorithm):
    """
    Parent class for radar algorithms
    """
    def __init__(self, **kwargs):
        self.type = kwargs.get("type")
        super().__init__(**kwargs)
        # Load smoothenting parameters
        # window type


    def calculate_out_shape(self):
        """
        Do not alter the data dimensionality
        """

        n_tx, n_rx, n_ramps, n_samples = self.in_data_shape

        if self.type=='range':
            n_fft_bins = self.in_data_shape[-1] // 2
            self.out_data_shape = (n_tx, n_rx, n_ramps, n_samples)
        elif self.type=='doppler':
            self.out_data_shape = self.in_data_shape

    def range_fft(self, data):
        """
        Apply FFT on last axis.

        """

        # Apply Range FFT only positive frequencies
        data = np.fft.fft(data, axis=-1)

        # Normalize
        y_data = data / self.in_data_shape[-1] * 2

        return data

    def doppler_fft(self, data):
    

        # Apply Doppler FFT
        data = np.fft.fft(data, axis=-2)
        data = np.fft.fftshift(data, axes=-2)

        # Normalise FFT
        y_data = data / self.in_data_shape[-2] * 2

        return y_data


    def _run(self, in_data):
        """
        Take the right FFT type and return result.
        """

        if self.type=="range":
            result = self.range_fft(in_data)
        elif self.type=="doppler":
            result = self.doppler_fft(in_data)

        return result