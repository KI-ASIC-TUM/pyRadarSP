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
        # TODO: Change variable name. Type is a reserved word and could 
        # bring conflicts
        self.type = kwargs.get("type")
        super().__init__(**kwargs)


    def calculate_out_shape(self):
        """
        In case of 1D FFT, the negative spectrum is removed
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
        normalized_data = data / self.in_data_shape[-1] * 2
        return normalized_data


    def doppler_fft(self, data):
        """
        Apply the FFT to last two axis and generate range-Doppler map
        """
        # Apply Doppler FFT
        data = np.fft.fft(data, axis=-2)
        data = np.fft.fftshift(data, axes=-2)
        # Normalize FFT
        normalized_data = data / self.in_data_shape[-2] * 2
        return normalized_data


    def _run(self, in_data):
        """
        Take the right FFT type and return result.
        """
        if self.type=="range":
            result = self.range_fft(in_data)
        elif self.type=="doppler":
            result = self.doppler_fft(in_data)
        return result
