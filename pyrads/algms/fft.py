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
    NAME = "FFT"
    TYPES = ("range", "doppler", "range-doppler")

    def __init__(self, *args, **kwargs):
        # TODO: Change variable name. Type is a reserved word and could 
        # bring conflicts
        self.type = kwargs.get("type")
        if self.type not in self.TYPES:
            raise ValueError("{} not a valid FFT type".format(self.type))
        self.out_format = kwargs.get("out_format", "modulus")
        self.normalize = kwargs.get("normalize", True)
        self.unitary = kwargs.get("unitary", False)
        self.logarithmic_out = kwargs.get("logarithmic_out", False)
        self.off_bins = kwargs.get("off_bins", 0)
        self.n_real_bins = 0
        super().__init__(*args, **kwargs)


    def calculate_out_shape(self):
        """
        Calculate the shape of the FFT output

        In case of 1D FFT, the negative spectrum is removed.
        """
        self.out_data_shape = self.in_data_shape
        self.n_real_bins = (self.out_data_shape[-1] // 2) - self.off_bins
        if self.type in ("range", "range-doppler"):
            self.out_data_shape = self.in_data_shape[:-1] + (self.n_real_bins,)
            # Add a new dimension in case polar format is used for output
            if self.out_format == "modulus-phase":
                self.out_data_shape += (2,)
        elif self.type=='doppler':
            self.out_data_shape = self.in_data_shape


    def format_fft(self, fft_data):
        """
        Adjust the output FFT data to the specified format
        """
        if self.out_format == "modulus":
            formatted_fft = np.abs(fft_data)
            # Arrange data between 0 and 1
            if self.unitary:
                formatted_fft = formatted_fft - formatted_fft.min()
                formatted_fft /= formatted_fft.max()
        elif self.out_format == "complex":
            formatted_fft = fft_data
        elif self.out_format == "modulus-phase":
            mod = np.abs(fft_data)
            angle = np.angle(fft_data)
            formatted_fft = np.stack((mod, angle), axis=-1)
        else:
            raise ValueError("Invalid format: {}".format(self.out_format))

        if self.logarithmic_out:
            result = 20*np.log((formatted_fft))
            # Formatt result to only positive values
            minimum = result.min(axis=(-2, -1))
            minimum = minimum.reshape(minimum.shape+(1,1))
            result -= minimum
        else:
            result = formatted_fft
        return result


    def range_fft(self, data):
        """
        Apply FFT on last axis.
        """
        # Apply Range FFT only positive frequencies
        data = np.fft.fft(data, axis=-1)
        # Normalize the FFT
        if self.normalize:
            data /= self.in_data_shape[-1]*2
        # Remove negative spectrum
        real_data = data[...,  self.off_bins : self.off_bins+self.n_real_bins]
        return real_data


    def doppler_fft(self, data):
        """
        Apply the FFT to last two axis and generate range-Doppler map
        """
        # Apply Doppler FFT
        data = np.fft.fft(data, axis=-2)
        data = np.fft.fftshift(data, axes=-2)
        # Normalize the FFT
        if self.normalize:
            data /= self.in_data_shape[-2]*2
        return data


    def _run(self, in_data):
        """
        Take the right FFT type and return result.
        """
        if self.type=="range":
            fft_result = self.range_fft(in_data)
        elif self.type=="doppler":
            fft_result = self.doppler_fft(in_data)
        elif self.type=="range-doppler":
            fft_range = self.range_fft(in_data)
            fft_result = self.doppler_fft(fft_range)
        formatted_result = self.format_fft(fft_result)
        return formatted_result
