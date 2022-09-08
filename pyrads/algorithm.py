#!/usr/bin/env python3
"""
Module containing the base algorithm class
"""
# Standard libraries
from abc import ABC, abstractmethod
import numpy as np
# Local libraries


class Algorithm(ABC):
    """
    Parent class for radar algorithms
    """
    def __init__(self, **kwargs):
        self.in_data_shape = kwargs.get("in_data_shape")
        self.out_data_shape = kwargs.get("out_data_shape", None)
        if not self.out_data_shape:
            self.calculate_out_shape()
        self.output = np.zeros(self.out_data_shape)

    @abstractmethod
    def calculate_out_shape(self):
        pass

    @abstractmethod
    def _run(self, in_data):
        """
        Functionality of children class is defined here
        """
        pass

    def __call__(self, in_data):
        """
        Call for the algorithm class. Do not modify in children class
        """
        self.output = self._run(in_data)
        # Check dimensionality
        if self.output.shape != self.out_data_shape:
            raise ValueError(
                "Output shape {} does not match with expected shape {}"
                "".format(self.output.shape, self.out_data_shape))
        return self.output
