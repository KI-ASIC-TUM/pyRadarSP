#!/usr/bin/env python3
"""
Module containing the base algorithm class
"""
# Standard libraries
from abc import ABC, abstractmethod
# Local libraries


class Algorithm(ABC):
    """
    Parent class for radar algorithms
    """
    def __init__(self, **kwargs):
        self.__name__ == "pyrads.Algorithm"
        self.in_data_shape = kwargs.get("in_data_shape")
        self.out_data_shape = kwargs.get("out_data_shape", None)
        if not self.out_data_shape:
            self.calculate_out_shape()

    @abstractmethod
    def calculate_out_shape(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def __call__(self):
        self.run()
