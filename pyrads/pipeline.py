#!/usr/bin/env python3
"""
Module containing the base pipeline class
"""
# Standard libraries
from abc import ABC, abstractmethod
import numpy as np
# Local libraries
import pyrads.algorithm


class Pipeline():
    """
    Parent class for radar algorithms
    """
    def __init__(self, chain=[], dataset=""):
        self._algorithms = []
        self._in_data = np.array([])
        self.output = np.array([])

        if len(dataset) > 0:
            self.add_data(dataset)

        if len(chain) > 0:
            self.add_chain(chain)


    def add_data(self, name):
        pass


    def __check_shape(self, in_shape):
        """
        Check that input shape matches the last element of the pipeline
        """
        # If there already are algorithms, use the last one as reference
        if len(self._algorithms) > 0:
            cur_out_data_shape = self._algorithms[-1].out_data_shape
        # Otherwise, use the input data shape
        elif self._in_data.size > 0:
            cur_out_data_shape = self._in_data.shape
        # Otherwise, ignore the check
        else:
            cur_out_data_shape = in_shape
        check = True if in_shape == cur_out_data_shape else False
        return check


    def add_chain(self, chain):
        """
        Add a chain of algorithms or pipelines
        """
        for element in chain:
            if isinstance(element, pyrads.algorithm.Algorithm):
                self.add_algorithm(element)
            elif isinstance(element, pyrads.pipeline.Pipeline):
                self.add_chain(element.__algorithms)
            else:
                raise TypeError("input type invalid")


    def add_algorithm(self, algorithm):
        """
        Add an algorithm to the processing chain
        """
        # Test that the input data format is compatible with the previous one
        assert self.__check_shape(algorithm.in_data_shape)
        # Add algorithm to the list
        self._algorithms.append(algorithm)


    def _run(self, in_data):
        # Check data shape fits the 1st algorithm in_shape
        if in_data.shape != self._algorithms[0].in_data_shape:
            raise ValueError(
                "Input shape {} does not match with input algorithm shape {}"
                "".format(in_data.shape, self._algorithms[0].in_data_shape))
        # Run the initial algorithm with the input data
        output = self._algorithms[0](in_data)
        # Run remaining algorithms iteratively
        # TODO
        return output


    def __getitem__(self, item):
        return self._algorithms[item]


    def __len__(self):
        return len(self._algorithms)


    def __call__(self, in_data):
        self.output = self._run(in_data)
        return self.output
